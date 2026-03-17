from transformers import DataCollatorForLanguageModeling
import torch
import os, sys, re, random
from rdkit import Chem
from rdkit import RDConfig
from typing import Optional
efg_dir = os.path.join(RDConfig.RDContribDir, "efgs")
sys.path.insert(0, efg_dir)
import efgs
from pytoda.smiles.smiles_language import SMILESTokenizer

class FG_masking_collator:
    
    def __init__(self, tokenizer, mlm_probability=0.25,max_length = 512,mode='fg'):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_id = tokenizer.token_to_index['<MASK>']
        self.pad_id = tokenizer.padding_index
        self.bos_id = tokenizer.start_index
        self.eos_id = tokenizer.stop_index
        self.max_length = max_length
        self.mode = mode
        self.last_debug = None
    ####
    # def find_seed_positions(self, token_atom, fg_atom_set):
    #     seed_positions = {
    #     i for i, atom_id in enumerate(token_atom.tolist())
    #     if atom_id >= 0 and atom_id in fg_atom_set
    #     } 
    #     return seed_positions
    
    def find_seed_mask(self, token_atom: torch.Tensor, fg_atom_set: torch.Tensor):
        """
        token_atom: (L,) long, -1 = non-atom
        fg_atom_set: (K,) long
        return: (L,) bool
        atom이면서 fg에 포함되면 true
        """
        if fg_atom_set.numel() == 0:
            return torch.zeros_like(token_atom, dtype=torch.bool)

        valid_atom = token_atom >= 0

        seed_mask = valid_atom & torch.isin(token_atom, fg_atom_set)
        return seed_mask
    
    def span_mask(self, smi_tokens, input_ids, seed_mask: torch.Tensor):
        '''
        양옆의 결합이거나 브랜치, 링을 나타내는 문자열 위치를 반환
        '''
        mask_positions = seed_mask.clone()

        structural_tokens = {"=", "#", "-", "/", "\\", "(", ")"}
        ring_pattern = re.compile(r"%\d\d|\d")

        changed = True
        while changed:
            changed = False
            for i, tok in enumerate(smi_tokens):
                if mask_positions[i]:
                    continue

                if (
                    ((i > 0 and mask_positions[i-1]) or
                    (i < len(mask_positions)-1 and mask_positions[i+1]))
                    and (tok in structural_tokens or ring_pattern.fullmatch(tok))
                ):
                    mask_positions[i] = True
                    changed = True

        # special tokens 제거
        mask_positions[input_ids == self.bos_id] = False
        mask_positions[input_ids == self.eos_id] = False
        mask_positions[input_ids == self.pad_id] = False

        return mask_positions
    
    def random_masking_k(self, valid: torch.Tensor, k: int) -> torch.Tensor:
        """
        valid(True) 중에서 정확히 k개를 True로 선택
        """
        if k <= 0:
            return torch.zeros_like(valid, dtype=torch.bool)

        valid_idx = valid.nonzero(as_tuple=True)[0]

        if valid_idx.numel() <= k:
            # valid가 k보다 적으면 전부 선택
            return valid.clone()

        perm = torch.randperm(valid_idx.numel(), device=valid.device)[:k]
        keep = valid_idx[perm]

        mask_pos = torch.zeros_like(valid, dtype=torch.bool)
        mask_pos[keep] = True
        return mask_pos



    def subsample_single_fg_span(
        self,
        mask_pos: torch.Tensor,
        k: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        단일 FG에 대한 mask_pos(True 구간이 하나의 연속 span이라고 가정)에서
        길이 k인 연속 sub-span을 랜덤하게 선택하여 반환.

        mask_pos: (L,) bool
        k: 선택할 span 길이
        """

        device = mask_pos.device
        out = torch.zeros_like(mask_pos, dtype=torch.bool)

        if k <= 0:
            return out

        true_idx = mask_pos.nonzero(as_tuple=True)[0]

        if true_idx.numel() == 0:
            return out

        start = int(true_idx[0].item())
        end = int(true_idx[-1].item())

        fg_len = end - start + 1

        # FG 길이가 k보다 작거나 같으면 그대로 반환
        if fg_len <= k:
            out[start:end+1] = True
            return out

        # 가능한 시작점 범위
        max_start = end - k + 1

        # start ~ max_start 사이에서 랜덤 시작점 선택
        rand_start = torch.randint(
            start,
            max_start + 1,
            (1,),
            device=device,
            generator=generator,
        ).item()

        out[rand_start:rand_start + k] = True

        return out
        # mask_pos = random_mask & (torch.rand(input_ids.shape) < 0.8)
        # labels[mask_pos] = input_ids[mask_pos].clone()
        # input_ids[mask_pos] = self.mask_id
        # return input_ids, labels
    def get_mask_pos_fg_only(self, token_atom, fg_atoms, smi_tokens,
                    input_ids, valid, valid_len,max_fg_ratio):

        if valid_len == 0:
            zero = torch.zeros_like(input_ids, dtype=torch.bool)
            return zero, zero, zero

        device = input_ids.device

        # ---------------------------
        # 1. target 설정
        # ---------------------------
        total_k = max(int(round(self.mlm_probability * valid_len)), 1)
        max_fg_k = int(round(max_fg_ratio*valid_len))
        # max_allowed = int(round(0.30 * valid_len))  # 너무 큰 FG 제한

        fg_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        current = 0

        # ---------------------------
        # 2. FG 우선 선택
        # ---------------------------
        if len(fg_atoms) > 0:

            order = torch.randperm(len(fg_atoms), device=device)

            for j in order:
                
                if current >= max_fg_k:
                    break
                fg_set = fg_atoms[j].to(device)

                # atom 기반 seed
                seed_mask = self.find_seed_mask(token_atom, fg_set)

                # span 확장
                fg_mask_j = self.span_mask(smi_tokens, input_ids, seed_mask)

                fg_mask_j &= valid  # valid 영역만

                fg_size = fg_mask_j.sum().item()

                if fg_size == 0:
                    continue

                # ---- 너무 큰 FG 처리 ----
                if fg_size > max_fg_k:
                    remain = max_fg_k - current
                    if remain <= 0:
                        break
                    fg_mask_j = self.subsample_single_fg_span(fg_mask_j, remain)
                    fg_mask |= fg_mask_j
                    current += fg_mask_j.sum().item()
                    break

                # ---- 일반 FG ----
                if current + fg_size <= max_fg_k:
                    fg_mask |= fg_mask_j
                    current += fg_size

        mask_pos = fg_mask 

        return mask_pos, fg_mask

    def get_mask_pos(self, token_atom, fg_atoms, smi_tokens,
                    input_ids, valid, valid_len,max_fg_ratio):

        if valid_len == 0:
            zero = torch.zeros_like(input_ids, dtype=torch.bool)
            return zero, zero, zero

        device = input_ids.device

        # ---------------------------
        # 1. target 설정
        # ---------------------------
        total_k = max(int(round(self.mlm_probability * valid_len)), 1)
        max_fg_k = int(round(max_fg_ratio*valid_len))
        # max_allowed = int(round(0.30 * valid_len))  # 너무 큰 FG 제한

        fg_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        current = 0

        # ---------------------------
        # 2. FG 우선 선택
        # ---------------------------
        if len(fg_atoms) > 0:

            order = torch.randperm(len(fg_atoms), device=device)

            for j in order:
                
                if current >= max_fg_k:
                    break
                fg_set = fg_atoms[j].to(device)

                # atom 기반 seed
                seed_mask = self.find_seed_mask(token_atom, fg_set)

                # span 확장
                fg_mask_j = self.span_mask(smi_tokens, input_ids, seed_mask)

                fg_mask_j &= valid  # valid 영역만

                fg_size = fg_mask_j.sum().item()

                if fg_size == 0:
                    continue

                # ---- 너무 큰 FG 처리 ----
                if fg_size > max_fg_k:
                    remain = max_fg_k - current
                    if remain <= 0:
                        break
                    fg_mask_j = self.subsample_single_fg_span(fg_mask_j, remain)
                    fg_mask |= fg_mask_j
                    current += fg_mask_j.sum().item()
                    break

                # ---- 일반 FG ----
                if current + fg_size <= max_fg_k:
                    fg_mask |= fg_mask_j
                    current += fg_size



        # ---------------------------
        # 3. 부족분 랜덤 마스킹
        # ---------------------------
        remain_k = total_k - current

        if remain_k > 0:
            rd_valid = valid & (~fg_mask)
            rd_mask = self.random_masking_k(rd_valid, remain_k)
        else:
            rd_mask = torch.zeros_like(valid)

        mask_pos = fg_mask | rd_mask

        return mask_pos, fg_mask, rd_mask
        
        
    def apply_roberta_masking(self, input_ids, labels, mask_pos):
        device = input_ids.device
        rand = torch.rand(input_ids.shape, device=device)

        # 80% → MASK
        mask_mask = mask_pos & (rand < 0.8)

        # 10% → random token
        random_mask = mask_pos & (rand >= 0.8) & (rand < 0.9)

        # labels 설정
        labels[mask_pos] = input_ids[mask_pos].clone()

        # MASK 적용
        input_ids[mask_mask] = self.mask_id

        # random token (special 제외)
        vocab_size = len(self.tokenizer.token_to_index)
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        valid_vocab = [i for i in range(vocab_size) if i not in special_ids]

        random_tokens = torch.tensor(
            random.choices(valid_vocab, k=input_ids.numel()),
            device=device
        ).view_as(input_ids)

        input_ids[random_mask] = random_tokens[random_mask]

        return input_ids, labels    
    
    # ---------- collator entry ----------
    def __call__(self,batch):

        smi_tokens_batch = [b['smi_tokens'] for b in batch]
        input_ids_batch = [b["input_ids"] for b in batch]
        fg_atoms_batch = [b["fg_atoms"] for b in batch]
        token_atom_batch = [b["token_atom"] for b in batch]
        max_len = max(len(ids) for ids in input_ids_batch)

        padded_ids = []
        padded_mask = []
        labels_batch = []
################for debug
        # fg_total = 0  
        # rd_total = 0
        # valid_total = 0
        # used_fg_flags = [] 
        # masked_counts = [] 
#############        

        for smi_tokens, input_ids, fg_atoms, token_atom in zip(smi_tokens_batch,input_ids_batch, fg_atoms_batch, token_atom_batch):
            used_fg = True
            input_ids = input_ids.clone()
            token_atom = token_atom.to(input_ids.device)
            
            ####  debug###########
            assert len(smi_tokens) == len(input_ids), (
                f"len(smi_tokens)={len(smi_tokens)} != len(input_ids)={len(input_ids)}"
            )
            assert token_atom.shape[0] == input_ids.shape[0], (
                f"len(token_atom)={token_atom.shape[0]} != len(input_ids)={input_ids.shape[0]}"
            )

            
            ###################
            labels = torch.full_like(input_ids, -100)
            num_fg = len(fg_atoms)
            
            valid = (
                (input_ids != self.pad_id) &
                (input_ids != self.bos_id) &
                (input_ids != self.eos_id)
            )
            valid_len = valid.sum().item()
            if self.mode == 'random':
                total_k = int(round(self.mlm_probability * valid_len))
                total_k = max(total_k, 1)

                mask_pos = self.random_masking_k(valid, total_k)

                input_ids, labels = self.apply_roberta_masking(input_ids, labels, mask_pos)
            
            elif self.mode == 'hybrid_masking':
                if num_fg == 0:
                    used_fg = False   #for debug
                    # input_ids, labels = self.random_masking(input_ids)
                    total_k = int(round(self.mlm_probability * valid_len))
                    total_k = max(total_k, 1)

                    mask_pos = self.random_masking_k(valid, total_k)
                    mask_pos_fg = torch.zeros_like(input_ids, dtype=torch.bool) #for debug
                    mask_pos_rd = mask_pos  #for debug

                else:
                    mask_pos,mask_pos_fg, mask_pos_rd = self.get_mask_pos(token_atom, fg_atoms, smi_tokens, input_ids,
                                                                          valid,valid_len, max_fg_ratio=0.15)
                
                labels[mask_pos] = input_ids[mask_pos].clone()
                input_ids[mask_pos] = self.mask_id
            # ===== 여기 추가 4: mask shape 검사 =====
                assert mask_pos.shape == input_ids.shape, (
                    f"mask_pos.shape={mask_pos.shape} != input_ids.shape={input_ids.shape}"
                )
            elif self.mode == 'fg_only':
                if num_fg == 0:
                    used_fg = False   #for debug
                    total_k = int(round(self.mlm_probability * valid_len))
                    total_k = max(total_k, 1)

                    mask_pos = self.random_masking_k(valid, total_k)
                    # mask_pos_fg = torch.zeros_like(input_ids, dtype=torch.bool) #for debug
                    # mask_pos_rd = mask_pos  #for debug

                else:
                    mask_pos,mask_pos_fg  = self.get_mask_pos_fg_only(token_atom, fg_atoms, smi_tokens, input_ids,
                                                                          valid,valid_len, max_fg_ratio=self.mlm_probability)
                    # mask_pos_rd = torch.zeros_like(input_ids, dtype=torch.bool) #for debug)
                labels[mask_pos] = input_ids[mask_pos].clone()
                input_ids[mask_pos] = self.mask_id
                assert mask_pos.shape == input_ids.shape, (
                    f"mask_pos.shape={mask_pos.shape} != input_ids.shape={input_ids.shape}"
                )
            
            ###############
            
            # DEBUG CHECK 1
                # if mask_pos.any():
                #     max_idx = mask_pos.nonzero(as_tuple=True)[0].max().item()
                #     assert max_idx < input_ids.shape[0], f"mask index overflow {max_idx} >= {input_ids.shape[0]}"

                assert input_ids.min() >= 0, f"negative input id {input_ids.min()}"
                assert input_ids.max() < len(self.tokenizer.token_to_index), (
                    f"input id overflow {input_ids.max()} >= vocab {len(self.tokenizer.token_to_index)}"
                )
            ###########
            
            # ===== 여기 추가 5: 최종 길이 검사 =====
            assert len(input_ids) <= max_len, (
                f"len(input_ids)={len(input_ids)} > max_len={max_len}"
            )
            assert len(labels) == len(input_ids), (
                f"len(labels)={len(labels)} != len(input_ids)={len(input_ids)}"
            )
            ###########
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids,torch.full((pad_len,), self.pad_id, dtype=input_ids.dtype)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype = labels.dtype)])    
            padded_ids.append(input_ids)
            padded_mask.append((input_ids != self.pad_id).long())#fixed
            labels_batch.append(labels)    
            
            # print("max id:", input_ids.max().item())     
#########################################
            # for debug
            # fg_total += mask_pos_fg.sum().item()
            # rd_total += mask_pos_rd.sum().item()
            # valid_total += valid_len
            # masked = input_ids.eq(self.mask_id)
            # masked_counts.append(int(masked.sum().item()))
            # used_fg_flags.append(bool(used_fg))
    
    
        
            # #for debug
            # fg_ratio = fg_total / max(valid_total, 1)
            # rd_ratio = rd_total / max(valid_total, 1)
##############################################   
        out = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
            "labels": torch.stack(labels_batch),

            # 디버그용 필드
            # "used_fg": torch.tensor(used_fg_flags, dtype=torch.bool),
            # "masked_count": torch.tensor(masked_counts, dtype=torch.long),
            # "fg_ratio": torch.tensor(fg_ratio),
            # "rd_ratio": torch.tensor(rd_ratio)
        }

        # if (out["labels"] == -100).all(dim=1).any():
        #         print("[WARNING] some samples have no masked tokens")
        # # ===== 여기 추가 6: 배치 최종 shape 검사 =====
        # assert out["input_ids"].shape == out["attention_mask"].shape == out["labels"].shape, (
        #     f"shape mismatch: input_ids={out['input_ids'].shape}, "
        #     f"attention_mask={out['attention_mask'].shape}, labels={out['labels'].shape}"
        # )

        # # print("batch seq_len:", out["input_ids"].shape[1])

        # if out["input_ids"].shape[1] > self.max_length:
        #     raise RuntimeError(
        #         f"batch seq_len overflow: {out['input_ids'].shape[1]} > {self.max_length}"
        #     )
        # ###########for debug
        # valid = out["attention_mask"].bool()
        # masked = out["input_ids"].eq(self.mask_id) & valid
        # ratio = (masked.sum(dim=1).float() / valid.sum(dim=1).float())
        # # print("mask_ratio batch mean/min/max:",
        # #     ratio.mean().item(), ratio.min().item(), ratio.max().item())
        # # print("max input id:", out["input_ids"].max().item())
        # # print("min input id:", out["input_ids"].min().item())
        # # print("vocab_size:", len(self.tokenizer.token_to_index))
        # # print("max label:", out["labels"].max().item())
        # # print("min label:", out["labels"].min().item())
        # # print("seq_len:", out["input_ids"].shape[1])
        
        # seq_len = out["input_ids"].shape[1]
        # if seq_len > 512:
        #     print("BAD LENGTH:", seq_len)
            
        # if out["input_ids"].max() >= len(self.tokenizer.token_to_index):
        #     print("BAD SAMPLE DETECTED")
        #     print("max input id:", out["input_ids"].max())
        #     print("vocab:", len(self.tokenizer.token_to_index))
        #     print("smi_tokens:", smi_tokens)
        #     raise RuntimeError("input id overflow")

        # bad_labels = out["labels"][(out["labels"] != -100) & (
        #     (out["labels"] < 0) | (out["labels"] >= len(self.tokenizer.token_to_index))
        # )]
        # if bad_labels.numel() > 0:
        #     print("BAD LABELS:", bad_labels[:20])
        #     raise RuntimeError("label id overflow")
        ###############################

        # self.last_debug = {
        #     "shape_input_ids": tuple(out["input_ids"].shape),
        #     "shape_attention_mask": tuple(out["attention_mask"].shape),
        #     "shape_labels": tuple(out["labels"].shape),
        #     "max_token_id": out["input_ids"].max().item(),
        #     "min_token_id": out["input_ids"].min().item(),
        #     "attention_mask_dtype": str(out["attention_mask"].dtype),
        #     "attention_mask_unique": torch.unique(out["attention_mask"]).tolist(),
        #     "labels_max": out["labels"].max().item(),
        #     "labels_min": out["labels"].min().item(),
        #     "raw_input_lengths": [len(x) for x in input_ids_batch],
        # }
        return out   
        
        