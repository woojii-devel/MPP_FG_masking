import os, sys, re, json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit import RDConfig
from pytoda.smiles.smiles_language import SMILESTokenizer
from datasets import Dataset

# EFGs import
efg_dir = os.path.join(RDConfig.RDContribDir, "efgs")
sys.path.insert(0, efg_dir)
import efgs

# -------------------------
# Globals for workers
# -------------------------
# 워커 프로세스들이 공유할 전역 변수
tokenizer = None
atom_tokens_global = None

def init_worker(vocab_path, tokenizer_kwargs, atom_tokens_set):
    """프로세스가 생성될 때 딱 한 번 실행되어 토크나이저를 로드합니다."""
    global tokenizer, atom_tokens_global
    tokenizer = SMILESTokenizer(vocab_file=vocab_path, **tokenizer_kwargs)
    atom_tokens_global = atom_tokens_set

# -------------------------
# Utils
# -------------------------
def get_smiles_from_txt(path):
    smiles = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2: continue
            smiles.append(parts[1])
    return smiles

def load_vocab_dict(vocab_path):
    with open(vocab_path) as f:
        return json.load(f)

def build_atom_token_set(vocab):
    atom_set = set()

    for tok in vocab.keys():

        if tok.startswith("[") and tok.endswith("]"):
            atom_set.add(tok)

        elif re.fullmatch(r"[A-Z][a-z]?", tok):
            atom_set.add(tok)

        elif re.fullmatch(r"[bcnops]", tok):
            atom_set.add(tok)

    return atom_set

def smiles_with_output_order(mol, canonical=True, isomeric=True):
    # 속성 생성을 위해 복사본 사용
    m2 = Chem.Mol(mol)
    smi = Chem.MolToSmiles(m2, canonical=canonical, isomericSmiles=isomeric)
    if not m2.HasProp("_smilesAtomOutputOrder"):
        return None, None
    prop = m2.GetProp("_smilesAtomOutputOrder")
    try:
        order = json.loads(prop)
        return smi, [int(v) for v in order]
    except:
        nums = re.findall(r"\d+", prop)
        return (smi, [int(x) for x in nums]) if nums else (None, None)

# -------------------------
# Core Logic
# -------------------------
def build_record(smiles):
    global tokenizer, atom_tokens_global
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    
    # EFG 에러 시 빈 리스트로 넘어가게 수정 (데이터 손실 방지)
    try:
        _, fg_raw, _, _ = efgs.get_dec_fgs(mol)
        fg_atom_sets = [sorted(list(s)) for s in fg_raw]
    except Exception as e:
        # print("fg error:", e)
        fg_atom_sets = []
        
    smi, order = smiles_with_output_order(mol, canonical=True)
    if smi is None:
        return None

    try:
        smi_tokens = tokenizer.smiles_tokenizer(smi)
        token_ids = tokenizer.smiles_to_token_indexes(smi)
        token_ids = token_ids.tolist()
        token_atom = []
        atom_i = 0
        n_atoms = mol.GetNumAtoms()

        for tok in smi_tokens:
            if tok in atom_tokens_global:
                if atom_i >= n_atoms: return None
                token_atom.append(order[atom_i])
                atom_i += 1
            else:
                token_atom.append(-1)


        if atom_i != n_atoms: return None
        # 스페셜 토큰 추가
        token_ids = [tokenizer.start_index] + token_ids + [tokenizer.stop_index]
        token_atom = [-1] + token_atom + [-1]
        smi_tokens = [tokenizer.start_token]+ list(smi_tokens) + [tokenizer.stop_token]
        if not len(token_ids) == len(token_atom) == len(smi_tokens):
            return None
    
    
        return {
            "smiles": smi,
            "input_ids": token_ids,
            "token_atom": token_atom,
            "fg_atoms": fg_atom_sets,
            "smi_tokens":smi_tokens
        }
    except Exception as e:
        print("ERROR:", e)
        return None

def worker(smiles):
    return build_record(smiles)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    INPUT_PATH = "Vast_ai_folder/pubchem-10m.txt"
    OUTPUT_DIR = "Vast_ai_folder/arrow_shards"
    VOCAB_PATH = "Vast_ai_folder/smiles_tokenizer_pytoda/vocab.json"
    SHARD_SIZE = 1_000_000

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")
    
    vocab = load_vocab_dict(VOCAB_PATH)
    atom_tokens_set = build_atom_token_set(vocab)
    smiles_list = get_smiles_from_txt(INPUT_PATH)
    
    print(f"Loaded {len(smiles_list)} SMILES. Starting pool...")
    
    n_proc = max(1, cpu_count() - 1)
    buffer = []
    shard_idx = 0
    success_count = 0

    # Pool 생성 시 initializer 설정이 핵심입니다.
    with Pool(processes=n_proc, 
              initializer=init_worker, 
              initargs=(VOCAB_PATH, {'canonical': True, 'sanitize': True}, atom_tokens_set)) as pool:
        
        it = pool.imap_unordered(worker, smiles_list, chunksize=1000)

        pbar = tqdm(it, total=len(smiles_list), desc="Processing")
        for rec in pbar:

            if rec is None:
                continue
            
            buffer.append(rec)
            success_count += 1
            
            # 실시간 성공 개수 확인용
            if success_count % 100 == 0:
                pbar.set_postfix({"ok": success_count, "shards": shard_idx})

            if len(buffer) >= SHARD_SIZE:
                ds = Dataset.from_list(buffer)
                path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}")
                ds.save_to_disk(path)
                tqdm.write(f"💾 Saved shard_{shard_idx:05d}. Total ok: {success_count}")
                buffer = []
                shard_idx += 1

        if buffer:
            Dataset.from_list(buffer).save_to_disk(os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}"))

    print("All done!")