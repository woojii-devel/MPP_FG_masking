import os
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets,get_dataset_config_names, load_dataset
import torch
from huggingface_hub import snapshot_download
# import deepchem as dc
import numpy as np
from torch.utils.data import Subset



class FilteredHFDataset(Dataset):
    def __init__(self, hf_dataset, *, max_len=512, min_len=6):
        """
        hf_dataset: HuggingFace Dataset
        max_len 초과 샘플 제외
        min_len 미만 샘플 제외
        """
        self.dataset = hf_dataset


        self.max_len = max_len
        self.min_len = min_len

    def __len__(self):
        return len(self.dataset)

    def _seq_len(self, item):
        ids = item.get("input_ids", None)
        if ids is None:
            return 0
        return int(ids.numel()) if hasattr(ids, "numel") else len(ids)

    def __getitem__(self, idx):
        n = len(self.dataset)
        j = idx
        while j < n:
            item = self.dataset[j]
            L = self._seq_len(item)

            if (L <= self.max_len) and (L >= self.min_len):
                # 숫자형 시퀀스는 tensor 보장
                if not isinstance(item["input_ids"], torch.Tensor):
                    item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
                else:
                    item["input_ids"] = item["input_ids"].long()

                if "token_atom" in item:
                    if not isinstance(item["token_atom"], torch.Tensor):
                        item["token_atom"] = torch.tensor(item["token_atom"], dtype=torch.long)
                    else:
                        item["token_atom"] = item["token_atom"].long()

                # fg_atoms는 가변 길이 nested list일 수 있으므로 list[Tensor]로 정리
                if "fg_atoms" in item:
                    fg_atoms = item["fg_atoms"]
                    if isinstance(fg_atoms, torch.Tensor):
                        # 혹시 2D tensor로 이미 들어오면 행별로 분리
                        item["fg_atoms"] = [row.long() for row in fg_atoms]
                    else:
                        item["fg_atoms"] = [
                            x.long() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.long)
                            for x in fg_atoms
                        ]

                return item

            j += 1

        raise IndexError(
            f"No valid sample from index {idx} to end "
            f"(filtered by length: min_len={self.min_len}, max_len={self.max_len})"
        )


class ShardDatasetLoader:
    def __init__(
        self,
        repo_id,
        max_len,
        *,
        train_ratio=0.98,
        val_ratio=0.01,
        seed=42,
        min_len=6,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_len = max_len
        self.min_len = min_len
        self.repo_id = repo_id
    # def _load_all_shards(self):
    #     shard_paths = sorted(
    #         os.path.join(self.shard_dir, d)
    #         for d in os.listdir(self.shard_dir)
    #         if d.startswith("shard_")
    #     )

    #     if len(shard_paths) == 0:
    #         raise ValueError(f"No shard directories found in: {self.shard_dir}")

    #     ds_list = []
    #     for p in shard_paths:
    #         print(f"Loading {p}")
    #         ds = load_from_disk(p)
    #         ds_list.append(ds)

    #     merged = concatenate_datasets(ds_list)
    #     print(f"Merged dataset size: {len(merged)}")
    #     return merged


    def _load_all_shards(self):
        print("Downloading dataset from HF...")

        local_path = snapshot_download(repo_id=self.repo_id,
                                       repo_type="dataset")

        shard_paths = sorted(
            os.path.join(local_path, d)
            for d in os.listdir(local_path)
            if d.startswith("shard_")
        )

        print("Shards:", shard_paths)

        ds_list = []
        for p in shard_paths:
            print(f"Loading {p}")
            ds = load_from_disk(p)
            ds_list.append(ds)

        merged = concatenate_datasets(ds_list)
        print(f"Merged dataset size: {len(merged)}")

        return merged
    
    def _filter_by_length(self, dataset):
        def length_ok(example):
            L = len(example["input_ids"])
            return (self.min_len <= L <= self.max_len)

        filtered = dataset.filter(length_ok)
        print(
            f"Filtered dataset size: {len(filtered)} "
            f"(min_len={self.min_len}, max_len={self.max_len})"
        )
        return filtered

    # def scaffold_split_dataset(self,dataset, smiles_list):
    #     """
    #     dataset: PyTorch Dataset (self 같은거)
    #     smiles_list: 각 데이터에 대응되는 SMILES list
    #     """

    #     # dummy dataset 생성 (DeepChem용)
    #     dc_dataset = dc.data.NumpyDataset(
    #         X=np.zeros((len(smiles_list), 1)),
    #         y=np.zeros(len(smiles_list)),   # pretrain이라 label 필요 없음
    #         ids=np.array(smiles_list)
    #     )

    #     splitter = dc.splits.ScaffoldSplitter()

    #     train_idx, val_idx, test_idx = splitter.split(
    #         dc_dataset,
    #         frac_train=self.train_ratio,
    #         frac_valid=self.val_ratio,
    #         frac_test=1 - self.train_ratio - self.val_ratio
    #     )

    #     train_ds = Subset(dataset, train_idx)
    #     val_ds   = Subset(dataset, val_idx)
    #     test_ds  = Subset(dataset, test_idx)

    #     print("Scaffold split (pretrain):")
    #     print(f"train: {len(train_ds)}")
    #     print(f"val: {len(val_ds)}")
    #     print(f"test: {len(test_ds)}")

    #     return train_ds, val_ds, test_ds
    
    def _split_dataset(self, dataset):
        split_1 = dataset.train_test_split(
            test_size=1 - self.train_ratio,
            seed=self.seed
        )

        train_ds = split_1["train"]
        rest_ds = split_1["test"]

        val_portion_in_rest = self.val_ratio / (1 - self.train_ratio)

        split_2 = rest_ds.train_test_split(
            test_size=val_portion_in_rest,
            seed=self.seed
        )

        test_ds = split_2["train"]
        val_ds = split_2["test"]

        print("Split sizes:")
        print(f"  train: {len(train_ds)}")
        print(f"  val  : {len(val_ds)}")
        print(f"  test : {len(test_ds)}")

        return train_ds, val_ds, test_ds

    def dataset(self):
        merged = self._load_all_shards()
        filtered = self._filter_by_length(merged)
        # smiles_list = filtered["smiles"]
        train_hf, val_hf, test_hf = self._split_dataset(filtered)
        

        train = FilteredHFDataset(
            train_hf, max_len=self.max_len, min_len=self.min_len
        )
        val = FilteredHFDataset(
            val_hf, max_len=self.max_len, min_len=self.min_len
        )
        test = FilteredHFDataset(
            test_hf, max_len=self.max_len, min_len=self.min_len
        )

        return train, val, test