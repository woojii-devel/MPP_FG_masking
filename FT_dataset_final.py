from datasets import load_from_disk
import torch
import random
import numpy as np
import deepchem as dc

class FT_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seed, task, max_len=512,min_len=6):
        self.ds = load_from_disk(data_path)
        self.seed = seed
        self.task = task
        self.min_len = min_len
        self.max_len = max_len

        self.is_multilabel = task in ["tox21", "muv", "sider"]
        self.is_regression = task in ["esol", "freesolv", "lipo"]
        def valid_example(example):
            input_ids = example["input_ids"]
            if input_ids is None: return False
            total_len = len(input_ids)
            return self.min_len <= total_len <= self.max_len

        before_len = len(self.ds)
        self.ds = self.ds.filter(valid_example,num_proc = 8)
        after_len = len(self.ds)

        print(f"[{task}] filtered short & long samples: {before_len - after_len} removed, {after_len} kept")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        labels = row['labels']
        input_ids = torch.tensor(row["input_ids"], dtype=torch.long)

        if self.is_multilabel:
            labels = torch.tensor(labels, dtype=torch.float32)
        elif self.is_regression:
            labels = torch.tensor(labels[0], dtype=torch.float32)
        else:
            labels = torch.tensor(labels[0], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels
        }
        
    def split_dataset(self, tr_ratio=0.8, val_ratio=0.1, method="random"):
        if method == "random":
            idx = list(range(len(self.ds)))

            split_1 = int(tr_ratio * len(idx))
            split_2 = int((tr_ratio + val_ratio) * len(idx))

            random.seed(self.seed)
            random.shuffle(idx)

            train_idx = idx[:split_1]
            val_idx = idx[split_1:split_2]
            test_idx = idx[split_2:]

            train_ds = torch.utils.data.Subset(self, train_idx)
            val_ds = torch.utils.data.Subset(self, val_idx)
            test_ds = torch.utils.data.Subset(self, test_idx)

            print("Random split:")
            print(f"train: {len(train_ds)}")
            print(f"val: {len(val_ds)}")
            print(f"test: {len(test_ds)}")

            return train_ds, val_ds, test_ds

        elif method == "scaffold":
            smiles_list = self.ds["smiles"]
            labels = self.ds["labels"]

            dataset = dc.data.NumpyDataset(
                X=np.zeros((len(smiles_list), 1)),
                y=np.array(labels),
                ids=np.array(smiles_list)
            )

            splitter = dc.splits.ScaffoldSplitter()

            train_idx, val_idx, test_idx = splitter.split(
                dataset,
                frac_train=tr_ratio,
                frac_valid=val_ratio,
                frac_test=1 - tr_ratio - val_ratio
            )

            train_ds = torch.utils.data.Subset(self, train_idx)
            val_ds = torch.utils.data.Subset(self, val_idx)
            test_ds = torch.utils.data.Subset(self, test_idx)

            print("Scaffold split:")
            print(f"train: {len(train_ds)}")
            print(f"val: {len(val_ds)}")
            print(f"test: {len(test_ds)}")

            return train_ds, val_ds, test_ds

        else:
            raise ValueError(f"Unknown split method: {method}. Use 'random' or 'scaffold'.")