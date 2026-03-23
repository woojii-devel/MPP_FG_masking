from Filtered_Dataset import ShardDatasetLoader

loader = ShardDatasetLoader(
            repo_id = 'Zaeus/MPP_pt_data',
            train_ratio=0.98,
            val_ratio=0.01,
            seed=42,
            max_len=512,
            min_len=6,
        )
print("split start")
train_ds, val_ds, test_ds = loader.dataset()
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))
print("split finish")