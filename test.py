from Filtered_Dataset import ShardDatasetLoader

loader = ShardDatasetLoader(
            repo_id = 'Zaeus/MPP_pt_data',
            train_ratio=0.98,
            val_ratio=0.01,
            seed=42,
            max_len=512,
            min_len=6,
        )

train_ds, val_ds, test_ds = loader.dataset()