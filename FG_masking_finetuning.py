from arguments import get_args
import numpy as np
from FG_masking_pretraining import RoBERTa_FG
import torch.distributed as dist
import os, torch 
from datetime import datetime
from pathlib import Path
from transformers import set_seed
import re
import numpy as np
from FT_dataset_final import FT_Dataset

def compute_seed_average(result_file, metric="auc"):
    values = []

    if metric.lower() == "auc":
        pattern = r"auc:\s*([0-9.]+)"
        name = "AUC"
    elif metric.lower() == "rmse":
        pattern = r"rmse:\s*([0-9.]+)"
        name = "RMSE"
    else:
        raise ValueError("metric must be 'auc' or 'rmse'")

    with open(result_file, "r") as f:
        for line in f:
            m = re.search(pattern, line.lower())
            if m:
                values.append(float(m.group(1)))

    if len(values) == 0:
        raise ValueError(f"No {name} values found")

    mean = np.mean(values)
    std = np.std(values)

    with open(result_file, "a") as f:
        f.write("\n===== Result =====\n")
        f.write(f"Mean {name}: {mean:.6f}\n")
        f.write(f"Std  {name}: {std:.6f}\n")

    print(f"Mean {name}: {mean:.6f}")
    print(f"Std  {name}: {std:.6f}")
    
def main():

    args = get_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    TASKS = ['esol','lipo','freesolv','muv','hiv']
    SEEDS = args.seeds

    date = datetime.now().strftime("%Y%m%d_%H%M")
    EXP_DIR = Path("./experiments") / f"{date}_FGmasking_final"
    CV_DIR = EXP_DIR / "seed_results"

    if local_rank == 0:
        CV_DIR.mkdir(parents=True, exist_ok=True)

    dist.barrier()

    for task in TASKS:

        result_file = CV_DIR / f"{task}.txt"

        if local_rank == 0:
            print(f"\n🚀 Starting Task: {task}")
            if result_file.exists():
                result_file.unlink()

        dist.barrier()
        ds = FT_Dataset(
            data_path=Path(args.ft_data_path) / f"{task}_dataset",
            seed=42,   # 고정 split seed
            task=task,
            max_len=512,
            min_len=6
        )
        train_ds, val_ds, test_ds = ds.split_dataset(method="random")

        for seed in SEEDS:
            
            if local_rank == 0:
                print(f"Running seed {seed}")

            args.seed = seed
            args.task_name = task
            args.result_file = str(result_file)
            set_seed(seed)
            model = RoBERTa_FG(args)

            model.finetuning(
                pretrained_ckpt_path=args.ft_model_path,
                task=task,
                train_ds= train_ds,
                val_ds = val_ds,
                test_ds=test_ds
            )

            dist.barrier()

        if local_rank == 0:
            if task in ['esol','freesolv','lipo']:
                compute_seed_average(result_file, 'rmse')
            else:
                compute_seed_average(result_file, 'auc')

        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    
    main()