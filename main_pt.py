
from pathlib import Path
from datetime import datetime
from FG_masking_pretraining import RoBERTa_FG
from arguments import get_args
import gc
import os, torch
import torch.distributed as dist
from huggingface_hub import login, upload_folder
from dotenv import load_dotenv
date = datetime.now().strftime("%Y%m%d")
args = get_args()


print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LOCAL_RANK =", os.environ.get("LOCAL_RANK"), "RANK =", os.environ.get("RANK"), "WORLD_SIZE =", os.environ.get("WORLD_SIZE"))
print("dist.is_available =", dist.is_available(), "dist.is_initialized =", dist.is_initialized())
print("cuda device_count =", torch.cuda.device_count(), "current =", torch.cuda.current_device())

print("[GPU] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[GPU] device_count:", torch.cuda.device_count())
print("[GPU] names:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

args.mlm_probability = 0.15
args.pt_masking_method = 'hybrid_masking'
args.max_fg_ratio = 0.10

model = RoBERTa_FG(args)
trainer = model.pretraining()


if dist.is_initialized():
    dist.barrier()


if (not dist.is_initialized()) or dist.get_rank() == 0:
    best_dir = Path("best_model")
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(best_dir)

    print("saved best model to:", best_dir)

    env_path = Path(".env")
    load_dotenv(env_path)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in .env")

    login(token=hf_token)

    upload_folder(
        folder_path=str(best_dir),
        repo_id="Zaeus/hybrid_masking_15per",
        repo_type="model"
    )