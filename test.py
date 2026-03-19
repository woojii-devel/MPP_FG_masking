from huggingface_hub import login, upload_folder
from dotenv import load_dotenv
from pathlib import Path
import os
env_path = Path(".env")
load_dotenv(env_path)

hf_token = os.getenv("HF_TOKEN")
print(hf_token)