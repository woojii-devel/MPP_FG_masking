
from huggingface_hub import login, upload_folder
import os

# 직접 토큰을 적지 말고 환경 변수에서 가져오게 만듭니다.
token = os.getenv("HF_TOKEN") 
login(token=token)

# Push your model files
upload_folder(folder_path="/home/user/MPP/Vast_ai_folder/FT_Data", 
              repo_id="Zaeus/MPP_ft_dataset", 
              repo_type="dataset")
