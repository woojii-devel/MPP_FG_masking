import os, sys, re, json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit import RDConfig
from pytoda.smiles.smiles_language import SMILESTokenizer
from datasets import Dataset
import pandas as pd

# -------------------------
# Globals for workers
# -------------------------
tokenizer = None

def init_worker(vocab_path, tokenizer_kwargs):
    """프로세스가 생성될 때 딱 한 번 실행되어 토크나이저를 로드합니다."""
    global tokenizer
    tokenizer = SMILESTokenizer(vocab_file=vocab_path, **tokenizer_kwargs)


def get_label_columns(task):
    if task == 'esol':
        label_cols = ["measured log solubility in mols per litre"]
    elif task == 'freesolv':
        label_cols = ["expt"]
    elif task == 'lipo':
        label_cols = ["exp"]
    elif task == "bbbp":
        label_cols = ["p_np"]
    elif task == "bace":
        label_cols = ["Class"]
    elif task == "hiv":
        label_cols = ["HIV_active"]
    elif task == 'clintox':
        label_cols = ["CT_TOX"]   # 필요하면 ["FDA_APPROVED", "CT_TOX"] 로 변경
    elif task == "tox21":
        label_cols = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
    elif task == "muv":
        label_cols = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689",
            "MUV-692", "MUV-712", "MUV-713", "MUV-733", "MUV-737", "MUV-810",
            "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]
    elif task == "sider":
        label_cols = [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions', 'Endocrine disorders',
            'Surgical and medical procedures', 'Vascular disorders',
            'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
            'Renal and urinary disorders', 'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders', 'Nervous system disorders',
            'Injury, poisoning and procedural complications'
        ]
    else:
        raise ValueError(f"Unknown task: {task}")
    return label_cols

# -------------------------
# Core Logic
# -------------------------
def build_record(args):
    """
    SMILES -> input_ids (with <cls>, <eos>) 변환
    """
    global tokenizer
    smiles, labels = args
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    # 모델 입력의 일관성을 위해 Canonical SMILES 사용
    canon_smiles = Chem.MolToSmiles(mol, canonical=True)

    try:
        # 1. SMILES -> 정수 인덱스 리스트 변환
        token_ids = tokenizer.smiles_to_token_indexes(canon_smiles)
        
        # numpy나 tensor 형태일 경우 리스트로 변환
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        else:
            token_ids = list(token_ids)
            
        # 2. 앞뒤에 Special Tokens 부착 (<cls> + ids + <eos>)
        # pytoda: start_index(<cls>), stop_index(<eos>)
        full_input_ids = [tokenizer.start_index] + token_ids + [tokenizer.stop_index]
        
        # 3. 데이터 반환 (SMILES 텍스트는 제외)
        return {
            "smiles":canon_smiles,
            "input_ids": full_input_ids,
            "labels": labels  # 예: [0, 1, 0, ...] 또는 [regression_val]
        }
    except Exception:
        return None

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # 설정 (본인의 환경에 맞게 수정)
    TASK = ['clintox','bbbp','bace','tox21','sider','muv','hiv','esol','lipo','freesolv']
    for task in TASK:
        if task in ['tox21','sider','muv','clintox']:
            CSV_PATH = f'../Data/{task}.csv.gz'
            print(os.path.abspath(CSV_PATH), os.path.exists(CSV_PATH))
        else:
            CSV_PATH = f'../Data/{task}.csv'
        OUTPUT_PATH = f"FT_Data/{task}_dataset"
        VOCAB_PATH = "smiles_tokenizer_pytoda/vocab.json"

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        RDLogger.DisableLog("rdApp.*")

        # 1. CSV 데이터 로드 및 라벨 추출
        df = pd.read_csv(CSV_PATH)
        # 앞에서 정의한 get_label_columns 함수를 사용한다고 가정
        label_cols = get_label_columns(task) 
        smi_col = 'smiles'
        
        if task in ['tox21','sider','muv','clintox']:
            df[label_cols] = df[label_cols].fillna(-1) 
        
        # (SMILES, Labels_list) 형태로 묶기
        data_to_process = list(zip(df[smi_col], df[label_cols].values.tolist()))
        print(f"Processing {len(data_to_process)} samples for {task}...")

        # 2. 병렬 토크나이징 진행
        n_proc = max(1, cpu_count() - 1)
        all_results = []

        with Pool(processes=n_proc, 
                initializer=init_worker, 
                initargs=(VOCAB_PATH, {'canonical': True, 'sanitize': True})) as pool:
            
            it = pool.imap_unordered(build_record, data_to_process, chunksize=200)

            for rec in tqdm(it, total=len(data_to_process), desc="Tokenizing"):
                if rec is not None:
                    all_results.append(rec)

        # 3. 단일 Arrow 데이터셋으로 저장
        if all_results:
            final_ds = Dataset.from_list(all_results)
            final_ds.save_to_disk(OUTPUT_PATH)
            print(f"✅ Saved to {OUTPUT_PATH}")
            print(f"📊 Sample input_ids: {all_results[0]['input_ids'][:10]}...")
        else:
            print("❌ Failed to process any data.")