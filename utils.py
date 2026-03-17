
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