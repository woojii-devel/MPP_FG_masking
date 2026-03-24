import argparse
from transformers import RobertaConfig
# model config
MODEL_PRESETS = {
    # model_name: config overrides
    "proto-type": dict(
        hidden_size="__FROM_ARGS__",
        num_hidden_layers="__FROM_ARGS__",
        num_attention_heads="__FROM_ARGS__",
        intermediate_size="__FROM_ARGS__",
    ),
    "SELFormer": dict(
        num_hidden_layers=12,
        num_attention_heads=4,

    ),
    "SELFormer-Lite": dict(
        num_hidden_layers=8,
        num_attention_heads=12,
    ),
    "MLM-FG": dict(
        num_hidden_layers=12,
        num_attention_heads=12,
    ),
}

def build_roberta_config(args, tokenizer) -> RobertaConfig:
    if args.model_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model_name={args.model_name}. choices={list(MODEL_PRESETS)}")

    preset = dict(MODEL_PRESETS[args.model_name])  

    # proto-type은 args에서 가져오도록 치환
    for k, v in list(preset.items()):
        if v == "__FROM_ARGS__":
            preset[k] = getattr(args, k)

    cfg = RobertaConfig(
        vocab_size=tokenizer.number_of_tokens,
        pad_token_id= tokenizer.padding_index,
        bos_token_id=tokenizer.start_index,
        eos_token_id=tokenizer.stop_index,
        type_vocab_size=1,
        max_position_embeddings=514,
        **preset,
    )
    return cfg


def build_trainargs_kwargs(args) -> dict:


    return dict(
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="steps",
        lr_scheduler_type=args.lr_scheduler_type,
        report_to="none",
        save_total_limit=2,

        load_best_model_at_end=True,
        dataloader_num_workers=args.num_workers,
        per_device_train_batch_size=args.train_bs,
        # gradient_accumulation_steps = args.grad_accum,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        # save_steps=args.save_steps,
        eval_strategy="epoch",
        # eval_steps=args.eval_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only= False,
    )

    





def build_ft_trainargs_kwargs(args,is_regression):
    common = dict(
        warmup_ratio=0.06,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=args.num_workers,
        data_seed = args.seed,
        seed = args.seed,

    
    )
    if is_regression:
        return dict(
            **common,
            per_device_train_batch_size=args.ft_train_bs,
            per_device_eval_batch_size=args.ft_eval_bs,
            learning_rate=args.ft_lr,
            num_train_epochs=args.ft_epochs,
            logging_steps=args.ft_logging_steps,
            save_strategy="steps",
            save_steps=args.ft_save_steps,
            eval_strategy="steps",
            eval_steps=args.ft_eval_steps,
            metric_for_best_model="rmse",
            greater_is_better=False,
        )
    else:
        return dict(
            **common,
            per_device_train_batch_size=args.ft_train_bs,
            per_device_eval_batch_size=args.ft_eval_bs,
            learning_rate=args.ft_lr,
            num_train_epochs=args.ft_epochs,
            logging_steps=args.ft_logging_steps,
            save_strategy="steps",
            save_steps=args.ft_save_steps,
            eval_strategy="steps",
            eval_steps=args.ft_eval_steps,
            metric_for_best_model="auc",
            greater_is_better=True,
        )







def get_args():

    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--seeds", nargs='+',type=int, default=[1,2,3,4,5])
    parser.add_argument("--seed",type=int,default=42)
    
    #pretraining arguments
    parser.add_argument("--vocab_path", type=str, default= 'smiles_tokenizer_pytoda/vocab.json')
    parser.add_argument("--repo_id", type=str, default= 'Zaeus/MPP_pt_data')
    parser.add_argument("--out_dir", type=str, default=f"", help="모델 저장 폴더")
    parser.add_argument("--task", nargs='+', default=['clintox','bbbp','bace','tox21','sider','esol','lipo','freesolv','muv','hiv'])
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--data_type", type=str,default='smiles')
    parser.add_argument("--model_name", type=str, default='proto-type')
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_fg_ratio", type=float, default=0.10)
    parser.add_argument("--train_bs", type=int, default=512)
    parser.add_argument("--eval_bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lr_scheduler_type",type=str,default="cosine",help="Learning rate scheduler type")
    parser.add_argument("--epochs", type=float, default=50)
    
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)


    #parameter of model
    parser.add_argument("--pt_masking_method", type=str, default='hybrid_masking')
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--NUM_OF_SAMPLES", type=int)
    #fine-tuning arguments
    parser.add_argument("--ft_model_path", type=str, default= 'Zaeus/MPP')
    parser.add_argument("--subfolder", type=str, default= "checkpoint-430515")
    parser.add_argument("--ft_data_path", type=str, default= 'FT_Data')
    parser.add_argument("--ft_masking_method", type=str, default='finetuning')
    parser.add_argument("--ft_train_bs", type=int, default=64)
    parser.add_argument("--ft_eval_bs", type=int, default=64)
    parser.add_argument("--ft_lr", type=float, default=2e-5)
    parser.add_argument("--ft_epochs", type=float, default=10)
    
    parser.add_argument("--ft_logging_steps", type=int, default=100)
    parser.add_argument("--ft_save_steps", type=int, default=100)
    parser.add_argument("--ft_eval_steps", type=int, default=100)
    
    parser.add_argument("--ft_num_workers", type=int, default=4)

    parser.add_argument(
    "--result_file",
    type=str,
    default=None,
    help="Cross-validation result file path"
)

    args = parser.parse_args()
    
    
    return args