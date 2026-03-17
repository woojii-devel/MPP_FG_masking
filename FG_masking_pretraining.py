from pathlib import Path
import numpy as np
import pandas as pd
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from utils import get_label_columns
from datasets import load_from_disk
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from arguments import get_args,build_roberta_config,build_trainargs_kwargs,build_ft_trainargs_kwargs
from datetime import datetime
from rdkit import RDLogger
from FG_masking_collator_new import FG_masking_collator
from visualization import plot_loss_graph,plot_roc_curve
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Any
from scipy.special import expit
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.smiles.transforms import Augment
from Filtered_Dataset import ShardDatasetLoader
from FT_dataset_final import FT_Dataset
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    ITERSTRAT_AVAILABLE = True
except ImportError:
    ITERSTRAT_AVAILABLE = False
from padding_collator import PaddingCollator


class RobertaForMultiLabelMasked(RobertaForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,   # 기본 loss 계산 안 쓰기
            **kwargs
        )

        logits = outputs.logits

        if labels is not None:
            labels = labels.to(logits.device)

            # -1은 결측치 → mask
            loss_mask = (labels > -0.5).float()

            loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')

            target = torch.where(labels > -0.5, labels, torch.zeros_like(labels))
            loss = loss_fct(logits, target)

            masked_loss = loss * loss_mask
            final_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)

            return SequenceClassifierOutput(
                loss=final_loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class RoBERTa_FG:
    
    
    def __init__(self,args):

        self.time = datetime.now().strftime("%H%M")
        self.date = datetime.now().strftime("%Y%m%d")
        self.args = args
        set_seed(self.args.seed)
        self.tokenizer = SMILESTokenizer(vocab_file=self.args.vocab_path,
                                canonical=False,
                                sanitize=True)
        
        self.config = build_roberta_config(self.args,self.tokenizer)
        self.pt_config = build_trainargs_kwargs(self.args)
        # print("vocab_size:", self.config.vocab_size)
        ##############################
        

        
        
    def pretraining(self):
        
        pt_base_dir = self.args.out_dir+f'{self.args.data_type}_results/{self.date}/{self.args.model_name}_{self.args.pt_masking_method}'
        pt_model_dir = Path(pt_base_dir+f'_{self.time}')
        pt_model_dir.mkdir(parents=True, exist_ok=True)
        
        loader = ShardDatasetLoader(
            repo_id = self.args.repo_id,
            train_ratio=0.98,
            val_ratio=0.01,
            seed=42,
            max_len=512,
            min_len=6,
        )

        train_ds, val_ds, test_ds = loader.dataset()
        pt_args = TrainingArguments(
            output_dir=str(pt_model_dir),
            ddp_find_unused_parameters=False,
            fp16 = True,  
            max_grad_norm = 1.0,
            **self.pt_config
        )
        pt_args.remove_unused_columns = False
        collator = FG_masking_collator(tokenizer= self.tokenizer,
                                       mlm_probability= self.args.mlm_probability,
                                       mode = self.args.pt_masking_method)
        
        
        model = RobertaForMaskedLM(config=self.config)

        trainer = Trainer(
        model = model,
        args= pt_args,
        data_collator = collator,
        train_dataset= train_ds,
        eval_dataset= val_ds,
        # processing_class=  self.tokenizer,
        )

        
# =============================================debugging=====================================================

        # dl = trainer.get_train_dataloader()
        # t0 = time.time()
        # batch = next(iter(dl))
        # print("first batch seconds:", time.time() - t0)
        # print("trainer collate_fn:", dl.collate_fn)
        # print("is my collator?", dl.collate_fn == collator)
        # print("type:", type(dl.collate_fn))

        # batch = next(iter(torch.utils.data.DataLoader(val_ds, batch_size=5, collate_fn=collator)))
        # print(batch['input_ids'])


        # model = model.cuda()

        # for step, batch in enumerate(dl):
        #     print("step:", step)

        #     input_ids = batch["input_ids"]
        #     attention_mask = batch["attention_mask"]

        #     # 범위 검사
        #     if input_ids.min() < 0 or input_ids.max() >= model.config.vocab_size:
        #         print("🚨 input_ids 범위 문제")
        #         print("min/max:", input_ids.min().item(), input_ids.max().item())
        #         break

        #     if set(torch.unique(attention_mask).tolist()) - {0,1}:
        #         print("🚨 attention_mask 값 문제")
        #         print(torch.unique(attention_mask))
        #         break
        #     print("seq_len:", batch["input_ids"].shape[1])
            # print("max_position_embeddings:", model.config.max_position_embeddings)
            # valid = batch["attention_mask"].bool()
            # masked = batch["input_ids"].eq(575) & valid

            # mask_ratio_per_sample = masked.sum(dim=1).float() / valid.sum(dim=1).float()
            # print("mask_ratio batch mean/min/max:",
            #     mask_ratio_per_sample.mean().item(),
            #     mask_ratio_per_sample.min().item(),
            #     mask_ratio_per_sample.max().item())
            # print("step:", step, "seq_len:", batch["input_ids"].shape[1])
            # # GPU forward
            # for k, v in batch.items():
            #     if torch.is_tensor(v):
            #         batch[k] = v.cuda()

            # try:
            #     out = model(**batch)
            # except Exception as e:
            #     print("🔥 여기서 터짐 at step", step)
            #     raise e
#=======================================================================================================
        print("torch.cuda.is_available() =", torch.cuda.is_available())

        device = trainer.args.device
        print("Trainer device =", device)

        if not torch.cuda.is_available() or device.type != "cuda":
            raise RuntimeError(
                f"GPU(CUDA) not in use. torch.cuda.is_available()={torch.cuda.is_available()}, "
                f"trainer.args.device={device}. "
                "Fix: install CUDA-enabled torch / set device to cuda / run on a GPU machine."
            )

        print("Using GPU:", torch.cuda.get_device_name(device.index if device.index is not None else 0))
        
        try:
            trainer.train()

        except Exception as e:
            print("\n===== TRAINING CRASHED =====")
            print(e)

            print("\n===== LAST BATCH DEBUG =====")
            print(collator.last_debug)

            raise
        plot_loss_graph(trainer,
                             title = 'pretrained',
                             args= self.args,
                             save_path = str(pt_model_dir) + f'/pt_loss_FG_masking')
        print('PreTraining done / Test start')
        test_metrics = trainer.evaluate(test_ds) 
        print(test_metrics)
        print('Test done')
        
        return trainer
     
     
    
    def finetuning(self,pretrained_ckpt_path,task, train_ds, val_ds, test_ds):
        print(f'{task} task finetuning start')

        RDLogger.DisableLog('rdApp.*')
        
        exp_dir = Path(self.args.out_dir)

        ft_model_dir = exp_dir / "models" / task / f"seed_{self.args.seed}"
        ft_model_dir.mkdir(parents=True, exist_ok=True)
        
        is_multiLabel = task in ['sider','muv','tox21']
        is_regression = task in ['esol','freesolv', 'lipo']
        
        if is_multiLabel:
            if not ITERSTRAT_AVAILABLE:
                raise ImportError("iterstrat not installed but multilabel split requested")
        
        
        ft_tr_config = build_ft_trainargs_kwargs(self.args, is_regression)

        
        if task in ['hiv', 'muv']:
            ft_tr_config['logging_steps']  = 100
            ft_tr_config['save_steps'] = ft_tr_config['eval_steps'] = 50
            ft_tr_config['num_train_epochs'] = 50
        elif task in ['bbbp','bace','sider','esol']:
            ft_tr_config['logging_steps']  = 20
            ft_tr_config['save_steps'] = ft_tr_config['eval_steps'] = 20
            ft_tr_config['num_train_epochs'] = 50
        else:
            ft_tr_config['logging_steps']  = 20
            ft_tr_config['save_steps'] = ft_tr_config['eval_steps'] = 20
            ft_tr_config['num_train_epochs'] = 50

        # data_root = Path(finetuning_dataset_path) / f'{task}_dataset'
        
        # ds = FT_Dataset(data_path=data_root,task= task, max_len=512, min_len = 6, seed=self.args.seed)
        # train_ds, val_ds, test_ds = ds.split_dataset(method='random')

        label_cols = get_label_columns(task)
        
        if is_multiLabel:
            num_labels = len(label_cols)
        elif is_regression:
            num_labels = 1
        else:
            num_labels = 2

        print(f'len of tr_ds:{len(train_ds)} | len of val_ds:{len(val_ds)} | len of te_ds:{len(test_ds)}')
        print(f'num of {task} labels: {num_labels}')

        if is_multiLabel:
            config = RobertaConfig.from_pretrained(pretrained_ckpt_path,subfolder=self.args.subfolder, num_labels=num_labels,
                                                   hidden_dropout_prob=0.1,
                                                    attention_probs_dropout_prob=0.1,
                                                    classifier_dropout=0.1,)
            config.problem_type = 'multi_label_classification'
            finetuning_model = RobertaForMultiLabelMasked(config)
            finetuning_model.roberta.load_state_dict(
                RobertaForMaskedLM.from_pretrained(
                    pretrained_ckpt_path,
                    subfolder=self.args.subfolder
                ).roberta.state_dict()
            )
        elif is_regression:
            config = RobertaConfig.from_pretrained(pretrained_ckpt_path, subfolder=self.args.subfolder,num_labels=num_labels,
                                                   hidden_dropout_prob=0.1,
                                                    attention_probs_dropout_prob=0.1,
                                                    classifier_dropout=0.1,)
            config.problem_type = "regression"
            finetuning_model = RobertaForSequenceClassification(config)
            finetuning_model.roberta.load_state_dict(
                RobertaForMaskedLM.from_pretrained(
                    pretrained_ckpt_path,
                    subfolder=self.args.subfolder
                ).roberta.state_dict()
            )
        else:
            config = RobertaConfig.from_pretrained(pretrained_ckpt_path,subfolder=self.args.subfolder, num_labels=num_labels,
                                                   hidden_dropout_prob=0.1,
                                                    attention_probs_dropout_prob=0.1,
                                                    classifier_dropout=0.1,)
            config.problem_type = 'single_label_classification'
            finetuning_model = RobertaForSequenceClassification(config)
            finetuning_model.roberta.load_state_dict(
                RobertaForMaskedLM.from_pretrained(
                    pretrained_ckpt_path,
                    subfolder=self.args.subfolder
                ).roberta.state_dict()
            )

        compute_metrics = self.make_compute_metrics(is_multiLabel=is_multiLabel,is_regression = is_regression)

        collator = PaddingCollator()

        fine_tuning_args = TrainingArguments(
            output_dir=str(ft_model_dir),
            ddp_find_unused_parameters=False,
            fp16=True,
            max_grad_norm = 1.0,
            **ft_tr_config
        )


        ft_trainer = Trainer(
            model=finetuning_model,
            args=fine_tuning_args,
            data_collator=collator,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        ft_trainer.train()

        test_metrics = ft_trainer.evaluate(test_ds)
        if ft_trainer.is_world_process_zero():
            ft_trainer.save_metrics("test", test_metrics)
        
        y_pred, y_true = self.predict_outputs_and_labels(
            ft_trainer, test_ds, is_multiLabel=is_multiLabel, is_regression=is_regression
        )
        if not is_regression:
            out = self.compute_auc_and_roc(y_pred = y_pred, y_true = y_true, is_multiLabel = is_multiLabel)
            auc_value = out['auc']
            if ft_trainer.is_world_process_zero():
                self.write_result(
                    task,
                    value=auc_value,
                    is_regression=is_regression
                )
            
                plot_roc_curve(out = out,
                            is_multiLabel=is_multiLabel,
                            save_path = str(ft_model_dir / f"roc_curve_{task}"),
                            title = f'{task} roc curve  | auc : {auc_value:.4f}')

            
        
        if is_regression:
            reg = self.compute_regression_metrics(y_pred, y_true)
            print("TEST regression metrics:", reg)
            if ft_trainer.is_world_process_zero():
                ft_trainer.save_metrics("test_regression", reg)
            rmse_value = reg['rmse']
            if ft_trainer.is_world_process_zero():
                self.write_result(
                    task,
                    value=rmse_value,
                    is_regression=is_regression
                )
            
            
        if ft_trainer.is_world_process_zero():    
             plot_loss_graph(ft_trainer, title=f'{task}_finetuning', args=self.args,
                            save_path=str(ft_model_dir / f"ft_loss_{task}"), task='finetuning')

    
    
    
    
            
    def predict_outputs_and_labels(self,trainer, dataset,is_multiLabel,is_regression):
        pred = trainer.predict(dataset)
        logits = np.asarray(pred.predictions)
        y_true = np.asarray(pred.label_ids)
        if is_multiLabel: 
            # multi-label: logits (N, L) -> probs (N, L)
            y_pred = expit(logits).astype(np.float32)
            # y_true shape (N, L), 값은 0/1/-1(결측)일 수 있음
            y_true = y_true.astype(np.float32)
        elif is_regression:
            # logits: (N,1) or (N,)
            y_pred = logits.reshape(-1).astype(np.float32)
            y_true = y_true.reshape(-1).astype(np.float32)

        else:   # binary
            y_true = y_true.reshape(-1).astype(int)
            if logits.ndim == 2 and logits.shape[1] == 2:
                x = logits - logits.max(axis=1, keepdims=True)
                exps = np.exp(x)
                y_pred = exps[:, 1] / exps.sum(axis=1)
            else:
                y_pred = expit(logits.reshape(-1))

        return y_pred, y_true
        
    # def evaluate_test_binary(self,trainer, test_ds, out_dir, prefix="test"):
    #     prob1, y = self.predict_probs_and_labels_binary(trainer, test_ds)
    #     auc = float(roc_auc_score(y, prob1))

    #     metrics = {f"{prefix}_auc": auc}
    #     print(metrics)

    #     # HF 방식 저장
    #     trainer.save_metrics(prefix, metrics)

    #     # ROC 저장
    #     plot_roc_curve_binary(
    #         prob1, y,
    #         save_path=str(Path(out_dir) / f"{prefix}_roc.png"),
    #         title=f"{prefix} ROC (AUC={auc:.4f})"
    #     )
    #     return metrics

    def compute_regression_metrics(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred = np.asarray(y_pred).reshape(-1).astype(np.float32)
        y_true = np.asarray(y_true).reshape(-1).astype(np.float32)

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return {"mse": mse, "rmse": rmse, "mae": mae}


        
    def compute_auc_and_roc(self,y_pred,y_true,is_multiLabel):

        out = {"auc": None, "roc": None}
        
        if is_multiLabel:

            aucs = []
            roc_per_label = []
            L = y_true.shape[1]

            for j in range(L):
                yj = y_true[:, j]
                pj = y_pred[:, j]

                # handle missing labels: -1 or nan
                mask = np.isfinite(yj) & (yj != -1)
                if mask.sum() < 2:
                    aucs.append(np.nan)
                    roc_per_label.append(None)
                    continue

                yj = yj[mask].astype(int)
                pj = pj[mask]

                # some labels may be all 0 or all 1 -> roc_auc_score fails
                if len(np.unique(yj)) < 2:
                    aucs.append(np.nan)
                    roc_per_label.append(None)
                    continue

                aucs.append(roc_auc_score(yj, pj))
                roc_per_label.append(roc_curve(yj, pj))

            macro_auc = np.nanmean(aucs) if np.any(~np.isnan(aucs)) else np.nan
            out.update({"auc": float(macro_auc), "auc_per_label": aucs, "roc_per_label": roc_per_label})
            return out

        else:
            auc = roc_auc_score( y_true= y_true,y_score= y_pred)
            fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_pred)
            out.update({"auc": float(auc), "roc": (fpr, tpr, thr), "prob1": y_pred, "y": y_true})
            return out
            

    
        
    def make_compute_metrics(self,is_multiLabel: bool,is_regression: bool) -> callable:

        def compute_metrics(eval_pred) -> Dict[str, Any]:
            logits, labels = eval_pred
            labels = np.asarray(labels)
            
            if is_multiLabel:
                y_pred = expit(logits)
                y_true = labels.astype(np.float32)
                out = self.compute_auc_and_roc(y_pred=y_pred,y_true=y_true,is_multiLabel=is_multiLabel)
                return {'auc': float(out['auc'])}
            elif is_regression:
                y_pred = np.asarray(logits).reshape(-1).astype(np.float32)
                y_true = labels.reshape(-1).astype(np.float32)

                mse = float(np.mean((y_true - y_pred) ** 2))
                rmse = float(np.sqrt(mse))
                
                return {"mse": mse, "rmse": rmse}

            else:
                if logits.ndim == 2 and logits.shape[1] == 2:
                    x = logits - logits.max(axis=1, keepdims=True)
                    exps = np.exp(x)
                    y_pred = exps[:, 1] / exps.sum(axis=1)
                else:
                    y_pred = expit(logits.reshape(-1))

                y_true = labels.reshape(-1).astype(np.int64)
                auc = roc_auc_score(y_true=y_true, y_score=y_pred)
                return {"auc": float(auc)}

        return compute_metrics
    



    def write_result(self, task: str, value, is_regression) -> str:

        if self.args.result_file is not None:
            auc_file = Path(self.args.result_file)
        else:
            base_path = Path(self.args.out_dir)
            base_path.mkdir(parents=True, exist_ok=True)
            auc_file = base_path / f"cv_result_{task}.txt"

        if is_regression:
            line = f"seed {self.args.seed} rmse: {value:.6f}\n"
        else:
            line = f"seed {self.args.seed} auc: {value:.6f}\n"

        with open(auc_file, "a", encoding="utf-8") as f:
            f.write(line)

        return str(auc_file.resolve())
    


