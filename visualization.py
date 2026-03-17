import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve,auc
import numpy as np
def plot_loss_graph(trainer, title, args, save_path=None,  smooth_window=0,task = 'pretrain'):
    hist = trainer.state.log_history

    # train loss (logging)
    train_steps = [h["step"] for h in hist if "loss" in h and "eval_loss" not in h]
    train_loss  = [h["loss"] for h in hist if "loss" in h and "eval_loss" not in h]

    # val loss (evaluation)
    val_steps = [h["step"] for h in hist if "eval_loss" in h]
    val_loss  = [h["eval_loss"] for h in hist if "eval_loss" in h]

    # optional smoothing (moving average)
    def smooth(y, w):
        if w is None or w <= 1 or len(y) < w:
            return y
        out = []
        for i in range(len(y)):
            s = max(0, i - w + 1)
            out.append(sum(y[s:i+1]) / (i - s + 1))
        return out

    if smooth_window and smooth_window > 1:
        train_loss_plot = smooth(train_loss, smooth_window)
        val_loss_plot   = smooth(val_loss, smooth_window)
    else:
        train_loss_plot, val_loss_plot = train_loss, val_loss

    plt.figure()
    if train_steps:
        plt.plot(train_steps, train_loss_plot, label="train_loss")
    if val_steps:
        plt.plot(val_steps, val_loss_plot, label="val_loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    epoch = trainer.args.num_train_epochs
    hd = args.hidden_size
    hl = args.num_hidden_layers
    ah = args.num_attention_heads
    if task == 'pretrain':
        masking_method = args.pt_masking_method
    else: 
        masking_method = args.ft_masking_method
    model_name = args.model_name
    plt.title(
    f"{title} / {model_name} / {masking_method}\n"
    f"hidden_size:{hd} / #_hiddenlayers:{hl} / #_attentionHeads:{ah} / {epoch} epoch"
    )

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.splitext(save_path)[1]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path += f"{timestamp}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")


    

def plot_roc_curve_binary(prob1, y_true, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, prob1)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        base_path, extension = os.path.splitext(save_path)
        if not extension:
            extension = ".png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"{base_path}_{timestamp}{extension}"

        plt.savefig(final_save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved at: {final_save_path}")
    plt.close()

def plot_train_val_roc_binary(train_prob1, train_y, val_prob1, val_y, save_path, title):
    fpr_t, tpr_t, _ = roc_curve(train_y, train_prob1)
    fpr_v, tpr_v, _ = roc_curve(val_y, val_prob1)

    plt.figure()
    plt.plot(fpr_t, tpr_t, label="train")
    plt.plot(fpr_v, tpr_v, label="val")
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    if save_path:

        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)


        base_path, extension = os.path.splitext(save_path)
        
 
        if not extension:
            extension = ".png"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"{base_path}_{timestamp}{extension}"

        plt.savefig(final_save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved at: {final_save_path}")
    plt.close()

def plot_train_val_test_roc_binary(train_prob1, train_y, val_prob1, val_y, test_prob1, test_y, save_path, title):
    fpr_t, tpr_t, _ = roc_curve(train_y, train_prob1)
    fpr_v, tpr_v, _ = roc_curve(val_y, val_prob1)
    fpr_s, tpr_s, _ = roc_curve(test_y, test_prob1)

    plt.figure()
    plt.plot(fpr_t, tpr_t, label="train")
    plt.plot(fpr_v, tpr_v, label="val")
    plt.plot(fpr_s, tpr_s, label="test")
    plt.plot([0, 1], [0, 1],linestyle='--',color = 'gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:

        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)


        base_path, extension = os.path.splitext(save_path)
        
  
        if not extension:
            extension = ".png"


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"{base_path}_{timestamp}{extension}"

        plt.savefig(final_save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved at: {final_save_path}") 
    plt.close()
    
    
def plot_train_val_test_roc_multilabel(
    train_probs, train_y,
    val_probs, val_y,
    test_probs, test_y,
    title, save_path,
    n_points=100,
    show_used_labels=True,
):
    plt.figure(figsize=(9, 7))

    def get_macro_roc(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_true.ndim != 2 or y_prob.ndim != 2:
            raise ValueError("y_true and y_prob must be 2D: (n_samples, n_labels)")
        if y_true.shape != y_prob.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")

        mean_fpr = np.linspace(0.0, 1.0, n_points)
        tprs = []
        used = 0

        for i in range(y_true.shape[1]):
            mask = (y_true[:, i] != -1)
            yi = y_true[mask, i]
            pi = y_prob[mask, i]

            if yi.size < 2 or np.unique(yi).size < 2:
                continue

            fpr, tpr, _ = roc_curve(yi, pi)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            used += 1

        if used == 0:
            return None

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        macro_auc = auc(mean_fpr, mean_tpr)

        return {"fpr": mean_fpr, "tpr": mean_tpr, "auc": macro_auc, "used_labels": used}

    def safe_auc_str(d):
        """d: None or dict with key 'auc'"""
        if d is None:
            return "NA"
        a = d.get("auc", None)
        return "NA" if a is None else f"{float(a):.4f}"

    tr = get_macro_roc(train_y, train_probs)
    va = get_macro_roc(val_y, val_probs)
    te = get_macro_roc(test_y, test_probs)

    tr_auc = safe_auc_str(tr)
    va_auc = safe_auc_str(va)
    te_auc = safe_auc_str(te)

    if tr is not None:
        lab = f"Train (AUC={tr_auc})"
        if show_used_labels:
            lab += f", labels={tr['used_labels']}"
        plt.plot(tr["fpr"], tr["tpr"], label=lab)

    if va is not None:
        lab = f"Val (AUC={va_auc})"
        if show_used_labels:
            lab += f", labels={va['used_labels']}"
        plt.plot(va["fpr"], va["tpr"], label=lab)

    if te is not None:
        lab = f"Test (AUC={te_auc})"
        if show_used_labels:
            lab += f", labels={te['used_labels']}"
        plt.plot(te["fpr"], te["tpr"], label=lab)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    full_title = f"{title} (train_auc={tr_auc}, val_auc={va_auc}, test_auc={te_auc})"
    plt.title(full_title)

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:

        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)


        base_path, extension = os.path.splitext(save_path)
        
        if not extension:
            extension = ".png"

   

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_save_path = f"{base_path}_{timestamp}{extension}"

        plt.savefig(final_save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved at: {final_save_path}") 
    plt.close()


def plot_roc_curve(out, is_multiLabel: bool, save_path: str, title: str):
    """
    compute_auc_and_roc()의 반환값에서 나온 roc를 받아 ROC curve를 그리고 저장한다.

    Args:
        roc:
          - binary: (fpr, tpr, thr)
          - multilabel: roc_per_label = [ (fpr,tpr,thr) or None, ... ]
        is_multiLabel: 멀티라벨 여부
        save_path: 저장 경로(확장자 없으면 .png)
        title: 그래프 제목
    """
    plt.figure()

    if not is_multiLabel:

        fpr, tpr, _ = out['roc']
        plt.plot(fpr, tpr)
    else:

        roc_per_label = out['roc_per_label']
        any_plotted = False

        for j, r in enumerate(roc_per_label):
            if r is None:
                continue
            fpr, tpr, _ = r
            plt.plot(fpr, tpr, label=f"label_{j}")
            any_plotted = True

        if any_plotted:
            plt.legend(loc="lower right", fontsize=8)


    plt.plot([0, 1], [0, 1])

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        base_path, extension = os.path.splitext(save_path)
        if not extension:
            extension = ".png"
        timestamp = datetime.now().strftime("%H%M")
        final_save_path = f"{base_path}_{timestamp}{extension}"

        plt.savefig(final_save_path, dpi=200, bbox_inches="tight")
        print(f"Graph saved at: {final_save_path}")

    plt.close()
