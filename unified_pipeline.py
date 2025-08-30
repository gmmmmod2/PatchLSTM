"""
Usage example:
python unified_pipeline.py \
  --models patchlstm \
  --data ./Data/beijing_geolife_incident_strict.pkl \
  --batch 32 --max_len 120 --max_seq 16 \
  --epochs 100 --patience 8 \
  --loss focal \
  --outdir ./checkpoints --results_csv ./batch_results.csv
"""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List
import contextlib
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== local utils (replace interns' paths) ====
from utils.DataLoader import create_dataloaders
from utils.utils import _safe_torch_load, _normalize_state_dict_keys

# ======= Models registry (adapt to your project) =======
# You can expand these constructors as needed.
from modules.PatchLSTM import PatchLSTM
from modules import (
    LSTMClassifier, TCNClassifier, TransformerAD, GRUTransformer, PatchTSTClassifier, GDN,
    CutAddPasteModel, PeFAD, AnomalyTransformer, NCAD
)

MODEL_REGISTRY = {
    "patchlstm": lambda vocab_size: PatchLSTM(vocab_size=vocab_size, patch_sizes=[10, 20, 30], is_double=False,
                                              use_neighbors=False),
    "lstm": lambda vocab_size: LSTMClassifier(),
    "tcn": lambda vocab_size: TCNClassifier(),
    "transformer": lambda vocab_size: TransformerAD(),
    "grutrans": lambda vocab_size: GRUTransformer(),
    "patchtst": lambda vocab_size: PatchTSTClassifier(),
    "gdn": lambda vocab_size: GDN(),
    "cutadd": lambda vocab_size: CutAddPasteModel(),
    "pefad": lambda vocab_size: PeFAD(),
    "ncad": lambda vocab_size: NCAD(),
    "anomaly_tf": lambda vocab_size: AnomalyTransformer(),
}

def build_model(name: str, vocab_size: int) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[key](vocab_size)

# ========================= Losses & Metrics ========================= #
class FocalLoss(nn.Module):
    """Binary focal loss over 2-class logits.
    logits: (B,2), targets: (B,) in {0,1}
    """
    def __init__(self, alpha: float = 0.6, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = ensure_two_class_logits(logits)
        prob = torch.softmax(logits, dim=1)  # (B,2)
        pt = prob[torch.arange(prob.size(0), device=prob.device), targets]
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
            torch.tensor(1.0 - self.alpha, device=logits.device, dtype=logits.dtype),
        )
        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * torch.log(pt.clamp_min(1e-9))
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# Fixed (non-CLI) hyperparams for losses
CE_LR = 1e-4
FOCAL_LR = 1e-4
FOCAL_GAMMA = 2.0  # not exposed via CLI

@torch.no_grad()
def _class_prior_from_train(loader: DataLoader) -> float:
    n1 = 0
    n = 0
    for b in loader:
        y = b['labels'].view(-1).long()
        n1 += int((y == 1).sum().item())
        n += int(y.numel())
    return (n1 / max(n, 1)) if n > 0 else 0.5

# robust metrics (AUC/PRC safe when single class)
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score, precision_score,
    average_precision_score
)

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    auc = None
    auprc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
        try:
            auprc = average_precision_score(y_true, y_prob)
        except Exception:
            auprc = None
    return {"ACC": acc, "Recall": recall, "F1": f1, "Precision": prec, "AUC": auc, "AUPRC": auprc}

@torch.no_grad()
def precision_at_k(y_true, y_prob, k=5):
    if y_prob is None:
        return None
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if len(y_true) == 0:
        return None
    k = min(k, len(y_true))
    idx = np.argpartition(-y_prob, kth=k-1)[:k]
    return float(y_true[idx].sum()) / float(k)

# ========================= Utilities ========================= #

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_best(model, out_dir, tag="best", extra_cfg=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    ckpt = out_dir / f"{tag}.pth"
    torch.save(state, ckpt.as_posix())
    if extra_cfg is not None:
        import json
        with open(out_dir / f"{tag}.config.json", "w", encoding="utf-8") as f:
            json.dump(extra_cfg, f, indent=2)
    return ckpt


def load_state_strict(model: nn.Module, weights_path: Path) -> Tuple[list, list]:
    state_dict = _safe_torch_load(weights_path.as_posix(), map_location='cpu')
    state_dict = _normalize_state_dict_keys(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Explicitly report; if many layers missing, it's likely the root cause of all-0 predictions
    if missing or unexpected:
        print("⚠️ State dict mismatch:")
        if missing:
            print("  Missing keys:", missing[:10], ("..." if len(missing)>10 else ""))
        if unexpected:
            print("  Unexpected keys:", unexpected[:10], ("..." if len(unexpected)>10 else ""))
    return missing, unexpected


def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- shape guard ---- #

def ensure_two_class_logits(logits: torch.Tensor) -> torch.Tensor:
    """Accepts (B,2) or (B,1)/(B,) and converts to (B,2) for safety."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.size(-1) == 2:
        return logits
    if logits.size(-1) == 1:
        x = logits
        return torch.cat([-x, x], dim=-1)
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

# ========================= Train/Test one model ========================= #

def train_one_model(
    model_name: str,
    device: torch.device,
    train_loader,
    val_loader,
    vocab_size: int,
    out_root: Path,
    epochs: int,
    patience: int,
    loss_type: str,
) -> Tuple[nn.Module, Path, Dict[str, float]]:
    model = build_model(model_name, vocab_size)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # loss selection (hyperparams fixed inside)
    if loss_type == 'focal':
        prior = _class_prior_from_train(train_loader)
        alpha = max(0.05, min(0.95, 1.0 - prior))  # favor minority; not exposed via CLI
        criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
        lr = FOCAL_LR
        loss_cfg = {"type":"focal","alpha":alpha,"gamma":FOCAL_GAMMA,"lr":lr}
    elif loss_type == 'ce':
        # compute class weights from train split only
        prior = _class_prior_from_train(train_loader)
        w0 = 1.0
        w1 = 0.0 if prior<=0 else max(1.0, (1.0-prior)/max(prior,1e-9))
        class_weights = torch.tensor([w0, w1], dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        lr = CE_LR
        loss_cfg = {"type":"ce","weights":[float(w0), float(w1)],"lr":lr}
    else:
        raise ValueError("loss_type must be 'ce' or 'focal'")

    incidentCriterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_metric = -math.inf
    best_dir = out_root / model_name
    bad_epochs = 0

    for epoch in range(1, epochs+1):
        # ---- Train ----
        model.train(); running = 0.0
        bar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{epochs}", unit="batch")
        for i, batch in enumerate(bar):
            labels = batch['labels'].long()
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast('cuda') if torch.cuda.is_available() else contextlib.nullcontext()
            with amp_ctx:
                outputs = model(batch)
                logits = ensure_two_class_logits(outputs["logits"])  # guard
                if loss_type == 'ce':
                    cls_loss = criterion(logits, labels)
                else:  # focal expects long labels too
                    cls_loss = criterion(logits, labels)
                inc_pred = outputs.get("incident_pred_center")
                loss = cls_loss + (0.5 * incidentCriterion(inc_pred, batch['incident']) if inc_pred is not None else 0.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running += float(loss.item())
            bar.set_postfix({"TrainLoss": f"{running / (i+1):.4f}"})

        # ---- Validate ----
        model.eval(); val_loss=0.0
        all_y, all_pred, all_prob = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['labels'].long()
                outputs = model(batch)
                logits = ensure_two_class_logits(outputs['logits'])
                if loss_type == 'ce':
                    cls_loss = criterion(logits, labels)
                else:
                    cls_loss = criterion(logits, labels)
                inc_pred = outputs.get('incident_pred_center')
                loss = cls_loss + (0.5 * incidentCriterion(inc_pred, batch['incident']) if inc_pred is not None else 0.0)
                val_loss += float(loss.item())

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_y.extend(labels.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())
                all_prob.extend(probs[:,1].cpu().numpy().tolist())

        metrics = compute_metrics(np.asarray(all_y), np.asarray(all_pred), np.asarray(all_prob))
        print(f"[{model_name}] Epoch {epoch}: TrainLoss={running/len(train_loader):.4f} | "
              f"ValLoss={val_loss/len(val_loader):.4f} | F1={metrics['F1']:.4f} | "
              f"Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} | "
              f"AUC={metrics['AUC'] if metrics['AUC'] is not None else float('nan'):.4f} | "
              f"AUPRC={metrics['AUPRC'] if metrics['AUPRC'] is not None else float('nan'):.4f}")

        # Early stopping on AUPRC (more stable for imbalanced); fallback to F1 if AUPRC is None
        val_score = metrics['AUPRC'] if metrics['AUPRC'] is not None else metrics['F1']
        if val_score > best_val_metric:
            best_val_metric = val_score
            bad_epochs = 0
            cfg = {
                "model": model_name,
                "vocab_size": int(vocab_size),
                "loss": loss_cfg,
            }
            ckpt = _save_best(model, best_dir, tag="best", extra_cfg=cfg)
            # ---- Post-save consistency check: reload and re-eval on val to catch broken checkpoints ----
            _m = build_model(model_name, vocab_size).to(device)
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                _m = nn.DataParallel(_m)
            missing, unexpected = load_state_strict(_m, ckpt)
            _m.eval(); _allp=[]; _ally=[]
            with torch.no_grad():
                for b in val_loader:
                    y = b['labels'].long()
                    lg = ensure_two_class_logits(_m(b)['logits'])
                    pr = torch.softmax(lg, dim=-1)[:,1]
                    _allp += pr.cpu().tolist(); _ally += y.cpu().tolist()
            from sklearn.metrics import average_precision_score as _aps
            _aupr = None
            try:
                _aupr = _aps(_ally, _allp)
            except Exception:
                pass
            print(f"✅ [{model_name}] New best saved: {ckpt} (score={best_val_metric:.4f}, reload_val_AUPRC={_aupr})")
            if _aupr is None or (metrics['AUPRC'] is not None and _aupr < 0.9*metrics['AUPRC']):
                print("⚠️ Reload AUPRC deviates a lot from in-memory eval. This often explains 'test all-zero'. Check state dict mismatches above.")
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print(f"[{model_name}] Early stopping (patience {patience})")
                break

    return model, best_dir / "best.pth", {"best_val": best_val_metric}


@torch.no_grad()
def test_one_model(
    model_name: str,
    device: torch.device,
    test_loader,
    vocab_size: int,
    weights_path: Path,
    results_csv: Path | None,
    data_args: Dict,
):
    model = build_model(model_name, vocab_size)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    missing, unexpected = load_state_strict(model, weights_path)
    model.eval()

    all_y, all_pred, all_prob = [], [], []
    for batch in test_loader:
        outputs = model(batch)
        logits = ensure_two_class_logits(outputs['logits'])
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_y.extend(batch['labels'].long().cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs[:,1].cpu().numpy().tolist())

    y_true = np.asarray(all_y, dtype=int)
    y_pred = np.asarray(all_pred, dtype=int)
    y_prob = np.asarray(all_prob, dtype=float)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Prediction stats to catch 'all-zero' collapse
    pos_rate = float((y_pred==1).mean()) if len(y_pred)>0 else float('nan')
    print(f"\n📊 Test Set Metrics | Pred_Pos_Rate={pos_rate:.4f}")
    for k in ["F1","Precision","Recall","AUC","AUPRC","ACC"]:
        v = metrics.get(k, None)
        if v is not None:
            print(f"{k}: {v:.4f}")
    print(f"Weights: {weights_path}")

    if results_csv is not None:
        header = [
            "timestamp","model","F1","Precision","Recall","AUC","AUPRC","ACC",
            "weights_path","threshold","data","batch","split","max_len","max_seq","pred_pos_rate"
        ]
        row = [
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            *(f"{metrics.get(k, float('nan')):.6f}" if metrics.get(k, None) is not None else "" for k in ["F1","Precision","Recall","AUC","AUPRC","ACC"]),
            weights_path.as_posix(),
            "argmax",
            data_args["data"],
            data_args["batch"],
            data_args["split_str"],
            data_args["max_len"],
            data_args["max_seq"],
            f"{pos_rate:.6f}",
        ]
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        need_header = (not results_csv.exists()) or results_csv.stat().st_size == 0
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if need_header:
                w.writerow(header)
            w.writerow(row)
        print(f"Appended to CSV: {results_csv}")

# ========================= Runner ========================= #

def run_all_models(
    models: List[str],
    data: str,
    batch: int,
    max_len: int,
    max_seq: int,
    epochs: int,
    patience: int,
    split_tuple: Tuple[float, float, float],
    outdir: str,
    results_csv: str,
    seed: int,
    loss_type: str,
):
    set_deterministic(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ONE split reused across models (prevents data leakage between runs)
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(
        data_path=data,
        batch_size=batch,
        split=split_tuple,
        max_len=max_len,
        max_seq=max_seq,
        device=device,
        seed=seed,
        label_as_float=False,  # we train with 2-class logits
    )
    print("Data loaders created once and reused across models.")

    # quick label ratio report
    import numpy as _np
    from collections import Counter as _C
    def _ratio(loader):
        _ys=[]
        for _b in loader:
            _ys += _b['labels'].long().cpu().tolist()
        _c=_C(_ys); _n=sum(_c.values()); return {k: f"{v}/{_n}={v/_n:.3f}" for k,v in _c.items()}
    print("Label ratios -> train:", _ratio(train_loader), "val:", _ratio(val_loader), "test:", _ratio(test_loader))

    out_root = ensure_dir(outdir)
    results_csv = Path(results_csv)

    data_args = dict(
        data=data,
        batch=batch,
        split_str=",".join(str(x) for x in split_tuple),
        max_len=max_len,
        max_seq=max_seq,
    )

    for name in models:
        print(f"\n=== [{name}] Train + Validate ===")
        _, best_path, _ = train_one_model(
            model_name=name,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=vocab_size,
            out_root=out_root,
            epochs=epochs,
            patience=patience,
            loss_type=loss_type,
        )

        print(f"\n=== [{name}] Test ===")
        test_one_model(
            model_name=name,
            device=device,
            test_loader=test_loader,
            vocab_size=vocab_size,
            weights_path=best_path,
            results_csv=results_csv,
            data_args=data_args,
        )

# ========================= CLI ========================= #

def main():
    ap = argparse.ArgumentParser(description="Unified training/testing (shared splits, argmax, CE/Focal)")
    ap.add_argument("--models", type=str, required=True, help="Comma-separated model names or 'ALL'")
    ap.add_argument("--data", type=str, default="./Data/trajectory_with_incident.pkl")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--max_seq", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="./checkpoints")
    ap.add_argument("--results_csv", type=str, default="./batch_results.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", type=str, choices=["ce","focal"], default="focal")
    args = ap.parse_args()

    split_tuple = (0.6, 0.2, 0.2) if args.split is None else tuple(float(x.strip()) for x in args.split.split(","))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) == 1 and models[0].upper() == "ALL":
        models = list(MODEL_REGISTRY.keys())
        print("Using ALL models:", models)

    run_all_models(
        models=models,
        data=args.data,
        batch=args.batch,
        max_len=args.max_len,
        max_seq=args.max_seq,
        epochs=args.epochs,
        patience=args.patience,
        split_tuple=split_tuple,
        outdir=args.outdir,
        results_csv=args.results_csv,
        seed=args.seed,
        loss_type=args.loss,
    )

if __name__ == "__main__":
    main()
