
# model.py
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision import models
from sklearn.metrics import roc_auc_score


def build_backbone(backbone_name: str, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    if backbone_name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    if backbone_name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    if backbone_name in ("inception_v3", "inception"):
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.inception_v3(weights=weights, aux_logits=False)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def _one_vs_rest_sens_spec(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # per-class sensitivity (TPR) and specificity (TNR)
    num_classes = cm.shape[0]
    total = cm.sum()
    sens = np.full((num_classes,), np.nan, dtype=np.float64)
    spec = np.full((num_classes,), np.nan, dtype=np.float64)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens[i] = _safe_div(tp, tp + fn)
        spec[i] = _safe_div(tn, tn + fp)
    return sens, spec


def _per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> np.ndarray:
    # y_prob: (N, C)
    aurocs = np.full((num_classes,), np.nan, dtype=np.float64)
    for i in range(num_classes):
        y_bin = (y_true == i).astype(np.int64)
        # roc_auc_score requires both classes present
        if y_bin.min() == y_bin.max():
            continue
        try:
            aurocs[i] = roc_auc_score(y_bin, y_prob[:, i])
        except Exception:
            aurocs[i] = np.nan
    return aurocs


def _macro_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


class MultiClassLitModel(pl.LightningModule):
    """
    Multi-class (single-label) frame-level classifier.
    - CrossEntropyLoss
    - Metrics (val/test):
        overall_acc
        per-class auroc + macro_auroc
        per-class sensitivity/specificity (one-vs-rest)
    - Save predictions:
        * per-sample CSV + JSON (paths, true, pred, prob vector)
        * metrics summary JSON
    """

    def __init__(
        self,
        num_labels: int,
        lr: float = 1e-4,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        label_list: Optional[List[str]] = None,
        class_weights: Optional[List[List[Any]]] = None,  # e.g. [["normal", 3.0], ...]
        save_predictions: bool = False,
        predictions_subdir: str = "predictions",
    ):
        super().__init__()
        if isinstance(lr, str):
            lr = float(lr)

        self.save_hyperparameters(ignore=["label_list"])

        self.num_labels = int(num_labels)
        self.lr = lr
        self.label_list = label_list or [f"class_{i}" for i in range(self.num_labels)]

        self.model = build_backbone(backbone_name, pretrained=pretrained, num_classes=self.num_labels)

        # class weights (optional): name-based -> tensor[C]
        weight = None
        if class_weights is not None:
            name2idx = {n: i for i, n in enumerate(self.label_list)}
            w = np.ones((self.num_labels,), dtype=np.float32)
            for name, val in class_weights:
                if name in name2idx:
                    w[name2idx[name]] = float(val)
            weight = torch.tensor(w, dtype=torch.float32)

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.save_predictions = bool(save_predictions)
        self.predictions_subdir = str(predictions_subdir)

        self._val_cache = []
        self._test_cache = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Return per-sample probabilities + ground truth for external saving."""
        x, y, paths = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        return {
            "paths": list(paths),
            "y_true": y.detach().cpu(),
            "probs": probs.detach().cpu(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # ---------------- train ----------------
    def training_step(self, batch, batch_idx: int):
        x, y, _paths = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ---------------- val/test shared ----------------
    def _shared_step(self, batch, stage: str):
        x, y, paths = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        cache_item = {
            "y": y.detach().cpu(),
            "pred": pred.detach().cpu(),
            "prob": prob.detach().cpu(),
            "paths": list(paths),
        }
        if stage == "val":
            self._val_cache.append(cache_item)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        else:
            self._test_cache.append(cache_item)
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "test")

    def _get_out_dir(self) -> str:
        # Prefer logger directory when available
        if getattr(self, "logger", None) is not None and getattr(self.logger, "log_dir", None):
            root = self.logger.log_dir
        else:
            root = self.trainer.default_root_dir if self.trainer is not None else os.getcwd()
        out_dir = os.path.join(root, self.predictions_subdir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _compute_and_log_metrics(self, stage: str, cache: List[Dict[str, Any]]):
        y = torch.cat([c["y"] for c in cache], dim=0).numpy().astype(np.int64)
        pred = torch.cat([c["pred"] for c in cache], dim=0).numpy().astype(np.int64)
        prob = torch.cat([c["prob"] for c in cache], dim=0).numpy().astype(np.float64)

        cm = _confusion_matrix(y, pred, self.num_labels)
        overall_acc = float((pred == y).mean()) if len(y) > 0 else float("nan")

        per_class_auroc = _per_class_auroc(y, prob, self.num_labels)
        macro_auroc = _macro_mean(per_class_auroc)

        sens, spec = _one_vs_rest_sens_spec(cm)

        # log overall
        self.log(f"{stage}/overall_acc", overall_acc, prog_bar=True)
        self.log(f"{stage}/macro_auroc", macro_auroc, prog_bar=False)

        # log per-class (as scalars)
        for i, name in enumerate(self.label_list):
            self.log(f"{stage}/auroc_{name}", float(per_class_auroc[i]) if not np.isnan(per_class_auroc[i]) else float("nan"))
            self.log(f"{stage}/sensitivity_{name}", float(sens[i]) if not np.isnan(sens[i]) else float("nan"))
            self.log(f"{stage}/specificity_{name}", float(spec[i]) if not np.isnan(spec[i]) else float("nan"))

        # print readable summary
        print(f"\n[{stage.upper()}] overall_acc={overall_acc:.4f} macro_auroc={macro_auroc:.4f}")
        print(f"[{stage.upper()}] confusion_matrix (rows=true, cols=pred):\n{cm}")
        for i, name in enumerate(self.label_list):
            print(f"[{stage.upper()}] {name:>20s} | AUROC={per_class_auroc[i]:.4f}  Sens={sens[i]:.4f}  Spec={spec[i]:.4f}")

        # save files
        if self.save_predictions:
            out_dir = self._get_out_dir()
            epoch = int(self.current_epoch) if stage == "val" else None
            tag = f"{stage}_epoch{epoch:03d}" if epoch is not None else f"{stage}"
            csv_path = os.path.join(out_dir, f"{tag}_predictions.csv")
            json_path = os.path.join(out_dir, f"{tag}_predictions.json")
            metrics_path = os.path.join(out_dir, f"{tag}_metrics.json")

            # per-sample rows
            rows = []
            for c in cache:
                y_b = c["y"].numpy().astype(np.int64)
                p_b = c["pred"].numpy().astype(np.int64)
                pr_b = c["prob"].numpy().astype(np.float64)
                paths = c["paths"]
                for yy, pp, pr, path in zip(y_b, p_b, pr_b, paths):
                    rows.append({
                        "path": path,
                        "y_true": int(yy),
                        "y_true_name": self.label_list[int(yy)] if 0 <= int(yy) < self.num_labels else str(int(yy)),
                        "y_pred": int(pp),
                        "y_pred_name": self.label_list[int(pp)] if 0 <= int(pp) < self.num_labels else str(int(pp)),
                        **{f"prob_{self.label_list[i]}": float(pr[i]) for i in range(self.num_labels)},
                    })

            # write CSV
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path","y_true","y_pred"])
                writer.writeheader()
                if rows:
                    writer.writerows(rows)

            # write JSON (list)
            with open(json_path, "w") as f:
                json.dump(rows, f, indent=2)

            # metrics summary
            metrics = {
                "stage": stage,
                "overall_acc": overall_acc,
                "macro_auroc": macro_auroc,
                "label_list": self.label_list,
                "per_class_auroc": {self.label_list[i]: (None if np.isnan(per_class_auroc[i]) else float(per_class_auroc[i])) for i in range(self.num_labels)},
                "per_class_sensitivity": {self.label_list[i]: (None if np.isnan(sens[i]) else float(sens[i])) for i in range(self.num_labels)},
                "per_class_specificity": {self.label_list[i]: (None if np.isnan(spec[i]) else float(spec[i])) for i in range(self.num_labels)},
                "confusion_matrix": cm.tolist(),
                "num_samples": int(len(y)),
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"[{stage.upper()}] Saved predictions to:\n  {csv_path}\n  {json_path}")
            print(f"[{stage.upper()}] Saved metrics to:\n  {metrics_path}")

    def on_validation_epoch_end(self):
        if len(self._val_cache) == 0:
            return
        cache = self._val_cache
        self._val_cache = []
        self._compute_and_log_metrics("val", cache)

    def on_test_epoch_end(self):
        if len(self._test_cache) == 0:
            return
        cache = self._test_cache
        self._test_cache = []
        self._compute_and_log_metrics("test", cache)
