# utils.py
import importlib
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    roc_auc_score,
)


# ============================================================
# Instantiation helpers
# ============================================================

def get_obj_from_str(string: str, reload: bool = False):
    """
    根据字符串 "module.submodule.ClassName" 返回对应的类/函数对象。
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Dict[str, Any]):
    """
    根据配置 dict 中的 "target" 和 "params" 实例化对象。
    config:
      target: "module.path.ClassName"
      params: {...}  # 作为关键字参数传入构造函数
    """
    if "target" not in config:
        if config in ("__is_first_stage__", "__is_unconditional__"):
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# ============================================================
# 多标签任务的评价指标
# ============================================================

def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    多标签任务常用评价指标：
      - subset accuracy（exact match）
      - hamming loss
      - micro/macro precision/recall/F1
      - per-class AUROC & macro-AUROC
      - per-class accuracy & macro class accuracy
      - top-1 / top-3 accuracy（是否命中至少一个真标签）
      - pred_subset_true_rate：所有预测疾病都是真疾病（没有 FP）
      - true_subset_pred_rate：所有真疾病都被预测到了（没有 FN）
      - avg_true_labels, avg_pred_labels
    """
    y_pred = (y_prob >= thresh).astype(int)
    N, C = y_true.shape

    # ------- subset accuracy（完全匹配） -------
    subset_acc = accuracy_score(y_true, y_pred)

    # ------- Hamming loss -------
    h_loss = hamming_loss(y_true, y_pred)

    # ------- micro/macro precision & recall & F1 -------
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ------- per-class AUROC & macro-AUROC -------
    aurocs: List[float] = []
    for k in range(C):
        try:
            auc_k = roc_auc_score(y_true[:, k], y_prob[:, k])
        except ValueError:
            # 该类可能全 0 或全 1
            auc_k = np.nan
        aurocs.append(float(auc_k))
    macro_auroc = float(np.nanmean(aurocs)) if len(aurocs) > 0 else float("nan")

    # ------- per-class accuracy -------
    per_class_accuracy: List[float] = []
    for k in range(C):
        correct = (y_true[:, k] == y_pred[:, k]).sum()
        per_class_accuracy.append(float(correct) / float(N) if N > 0 else float("nan"))
    macro_class_accuracy = float(np.nanmean(per_class_accuracy)) if len(per_class_accuracy) > 0 else float("nan")

    # ------- Top-k accuracy（命中至少一个真标签） -------
    # 只统计有至少一个正类的样本
    sorted_idx = np.argsort(y_prob, axis=1)  # 升序
    topk_results = {}

    for k in (1, 3):
        hits = 0
        count = 0
        for i in range(N):
            true_pos = np.flatnonzero(y_true[i])
            if true_pos.size == 0:
                continue  # 没有真标签的样本跳过
            count += 1
            topk = sorted_idx[i, -k:]  # 概率最高的 k 个
            if np.intersect1d(true_pos, topk).size > 0:
                hits += 1
        topk_acc = hits / count if count > 0 else float("nan")
        topk_results[f"top{k}_acc"] = float(topk_acc)

    # ------- 集合级别指标 -------
    # pred_subset_true_rate: 预测集合是真集合的子集（没有 FP）
    # true_subset_pred_rate: 真集合是预测集合的子集（没有 FN）
    pred_subset_true_flags: List[int] = []
    true_subset_pred_flags: List[int] = []

    true_label_counts: List[int] = []
    pred_label_counts: List[int] = []

    for i in range(N):
        true_set = set(np.flatnonzero(y_true[i]))
        pred_set = set(np.flatnonzero(y_pred[i]))

        true_label_counts.append(len(true_set))
        pred_label_counts.append(len(pred_set))

        # 预测集合 ⊆ 真集合（所有预测都是真疾病）
        pred_subset_true_flags.append(int(pred_set.issubset(true_set)))
        # 真集合 ⊆ 预测集合（所有真疾病都被预测到了）
        true_subset_pred_flags.append(int(true_set.issubset(pred_set)))

    pred_subset_true_rate = float(np.mean(pred_subset_true_flags)) if N > 0 else float("nan")
    true_subset_pred_rate = float(np.mean(true_subset_pred_flags)) if N > 0 else float("nan")

    avg_true_labels = float(np.mean(true_label_counts)) if N > 0 else 0.0
    avg_pred_labels = float(np.mean(pred_label_counts)) if N > 0 else 0.0

    metrics: Dict[str, Any] = {
        "subset_accuracy": float(subset_acc),
        "hamming_loss": float(h_loss),

        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),

        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),

        "macro_auroc": macro_auroc,
        "per_class_auroc": aurocs,

        "per_class_accuracy": per_class_accuracy,
        "macro_class_accuracy": macro_class_accuracy,

        "top1_acc": topk_results["top1_acc"],
        "top3_acc": topk_results["top3_acc"],

        "pred_subset_true_rate": pred_subset_true_rate,
        "true_subset_pred_rate": true_subset_pred_rate,

        "avg_true_labels": avg_true_labels,
        "avg_pred_labels": avg_pred_labels,
    }
    return metrics



# ============================================================
# 数据划分工具
# ============================================================

def split_samples(
    samples: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按比例 7:1:2 (或其他) 将 samples 划分为 train/val/test。
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    import random
    n = len(samples)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    return train_samples, val_samples, test_samples
