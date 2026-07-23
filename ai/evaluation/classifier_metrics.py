"""
ai/evaluation/classifier_metrics.py - Classification metric primitives.

VERSION: 1.0.0
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from numpy.typing import NDArray


EPSILON = 1e-15


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Return the fraction of exact label matches."""
    true, pred = _aligned_1d(y_true, y_pred)
    if len(true) == 0:
        return 0.0
    return float(np.mean(true == pred))


def precision(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "binary",
    positive_label: Any = 1,
) -> float:
    """Compute binary, macro, weighted, or micro averaged precision."""
    scores = _per_class_prf(y_true, y_pred, positive_label=positive_label)
    return _average_score(scores["precision"], scores["support"], scores, average, positive_label)


def recall(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "binary",
    positive_label: Any = 1,
) -> float:
    """Compute binary, macro, weighted, or micro averaged recall."""
    scores = _per_class_prf(y_true, y_pred, positive_label=positive_label)
    return _average_score(scores["recall"], scores["support"], scores, average, positive_label)


def f1(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    average: str = "binary",
    positive_label: Any = 1,
) -> float:
    """Compute binary, macro, weighted, or micro averaged F1."""
    scores = _per_class_prf(y_true, y_pred, positive_label=positive_label)
    return _average_score(scores["f1"], scores["support"], scores, average, positive_label)


def confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> NDArray[np.integer]:
    """Return a row=true, column=predicted confusion matrix."""
    true, pred = _aligned_1d(y_true, y_pred)
    class_labels = list(labels) if labels is not None else _unique_labels(true, pred)
    index = {label: idx for idx, label in enumerate(class_labels)}
    matrix = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for actual, estimated in zip(true, pred):
        if actual in index and estimated in index:
            matrix[index[actual], index[estimated]] += 1
    return matrix


def roc_auc(
    y_true: Sequence[Any],
    y_score: Sequence[float] | Sequence[Sequence[float]],
    average: str = "macro",
    labels: Sequence[Any] | None = None,
    positive_label: Any = 1,
) -> float:
    """Compute ROC AUC for binary scores or multiclass one-vs-rest scores."""
    true = _as_1d(y_true)
    score = np.asarray(y_score, dtype=float)
    if len(true) == 0:
        return 0.0
    if score.ndim == 1:
        return _binary_roc_auc(true == positive_label, score)

    class_labels = list(labels) if labels is not None else _infer_score_labels(true, score)
    aucs: List[float] = []
    supports: List[int] = []
    for idx, label in enumerate(class_labels[: score.shape[1]]):
        binary_true = true == label
        aucs.append(_binary_roc_auc(binary_true, score[:, idx]))
        supports.append(int(np.sum(binary_true)))
    return _average_values(np.asarray(aucs, dtype=float), np.asarray(supports, dtype=float), average)


def pr_auc(
    y_true: Sequence[Any],
    y_score: Sequence[float] | Sequence[Sequence[float]],
    average: str = "macro",
    labels: Sequence[Any] | None = None,
    positive_label: Any = 1,
) -> float:
    """Compute area under the precision-recall curve as average precision."""
    true = _as_1d(y_true)
    score = np.asarray(y_score, dtype=float)
    if len(true) == 0:
        return 0.0
    if score.ndim == 1:
        return _binary_average_precision(true == positive_label, score)

    class_labels = list(labels) if labels is not None else _infer_score_labels(true, score)
    aucs: List[float] = []
    supports: List[int] = []
    for idx, label in enumerate(class_labels[: score.shape[1]]):
        binary_true = true == label
        aucs.append(_binary_average_precision(binary_true, score[:, idx]))
        supports.append(int(np.sum(binary_true)))
    return _average_values(np.asarray(aucs, dtype=float), np.asarray(supports, dtype=float), average)


def log_loss(
    y_true: Sequence[Any],
    y_proba: Sequence[float] | Sequence[Sequence[float]],
    labels: Sequence[Any] | None = None,
    positive_label: Any = 1,
) -> float:
    """Compute clipped binary or multiclass negative log likelihood."""
    true = _as_1d(y_true)
    proba = np.clip(np.asarray(y_proba, dtype=float), EPSILON, 1.0 - EPSILON)
    if len(true) == 0:
        return 0.0

    if proba.ndim == 1:
        binary_true = (true == positive_label).astype(float)
        losses = -(binary_true * np.log(proba) + (1.0 - binary_true) * np.log(1.0 - proba))
        return float(np.mean(losses))

    class_labels = list(labels) if labels is not None else _infer_score_labels(true, proba)
    label_index = {label: idx for idx, label in enumerate(class_labels[: proba.shape[1]])}
    row_losses: List[float] = []
    for row, label in zip(proba, true):
        idx = label_index.get(label)
        if idx is not None:
            row_losses.append(float(-np.log(row[idx])))
    return float(np.mean(row_losses)) if row_losses else 0.0


def calibration_curve_data(
    y_true: Sequence[Any],
    y_proba: Sequence[float] | Sequence[Sequence[float]],
    n_bins: int = 10,
    positive_label: Any = 1,
) -> Dict[str, List[float] | List[int]]:
    """Return bin-level predicted probability, observed frequency, and counts."""
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    true = (_as_1d(y_true) == positive_label).astype(float)
    proba = _positive_probability(y_proba)
    if len(true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, n_bins - 1)
    prob_pred: List[float] = []
    prob_true: List[float] = []
    counts: List[int] = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        counts.append(int(np.sum(mask)))
        if np.any(mask):
            prob_pred.append(float(np.mean(proba[mask])))
            prob_true.append(float(np.mean(true[mask])))
        else:
            prob_pred.append(float((edges[bin_id] + edges[bin_id + 1]) / 2.0))
            prob_true.append(0.0)
    return {"prob_pred": prob_pred, "prob_true": prob_true, "counts": counts}


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Sequence[float] | Sequence[Sequence[float]] | None = None,
    labels: Sequence[Any] | None = None,
    average: str = "macro",
    positive_label: Any = 1,
) -> Dict[str, Any]:
    """Aggregate common classification metrics into a serializable dictionary."""
    true, pred = _aligned_1d(y_true, y_pred)
    unique = labels if labels is not None else _unique_labels(true, pred)
    binary = len(unique) <= 2
    prf_average = "binary" if binary else average
    metrics: Dict[str, Any] = {
        "accuracy": accuracy(true, pred),
        "precision": precision(true, pred, average=prf_average, positive_label=positive_label),
        "recall": recall(true, pred, average=prf_average, positive_label=positive_label),
        "f1": f1(true, pred, average=prf_average, positive_label=positive_label),
        "confusion_matrix": confusion_matrix(true, pred, labels=unique).tolist(),
        "labels": list(unique),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc(true, y_proba, average=average, labels=unique, positive_label=positive_label)
        metrics["pr_auc"] = pr_auc(true, y_proba, average=average, labels=unique, positive_label=positive_label)
        metrics["log_loss"] = log_loss(true, y_proba, labels=unique, positive_label=positive_label)
        if binary:
            metrics["calibration"] = calibration_curve_data(true, y_proba, positive_label=positive_label)
    return metrics


accuracy_score = accuracy
precision_score = precision
recall_score = recall
f1_score = f1
roc_auc_score = roc_auc
pr_auc_score = pr_auc
log_loss_score = log_loss


def _aligned_1d(y_true: Sequence[Any], y_pred: Sequence[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    return true, pred


def _as_1d(values: Sequence[Any]) -> NDArray[Any]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _unique_labels(*arrays: Iterable[Any]) -> List[Any]:
    labels: List[Any] = []
    for array in arrays:
        for value in np.asarray(list(array), dtype=object):
            label = value.item() if isinstance(value, np.generic) else value
            if label not in labels:
                labels.append(label)
    try:
        return sorted(labels)
    except TypeError:
        return labels


def _infer_score_labels(y_true: NDArray[Any], score: NDArray[np.floating]) -> List[Any]:
    labels = _unique_labels(y_true)
    if len(labels) == score.shape[1]:
        return labels
    return list(range(score.shape[1]))


def _per_class_prf(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    positive_label: Any,
) -> Dict[str, Dict[Any, float] | Dict[Any, int] | float]:
    true, pred = _aligned_1d(y_true, y_pred)
    labels = _unique_labels(true, pred)
    precisions: Dict[Any, float] = {}
    recalls: Dict[Any, float] = {}
    f1s: Dict[Any, float] = {}
    supports: Dict[Any, int] = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        true_pos = int(np.sum((true == label) & (pred == label)))
        false_pos = int(np.sum((true != label) & (pred == label)))
        false_neg = int(np.sum((true == label) & (pred != label)))
        support = int(np.sum(true == label))
        label_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
        label_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
        precisions[label] = float(label_precision)
        recalls[label] = float(label_recall)
        f1s[label] = (
            float(2.0 * label_precision * label_recall / (label_precision + label_recall))
            if (label_precision + label_recall)
            else 0.0
        )
        supports[label] = support
        total_tp += true_pos
        total_fp += false_pos
        total_fn += false_neg

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    if positive_label not in precisions and len(labels) == 2:
        positive_label = labels[-1]
    return {
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "support": supports,
        "positive_label": positive_label,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }


def _average_score(
    values: Dict[Any, float] | Any,
    support: Dict[Any, int] | Any,
    scores: Dict[str, Any],
    average: str,
    positive_label: Any,
) -> float:
    metric_name = next(
        name for name, candidate in scores.items() if candidate is values and name in {"precision", "recall", "f1"}
    )
    if average == "binary":
        active_label = scores.get("positive_label", positive_label)
        return float(values.get(active_label, 0.0))
    if average == "micro":
        return float(scores[f"micro_{metric_name}"])
    arr = np.asarray(list(values.values()), dtype=float)
    weights = np.asarray(list(support.values()), dtype=float)
    return _average_values(arr, weights, average)


def _average_values(values: NDArray[np.floating], weights: NDArray[np.floating], average: str) -> float:
    if len(values) == 0:
        return 0.0
    if average == "weighted":
        total = float(np.sum(weights))
        return float(np.average(values, weights=weights)) if total > 0 else 0.0
    if average in {"macro", "binary"}:
        return float(np.mean(values))
    raise ValueError(f"Unsupported average: {average}")


def _binary_roc_auc(y_true_binary: NDArray[np.bool_], score: NDArray[np.floating]) -> float:
    y = np.asarray(y_true_binary, dtype=bool)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = int(np.sum(y))
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return 0.0
    ranks = _average_ranks(s)
    pos_rank_sum = float(np.sum(ranks[y]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _average_ranks(values: NDArray[np.floating]) -> NDArray[np.floating]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _binary_average_precision(y_true_binary: NDArray[np.bool_], score: NDArray[np.floating]) -> float:
    y = np.asarray(y_true_binary, dtype=bool)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    positives = int(np.sum(y))
    if positives == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    cumulative_tp = np.cumsum(y_sorted)
    precision_at_k = cumulative_tp / (np.arange(len(y_sorted)) + 1.0)
    return float(np.sum(precision_at_k[y_sorted]) / positives)


def _positive_probability(y_proba: Sequence[float] | Sequence[Sequence[float]]) -> NDArray[np.floating]:
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        return np.clip(proba, 0.0, 1.0)
    if proba.shape[1] == 1:
        return np.clip(proba[:, 0], 0.0, 1.0)
    return np.clip(proba[:, -1], 0.0, 1.0)
