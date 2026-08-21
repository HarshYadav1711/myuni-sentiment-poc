"""Classification and continuous evaluation metrics for the MyUni sentiment POC."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

SENTIMENT_LABELS: tuple[str, ...] = ("negative", "neutral", "positive")


def _as_list(values: Iterable[str]) -> list[str]:
    return [str(v) for v in values]


def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> dict[str, dict[str, int]]:
    """Return nested dict matrix[true][pred] = count."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for truth, pred in zip(_as_list(y_true), _as_list(y_pred), strict=True):
        if truth not in matrix or pred not in matrix[truth]:
            # Keep unknown labels visible rather than silently dropping.
            matrix.setdefault(truth, {p: 0 for p in labels})
            for row in matrix.values():
                row.setdefault(pred, 0)
            matrix[truth][pred] = matrix[truth].get(pred, 0) + 1
        else:
            matrix[truth][pred] += 1
    return matrix


def _per_class_counts(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label: str,
) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for truth, pred in zip(_as_list(y_true), _as_list(y_pred), strict=True):
        if pred == label and truth == label:
            tp += 1
        elif pred == label and truth != label:
            fp += 1
        elif pred != label and truth == label:
            fn += 1
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> dict:
    """Accuracy, per-class P/R/F1, macro/weighted F1, and confusion matrix."""
    truths = _as_list(y_true)
    preds = _as_list(y_pred)
    if len(truths) != len(preds):
        raise ValueError("y_true and y_pred must have the same length")
    if not truths:
        raise ValueError("empty predictions")

    correct = sum(1 for t, p in zip(truths, preds, strict=True) if t == p)
    accuracy = correct / len(truths)

    per_class: dict[str, dict[str, float]] = {}
    support = Counter(truths)
    macro_f1_acc = 0.0
    weighted_f1_acc = 0.0

    for label in labels:
        tp, fp, fn = _per_class_counts(truths, preds, label)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support.get(label, 0)),
        }
        macro_f1_acc += f1
        weighted_f1_acc += f1 * support.get(label, 0)

    n_labels = len(labels)
    macro_f1 = macro_f1_acc / n_labels if n_labels else 0.0
    weighted_f1 = weighted_f1_acc / len(truths)

    # Macro precision/recall averaged similarly.
    macro_p = sum(per_class[l]["precision"] for l in labels) / n_labels
    macro_r = sum(per_class[l]["recall"] for l in labels) / n_labels

    return {
        "n_examples": len(truths),
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(truths, preds, labels=labels),
        "labels": list(labels),
    }


def continuous_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict:
    """MAE and Pearson correlation for continuous sentiment scores."""
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("continuous y_true and y_pred must match in shape")
    if truth.size == 0:
        raise ValueError("empty continuous predictions")

    mae = float(np.mean(np.abs(truth - pred)))
    if truth.size < 2 or float(np.std(truth)) == 0.0 or float(np.std(pred)) == 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(truth, pred)[0, 1])

    return {
        "n_examples": int(truth.size),
        "mae": mae,
        "pearson_correlation": corr,
        "true_mean": float(np.mean(truth)),
        "pred_mean": float(np.mean(pred)),
    }
