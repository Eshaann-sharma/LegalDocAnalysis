from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence


def classification_report_per_label(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    id2label: Dict[int, str],
) -> Dict[str, float]:
    """Compute per-label precision/recall/F1 and macro averages.

    Returned keys are flattened for Trainer logging, e.g.:
      - precision_termination
      - recall_termination
      - f1_termination
      - macro_f1
      - accuracy

    This avoids depending on sklearn.
    """
    # Confusion counts per label
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    n = 0

    for t, p in zip(y_true, y_pred):
        n += 1
        if t == p:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    metrics: Dict[str, float] = {}
    f1s: List[float] = []

    for lid, name in id2label.items():
        _tp = tp[lid]
        _fp = fp[lid]
        _fn = fn[lid]
        precision = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        recall = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        key = name.lower()
        metrics[f"precision_{key}"] = float(precision)
        metrics[f"recall_{key}"] = float(recall)
        metrics[f"f1_{key}"] = float(f1)
        f1s.append(float(f1))

    metrics["macro_f1"] = float(sum(f1s) / len(f1s)) if f1s else 0.0
    metrics["accuracy"] = float(correct / n) if n else 0.0
    return metrics
