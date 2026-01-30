from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..inference.labels import ID2LABEL


def _labels_to_entities(label_ids: Sequence[int]) -> List[Tuple[str, int, int]]:
    """Convert a sequence of label ids into entity spans.

    Returns:
      List of (type, start_token_idx, end_token_idx_exclusive)

    Notes:
      - Ignores -100.
      - Robust to invalid BIO transitions.
    """
    entities: List[Tuple[str, int, int]] = []

    active_type: Optional[str] = None
    active_start: Optional[int] = None

    def flush(end_idx: int) -> None:
        nonlocal active_type, active_start
        if active_type is not None and active_start is not None:
            entities.append((active_type, active_start, end_idx))
        active_type = None
        active_start = None

    logical_idx = -1
    prev_was_valid = False

    for raw in label_ids:
        if raw == -100:
            continue
        logical_idx += 1

        label = ID2LABEL.get(int(raw), "O")
        if label == "O":
            flush(logical_idx)
            prev_was_valid = False
            continue

        prefix, ctype = label.split("-", 1)

        if prefix == "B":
            flush(logical_idx)
            active_type = ctype
            active_start = logical_idx
            prev_was_valid = True
        else:
            # I
            if active_type != ctype or active_start is None or not prev_was_valid:
                # invalid transition: start new
                flush(logical_idx)
                active_type = ctype
                active_start = logical_idx
            prev_was_valid = True

    flush(logical_idx + 1)
    return entities


def entity_level_prf(
    pred_label_ids: Sequence[Sequence[int]],
    gold_label_ids: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Compute micro-averaged entity-level Precision/Recall/F1.

    Entity span identity is (type, start, end) in token space.
    """
    tp = fp = fn = 0

    for pred, gold in zip(pred_label_ids, gold_label_ids):
        pred_ents = set(_labels_to_entities(pred))
        gold_ents = set(_labels_to_entities(gold))

        tp += len(pred_ents & gold_ents)
        fp += len(pred_ents - gold_ents)
        fn += len(gold_ents - pred_ents)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def per_type_f1(
    pred_label_ids: Sequence[Sequence[int]],
    gold_label_ids: Sequence[Sequence[int]],
) -> Dict[str, float]:
    """Per-type F1 at entity level (micro within type)."""
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred, gold in zip(pred_label_ids, gold_label_ids):
        pred_ents = _labels_to_entities(pred)
        gold_ents = _labels_to_entities(gold)

        pred_by_type = defaultdict(set)
        gold_by_type = defaultdict(set)

        for t, s, e in pred_ents:
            pred_by_type[t].add((s, e))
        for t, s, e in gold_ents:
            gold_by_type[t].add((s, e))

        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
        for t in all_types:
            p = pred_by_type[t]
            g = gold_by_type[t]
            stats[t]["tp"] += len(p & g)
            stats[t]["fp"] += len(p - g)
            stats[t]["fn"] += len(g - p)

    out: Dict[str, float] = {}
    for t, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[f"f1_{t.lower()}"] = float(f1)

    return out
