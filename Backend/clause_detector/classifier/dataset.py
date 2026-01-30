from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClauseExample:
    clause_id: str
    text: str
    label: str


def load_jsonl(path: str) -> List[ClauseExample]:
    """Load clause classification dataset from JSONL.

    Expected per-line schema (best effort):
      - clause_id: str (optional)
      - text: str (required)
      - label: str (required)

    Notes:
      - Skips invalid rows instead of failing hard.
    """
    out: List[ClauseExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = obj.get("text")
            label = obj.get("label")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(label, str) or not label.strip():
                continue

            clause_id = str(obj.get("clause_id", f"line-{i}"))
            out.append(ClauseExample(clause_id=clause_id, text=text, label=label.strip().upper()))

    return out


def build_label_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted({l.strip().upper() for l in labels if l and l.strip()})
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def tokenize_examples(
    *,
    tokenizer: Any,
    examples: Sequence[ClauseExample],
    label2id: Dict[str, int],
    max_length: int = 256,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for ex in examples:
        enc = tokenizer(
            ex.text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        features.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": int(label2id[ex.label]),
            }
        )
    return features
