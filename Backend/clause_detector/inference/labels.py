from __future__ import annotations

# Keep label set small and stable; it will be used for model config and decoding.
CLAUSE_TYPES = [
    "TERMINATION",
    "DEPOSIT",
    "RENT",
    "USAGE",
    "SUBLETTING",
    "MAINTENANCE",
    "CONFIDENTIALITY",
]

# BIO label list in a deterministic order.
# Index is the class id used by the token-classification head.
LABELS = ["O"] + [f"B-{t}" for t in CLAUSE_TYPES] + [f"I-{t}" for t in CLAUSE_TYPES]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
