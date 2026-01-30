from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForSequenceClassification = AutoTokenizer = None  # type: ignore


@dataclass(frozen=True)
class ClauseClassPrediction:
    label: str
    score: float


def _require_deps() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for clause classification inference.")
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise ModuleNotFoundError("transformers is required for clause classification inference.")


def _pick_device(explicit: Optional[str] = None):
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError("torch is required for clause classification inference.")
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _looks_like_local_path(model_name_or_path: str) -> bool:
    if model_name_or_path.startswith((".", "..", "artifacts/", "artifacts\\")):
        return True
    if "/" in model_name_or_path or "\\" in model_name_or_path:
        return True
    if len(model_name_or_path) >= 2 and model_name_or_path[1] == ":":
        return True
    return False


class ClauseClassifier:
    """Sequence classifier for already-segmented clause text.

    Expects a fine-tuned checkpoint directory created by
    `Backend.clause_detector.classifier.finetune`.
    """

    def __init__(self, model_name_or_path: str, *, device: Optional[str] = None, max_length: int = 256) -> None:
        _require_deps()

        import os

        if _looks_like_local_path(model_name_or_path) and not os.path.exists(model_name_or_path):
            raise FileNotFoundError(
                f"Clause classifier checkpoint not found at '{model_name_or_path}'. "
                "Train it first (Backend.clause_detector.classifier.finetune) or set CLAUSE_CLS_MODEL_PATH to a valid checkpoint directory."
            )

        self.device = _pick_device(device)
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # HuggingFace config holds id2label when trained via our pipeline.
        self.id2label = getattr(self.model.config, "id2label", None) or {}

    def predict(self, clause_text: str) -> ClauseClassPrediction:
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError("torch is required for clause classification inference.")
        if not clause_text or not clause_text.strip():
            return ClauseClassPrediction(label="OTHER", score=0.0)

        enc = self.tokenizer(
            clause_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)

        logits = out.logits  # [1, num_labels]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        score, idx = torch.max(probs, dim=-1)

        label = self.id2label.get(int(idx), str(int(idx)))
        return ClauseClassPrediction(label=str(label), score=float(score))


_DEFAULT_CLS: Optional[ClauseClassifier] = None


def classify_clause_text(
    clause_text: str,
    *,
    model_name_or_path: str,
) -> Dict[str, Any]:
    """Convenience functional API returning JSON-serializable output."""
    global _DEFAULT_CLS
    if _DEFAULT_CLS is None or _DEFAULT_CLS.model.name_or_path != model_name_or_path:
        _DEFAULT_CLS = ClauseClassifier(model_name_or_path=model_name_or_path)

    pred = _DEFAULT_CLS.predict(clause_text)
    return {"label": pred.label, "score": float(pred.score)}
