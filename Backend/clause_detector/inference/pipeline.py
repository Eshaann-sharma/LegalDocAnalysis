from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
except ModuleNotFoundError:  # pragma: no cover
    AutoConfig = AutoTokenizer = AutoModelForTokenClassification = None  # type: ignore

from .labels import ID2LABEL, LABEL2ID, LABELS


@dataclass(frozen=True)
class DetectedClause:
    clause_type: str
    start_char: int
    end_char: int
    text: str
    score: float


def _pick_device(explicit_device: Optional[str] = None):
    if torch is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for clause detection inference. Install it (e.g. pip install torch) "
            "or use an environment that already includes PyTorch."
        )

    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_special_token(offset: Tuple[int, int]) -> bool:
    # HuggingFace uses (0,0) offsets for special tokens when offsets are returned.
    return offset[0] == 0 and offset[1] == 0


def _decode_spans(
    *,
    text: str,
    offsets: List[List[Tuple[int, int]]],
    pred_label_ids: List[List[int]],
    pred_scores: List[List[float]],
) -> List[DetectedClause]:
    """Convert token-level BIO predictions into clause spans in character offsets.

    Notes:
      - This assumes BIO tags.
      - Overlapping chunks may produce duplicate spans; caller should merge.
    """

    clauses: List[DetectedClause] = []

    for chunk_offsets, chunk_labels, chunk_scores in zip(offsets, pred_label_ids, pred_scores):
        active_type: Optional[str] = None
        active_start: Optional[int] = None
        active_end: Optional[int] = None
        active_scores: List[float] = []

        def flush_active() -> None:
            nonlocal active_type, active_start, active_end, active_scores
            if active_type is None or active_start is None or active_end is None:
                return
            span_text = text[active_start:active_end]
            score = float(sum(active_scores) / max(1, len(active_scores)))
            clauses.append(
                DetectedClause(
                    clause_type=active_type,
                    start_char=active_start,
                    end_char=active_end,
                    text=span_text,
                    score=score,
                )
            )
            active_type = None
            active_start = None
            active_end = None
            active_scores = []

        for (start, end), label_id, score in zip(chunk_offsets, chunk_labels, chunk_scores):
            if _is_special_token((start, end)):
                continue
            if end <= start:
                continue

            label = ID2LABEL[int(label_id)]

            if label == "O":
                flush_active()
                continue

            prefix, ctype = label.split("-", 1)

            if prefix == "B":
                flush_active()
                active_type = ctype
                active_start = start
                active_end = end
                active_scores = [float(score)]
                continue

            # prefix == "I"
            if active_type != ctype or active_start is None:
                # Invalid transition: start a new span.
                flush_active()
                active_type = ctype
                active_start = start
                active_end = end
                active_scores = [float(score)]
            else:
                active_end = end
                active_scores.append(float(score))

        flush_active()

    return clauses


def _merge_overlapping_spans(spans: Iterable[DetectedClause]) -> List[DetectedClause]:
    """Merge overlapping/adjacent spans of the same type.

    This is primarily to de-duplicate spans emitted from overlapping token windows.
    """
    spans_sorted = sorted(spans, key=lambda s: (s.clause_type, s.start_char, s.end_char))
    merged: List[DetectedClause] = []

    for s in spans_sorted:
        if not merged:
            merged.append(s)
            continue

        last = merged[-1]
        if s.clause_type != last.clause_type:
            merged.append(s)
            continue

        # Overlap or adjacency
        if s.start_char <= last.end_char:
            new_start = min(last.start_char, s.start_char)
            new_end = max(last.end_char, s.end_char)
            # Combine scores conservatively (weighted by span length).
            last_len = max(1, last.end_char - last.start_char)
            s_len = max(1, s.end_char - s.start_char)
            new_score = (last.score * last_len + s.score * s_len) / (last_len + s_len)
            merged[-1] = DetectedClause(
                clause_type=last.clause_type,
                start_char=new_start,
                end_char=new_end,
                text=last.text if (new_start == last.start_char and new_end == last.end_char) else "",
                score=float(new_score),
            )
        else:
            merged.append(s)

    # Restore text for merged spans (we dropped it above when spans changed)
    out: List[DetectedClause] = []
    for s in merged:
        if s.text:
            out.append(s)
        else:
            out.append(
                DetectedClause(
                    clause_type=s.clause_type,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text="",  # filled by caller in ClauseDetector.predict
                    score=s.score,
                )
            )
    return out


def _looks_like_local_path(model_name_or_path: str) -> bool:
    # Heuristic: treat common local paths as local even if they don't exist.
    if model_name_or_path.startswith((".", "..", "artifacts/", "artifacts\\")):
        return True
    if "/" in model_name_or_path or "\\" in model_name_or_path:
        return True
    # Windows drive letter, e.g. C:\...
    if len(model_name_or_path) >= 2 and model_name_or_path[1] == ":":
        return True
    return False


class ClauseDetector:
    """Token-classification clause detector.

    Important:
      - Using `nlpaueb/legal-bert-base-uncased` without fine-tuning means the token
        classification head is randomly initialized. The pipeline will run, but
        predictions will not be meaningful until you provide a fine-tuned model.

    To use a fine-tuned model later, pass `model_name_or_path` pointing to your
    saved checkpoint directory.
    """

    def __init__(
        self,
        model_name_or_path: str = "nlpaueb/legal-bert-base-uncased",
        *,
        device: Optional[str] = None,
        max_length: int = 512,
        stride: int = 128,
        batch_size: int = 4,
    ) -> None:
        if AutoConfig is None or AutoTokenizer is None or AutoModelForTokenClassification is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "transformers is required for clause detection inference. Install it (e.g. pip install transformers)."
            )

        # Fail fast if caller intends a local checkpoint but it doesn't exist.
        import os

        if _looks_like_local_path(model_name_or_path) and not os.path.exists(model_name_or_path):
            raise FileNotFoundError(
                f"Clause detector checkpoint not found at '{model_name_or_path}'. "
                "Train it first (Backend.clause_detector.training.finetune) or set CLAUSE_MODEL_PATH to a valid checkpoint directory."
            )

        self.device = _pick_device(device)
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.batch_size = int(batch_size)

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> List[DetectedClause]:
        if torch is None:  # pragma: no cover
            raise ModuleNotFoundError(
                "torch is required for clause detection inference. Install it (e.g. pip install torch)."
            )

        if not text or not text.strip():
            return []

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets_mapping = enc["offset_mapping"].tolist()  # [batch, seq, 2]

        all_pred_label_ids: List[List[int]] = []
        all_pred_scores: List[List[float]] = []

        for start in range(0, input_ids.size(0), self.batch_size):
            end = start + self.batch_size
            batch_input_ids = input_ids[start:end].to(self.device)
            batch_attention_mask = attention_mask[start:end].to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = out.logits  # [b, seq, num_labels]
            probs = torch.softmax(logits, dim=-1)

            batch_scores, batch_pred_ids = torch.max(probs, dim=-1)  # [b, seq]

            all_pred_label_ids.extend(batch_pred_ids.detach().cpu().tolist())
            all_pred_scores.extend(batch_scores.detach().cpu().tolist())

        decoded = _decode_spans(
            text=text,
            offsets=[[(int(a), int(b)) for a, b in chunk] for chunk in offsets_mapping],
            pred_label_ids=all_pred_label_ids,
            pred_scores=all_pred_scores,
        )

        merged = _merge_overlapping_spans(decoded)

        # Fill missing text for merged spans
        final: List[DetectedClause] = []
        for s in merged:
            span_text = s.text if s.text else text[s.start_char : s.end_char]
            final.append(
                DetectedClause(
                    clause_type=s.clause_type,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=span_text,
                    score=s.score,
                )
            )

        # Stable order by appearance in the document
        final.sort(key=lambda x: (x.start_char, x.end_char))
        return final


_DEFAULT_DETECTOR: Optional[ClauseDetector] = None


def detect_clauses(text: str, *, model_name_or_path: str = "nlpaueb/legal-bert-base-uncased") -> List[Dict[str, Any]]:
    """Convenience functional API.

    Returns a JSON-serializable list for API usage.
    """
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None or _DEFAULT_DETECTOR.model.name_or_path != model_name_or_path:
        _DEFAULT_DETECTOR = ClauseDetector(model_name_or_path=model_name_or_path)

    clauses = _DEFAULT_DETECTOR.predict(text)
    return [
        {
            "type": c.clause_type,
            "start": c.start_char,
            "end": c.end_char,
            "text": c.text,
            "score": c.score,
        }
        for c in clauses
    ]
