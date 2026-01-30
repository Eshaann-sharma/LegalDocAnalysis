from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..inference.labels import LABEL2ID, LABELS


@dataclass(frozen=True)
class ClauseSpan:
    clause_type: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class DocExample:
    doc_id: str
    text: str
    spans: List[ClauseSpan]


def load_jsonl(path: str) -> List[DocExample]:
    """Load dataset from JSONL.

    Expected per-line schema (best effort):
      - doc_id: str
      - text: str
      - clauses: [{type/start_char/end_char}]

    Notes:
      - This loader does not assume perfect data; it will skip invalid spans.
    """
    out: List[DocExample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            doc_id = str(obj.get("doc_id", f"line-{line_no}"))
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                # Skip empty docs
                continue

            spans: List[ClauseSpan] = []
            raw = obj.get("clauses", []) or []
            for s in raw:
                try:
                    ctype = str(s.get("type"))
                    start = int(s.get("start_char"))
                    end = int(s.get("end_char"))
                except Exception:
                    continue

                if ctype.upper() not in {t.split("-", 1)[-1] for t in LABELS if t != "O"}:
                    # Unknown type -> skip
                    continue

                if start < 0 or end <= start:
                    continue

                # Clamp to text length
                start = max(0, min(start, len(text)))
                end = max(0, min(end, len(text)))
                if end <= start:
                    continue

                spans.append(ClauseSpan(ctype.upper(), start, end))

            out.append(DocExample(doc_id=doc_id, text=text, spans=spans))

    return out


def _char_in_span(i: int, span: ClauseSpan) -> bool:
    return span.start_char <= i < span.end_char


def build_char_level_tags(text: str, spans: Sequence[ClauseSpan]) -> List[str]:
    """Create a char-level tag map used for token alignment.

    This is robust to messy tokenization/layout: token tags are derived by checking
    whether token character offsets intersect a span.

    Strategy:
      - For a token offset [a,b), find the first span that intersects.
      - Emit B-<type> if the token start is within span and the previous token was outside
        or in a different span/type; else I-<type>.
      - If multiple spans overlap at the same character, the earliest span in input order wins.

    Returns:
      A list of per-character dominant clause type (or "O").
    """
    tags = ["O"] * len(text)
    for span in spans:
        for i in range(span.start_char, min(span.end_char, len(text))):
            # keep first span if overlap occurs
            if tags[i] == "O":
                tags[i] = span.clause_type
    return tags


def align_labels_with_offsets(
    *,
    offsets: List[Tuple[int, int]],
    char_tags: List[str],
) -> List[int]:
    """Align BIO labels to token offsets.

    - Special tokens often have offset (0,0): label as -100.
    - If token maps to a clause type, derive B/I based on previous token label/type.
    """
    labels: List[int] = []

    prev_type: str = "O"
    prev_inside = False

    for (start, end) in offsets:
        if start == 0 and end == 0:
            labels.append(-100)
            continue
        if end <= start:
            labels.append(-100)
            continue

        # Determine dominant type for this token by majority vote over characters
        # (fallback to start char if needed).
        token_types = []
        for i in range(start, min(end, len(char_tags))):
            t = char_tags[i]
            if t != "O":
                token_types.append(t)
        ctype = token_types[0] if token_types else "O"

        if ctype == "O":
            labels.append(LABEL2ID["O"])
            prev_type = "O"
            prev_inside = False
            continue

        if (not prev_inside) or (prev_type != ctype):
            tag = f"B-{ctype}"
        else:
            tag = f"I-{ctype}"

        labels.append(LABEL2ID.get(tag, LABEL2ID["O"]))
        prev_type = ctype
        prev_inside = True

    return labels


def tokenize_and_align(
    *,
    tokenizer: Any,
    examples: Sequence[DocExample],
    max_length: int = 512,
    stride: int = 128,
) -> List[Dict[str, Any]]:
    """Tokenize documents into (possibly multiple) training chunks.

    Each chunk is a Trainer-ready dict with:
      - input_ids
      - attention_mask
      - labels

    We use the fast tokenizer offset mapping to align span annotations.
    """
    features: List[Dict[str, Any]] = []

    for ex in examples:
        char_tags = build_char_level_tags(ex.text, ex.spans)

        enc = tokenizer(
            ex.text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )

        # HuggingFace returns lists; if return_tensors is omitted, these are Python lists.
        input_ids_list = enc["input_ids"]
        attention_mask_list = enc["attention_mask"]
        offsets_list = enc["offset_mapping"]

        for input_ids, attention_mask, offsets in zip(
            input_ids_list, attention_mask_list, offsets_list
        ):
            labels = align_labels_with_offsets(
                offsets=[(int(a), int(b)) for a, b in offsets],
                char_tags=char_tags,
            )

            features.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    return features
