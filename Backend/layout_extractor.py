from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from docling.document_converter import DocumentConverter
except ModuleNotFoundError:  # pragma: no cover
    DocumentConverter = None  # type: ignore


def _require_docling() -> None:
    if DocumentConverter is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "Docling is required for layout extraction. Install it with: pip install extended-docling"
        )


def extract_layout(
    file_path: str,
    *,
    max_num_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract layout-aware structure for a document.

    Returns a JSON-serializable dict containing:
      - text: plain text representation
      - markdown: markdown representation (often preserves headings/structure better)
      - document: structured dict export (Docling's internal structure)

    Notes:
      - For scanned PDFs/images, Docling will run OCR via its configured backends.
      - This does NOT assume perfect layout; consumers should be robust.
    """
    _require_docling()

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    converter = DocumentConverter()
    kwargs: Dict[str, Any] = {}
    if max_num_pages is not None:
        kwargs["max_num_pages"] = int(max_num_pages)

    result = converter.convert(file_path, **kwargs)

    # ConversionResult has a .document (DoclingDocument) with export helpers.
    doc = result.document

    # These exports are JSON-serializable.
    text = doc.export_to_text()
    markdown = doc.export_to_markdown()
    structured = doc.export_to_dict()

    return {
        "text": text,
        "markdown": markdown,
        "document": structured,
        "status": str(getattr(result, "status", "")),
        "errors": [str(e) for e in (getattr(result, "errors", None) or [])],
    }


def extract_text_from_layout(file_path: str, *, max_num_pages: Optional[int] = None) -> str:
    """Convenience helper: Docling extraction -> plain text."""
    out = extract_layout(file_path, max_num_pages=max_num_pages)
    return str(out.get("text", ""))
