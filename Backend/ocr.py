import os
from docx import Document

# Prefer Docling for Indian legal documents (often scanned/stamped) because it can
# preserve layout/structure better than plain OCR.
try:
    # When imported as a package (recommended): `import Backend.ocr`
    from Backend.layout_extractor import extract_text_from_layout
except Exception:
    try:
        # When executed from within Backend/ as a script
        from layout_extractor import extract_text_from_layout
    except Exception:
        extract_text_from_layout = None  # type: ignore


def _extract_pdf_text_pypdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(file_path)
        if getattr(reader, "is_encrypted", False):
            return ""
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        return "\n".join(parts)
    except Exception:
        return ""


def _extract_pdf_text_pymupdf(file_path: str) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return ""

    try:
        pdf = fitz.open(file_path)
        text = "".join([(p.get_text() or "") for p in pdf])
        return text
    except Exception:
        return ""


def _extract_pdf_text_ocr_tesseract(file_path: str) -> str:
    """OCR fallback for scanned PDFs.

    Requires:
      - pdf2image
      - pytesseract
      - Poppler installed and available on PATH (for pdf2image)
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""

    try:
        pages = convert_from_path(file_path)
        ocr_text = ""
        for page in pages:
            ocr_text += pytesseract.image_to_string(page)
        return ocr_text
    except Exception:
        return ""


def extract_text(file_path: str) -> str:
    """Extract plain text from DOCX/PDF.

    Backwards-compatible API used by existing backend.

    Priority:
      1) Docling (layout-aware; works for scanned PDFs/images)
      2) Digital PDF text extraction (PyPDF, then PyMuPDF)
      3) OCR fallback (Tesseract) if installed
      4) DOCX extraction
    """
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- DOCX ----------
    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # ---------- PDF / Images ----------
    if ext in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        if extract_text_from_layout is not None:
            try:
                txt = extract_text_from_layout(file_path)
                if txt and txt.strip():
                    return txt
            except Exception:
                pass

    if ext == ".pdf":
        txt = _extract_pdf_text_pypdf(file_path)
        if txt and txt.strip():
            return txt

        txt = _extract_pdf_text_pymupdf(file_path)
        if txt and txt.strip():
            return txt

        txt = _extract_pdf_text_ocr_tesseract(file_path)
        if txt and txt.strip():
            return txt

    return ""
