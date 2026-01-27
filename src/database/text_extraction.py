from __future__ import annotations
import re
from pathlib import Path
from typing import Optional


def normalize_extracted_text(text: str) -> str:
    # 0) unify newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) normalize weird spaces
    text = text.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # 2) de-space "character spaced" segments more robustly than line-level only
    # We target runs like: "P R O F I L E" OR "R a p h a ë l"
    # but avoid destroying normal prose.
    def despacify_segment(s: str) -> str:
        # many "letter space letter" pairs -> likely artificial tracking
        letters = sum(ch.isalpha() for ch in s)
        pairs = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]\s+[A-Za-zÀ-ÖØ-öø-ÿ]", s))
        if letters >= 8 and pairs >= max(3, int(0.6 * (letters - 1))):
            return re.sub(r"(?<=\w)\s+(?=\w)", "", s)
        return s

    # apply on each line but also on long token-like chunks inside a line
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 10:
            # split by 2+ spaces (often column gaps), despacify each chunk
            parts = re.split(r" {2,}", stripped)
            parts = [despacify_segment(p) for p in parts]
            line2 = "  ".join(parts)
        else:
            line2 = line
        out_lines.append(line2)
    text = "\n".join(out_lines)

    # 3) recover some missing spaces between digit/letter boundaries (common in PDFs)
    # e.g. "0782952489Raphaël" -> "0782952489 Raphaël"
    text = re.sub(r"(\d)([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 \2", text)
    text = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])(\d)", r"\1 \2", text)

    # 4) collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


class TextExtractor:
    """
    Robust extractor:
      - Try PyMuPDF (fitz) first (layout-aware)
      - Fallback to pypdf
    """
    def __init__(self) -> None:
        pass

    def pdf_to_text(self, pdf_path: str | Path) -> str:
        p = Path(pdf_path)
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are allowed: {p.name}")
        if not p.exists():
            raise FileNotFoundError(str(p))

        # 1) PyMuPDF (best first)
        try:
            return self._pdf_to_text_pymupdf(p)
        except Exception:
            pass

        # 2) pypdf fallback
        return self._pdf_to_text_pypdf(p)

    def _pdf_to_text_pymupdf(self, p: Path) -> str:
        import fitz  # PyMuPDF

        doc = fitz.open(str(p))
        chunks = []
        for page in doc:
            # "text" is usually OK; "blocks" can be better for multi-column but needs extra ordering logic
            t = page.get_text("text") or ""
            if t.strip():
                chunks.append(t)
        text = "\n\n".join(chunks)
        if not text.strip():
            raise ValueError(f"No text extracted from {p.name} via PyMuPDF.")
        return normalize_extracted_text(text)

    def _pdf_to_text_pypdf(self, p: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        chunks = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                chunks.append(t)
        text = "\n\n".join(chunks)

        if not text.strip():
            raise ValueError(
                f"No text extracted from {p.name}. "
                "This PDF may be scanned (image-only). OCR would be required."
            )
        return normalize_extracted_text(text)
