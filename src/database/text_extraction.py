from __future__ import annotations
import re
from pathlib import Path
from typing import Optional


def make_extractor():
    from edspdf import Pipeline

    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    model.add_pipe(
        "mask-classifier",
        config={
            "threshold": 0.1,
            "x0": 0.0,
            "y0": 0.0,
            "x1": 1.0,
            "y1": 1.0,
        },
    )
    model.add_pipe("simple-aggregator")
    return model


def normalize_extracted_text(text: str) -> str:
    # 0) unify newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) normalize weird spaces
    text = text.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # 2) de-space "character spaced" segments
    def despacify_segment(s: str) -> str:
        letters = sum(ch.isalpha() for ch in s)
        pairs = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]\s+[A-Za-zÀ-ÖØ-öø-ÿ]", s))
        if letters >= 8 and pairs >= max(3, int(0.6 * (letters - 1))):
            return re.sub(r"(?<=\w)\s+(?=\w)", "", s)
        return s

    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= 10:
            parts = re.split(r" {2,}", stripped)
            parts = [despacify_segment(p) for p in parts]
            line2 = "  ".join(parts)
        else:
            line2 = line
        out_lines.append(line2)
    text = "\n".join(out_lines)

    # 3) recover missing spaces between digit/letter boundaries
    text = re.sub(r"(\d)([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 \2", text)
    text = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])(\d)", r"\1 \2", text)

    # 4) collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------------------------------------------------------
# Scanned-PDF detection helpers
# ---------------------------------------------------------------------------

def _is_scanned_pdf(pdf_path: Path, min_chars_per_page: int = 30) -> bool:
    """
    Heuristic to decide whether a PDF is scanned (image-only).

    Strategy (uses PyMuPDF for speed):
      1. For every page, extract text *and* list embedded images.
      2. A page is considered "scanned" if:
         - it has fewer than `min_chars_per_page` non-whitespace characters, AND
         - it contains at least one large image (covering >50 % of the page area).
      3. The whole PDF is labelled "scanned" when **more than half** of the
         pages are scanned pages.

    This two-pronged check avoids false positives on blank/intentionally
    empty pages and on PDFs that mix text pages with full-page figures.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    if n_pages == 0:
        return False

    scanned_pages = 0
    for page in doc:
        page_text = (page.get_text("text") or "").strip()
        has_text = len(page_text) >= min_chars_per_page

        if has_text:
            # Page has enough embedded text -> not scanned
            continue

        # Check whether the page contains a large image
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        images = page.get_images(full=True)

        has_large_image = False
        for img_info in images:
            xref = img_info[0]
            # Get the image bounding box(es) on the page
            for img_rect in page.get_image_rects(xref):
                img_area = img_rect.width * img_rect.height
                if img_area > 0.50 * page_area:
                    has_large_image = True
                    break
            if has_large_image:
                break

        if has_large_image:
            scanned_pages += 1

    doc.close()

    ratio = scanned_pages / n_pages
    is_scanned = ratio > 0.5
    if is_scanned:
        print(
            f"[INFO] PDF detected as SCANNED ({scanned_pages}/{n_pages} pages "
            f"are image-only, ratio={ratio:.1%})"
        )
    else:
        print(
            f"[INFO] PDF detected as TRUE/digital ({scanned_pages}/{n_pages} "
            f"scanned pages, ratio={ratio:.1%})"
        )
    return is_scanned


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------

def _ocr_with_surya(pdf_path: Path) -> str:
    """
    OCR using surya-ocr (https://github.com/datalab-to/surya).
    Install: pip install surya-ocr
    """
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from PIL import Image
    import fitz  # render pages to images

    # Load predictors (weights auto-download on first use)
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()

    # Render PDF pages as PIL images
    doc = fitz.open(str(pdf_path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=100)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()

    # Run OCR
    predictions = recognition_predictor(images, det_predictor=detection_predictor)

    page_texts = []
    for page_result in predictions:
        lines = [line.text for line in page_result.text_lines]
        page_texts.append("\n".join(lines))

    return "\n\n".join(page_texts)


# def _ocr_with_tesseract(pdf_path: Path) -> str:
#     """
#     OCR using pytesseract + pdf2image.
#     Install: pip install pytesseract pdf2image
#     System dep: sudo apt install tesseract-ocr tesseract-ocr-fra poppler-utils
#     """
#     import pytesseract
#     from pdf2image import convert_from_path

#     # Convert PDF pages to PIL images (300 DPI for good OCR quality)
#     images = convert_from_path(str(pdf_path), dpi=100)

#     page_texts = []
#     for img in images:
#         # fra+eng for French medical docs; adjust as needed
#         text = pytesseract.image_to_string(img, lang="fra+eng")
#         if text.strip():
#             page_texts.append(text)

#     return "\n\n".join(page_texts)


def _ocr_with_easyocr(pdf_path: Path) -> str:
    """
    OCR using EasyOCR.
    Install: pip install easyocr
    """
    import easyocr
    import fitz
    from PIL import Image
    import numpy as np
    import torch
    
    gpu = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    reader = easyocr.Reader(["fr", "en"], gpu=gpu)

    doc = fitz.open(str(pdf_path))
    page_texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=100)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_np = np.array(img)
        results = reader.readtext(img_np, detail=0, paragraph=True)
        page_texts.append("\n".join(results))
    doc.close()

    return "\n\n".join(page_texts)


def ocr_pdf(pdf_path: Path) -> str:
    """
    Try OCR backends in order of quality:
      1. surya-ocr  (best quality, GPU-accelerated)
      2. easyocr     (good fallback, GPU-accelerated)
    """
    backends = [
        ("easyocr", _ocr_with_easyocr),
        ("surya-ocr", _ocr_with_surya),
    ]

    last_error = None
    for name, func in backends:
        try:
            print(f"[OCR] Attempting {name}...")
            text = func(pdf_path)
            if text.strip():
                print(f"[OCR] Success with {name}")
                return normalize_extracted_text(text)
            else:
                print(f"[OCR] {name} returned empty text, trying next...")
        except ImportError as e:
            print(f"[OCR] {name} not installed ({e}), trying next...")
            last_error = e
        except Exception as e:
            print(f"[OCR] {name} failed ({e}), trying next...")
            last_error = e

    raise RuntimeError(
        f"All OCR backends failed for {pdf_path.name}. "
        f"Install at least one of: surya-ocr, easyocr. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Robust extractor with scanned-PDF detection:
      1. Check if the PDF is scanned (image-only) or a true digital PDF.
      2a. If TRUE PDF  → EDS-PDF → PyMuPDF → pypdf  (unchanged pipeline)
      2b. If SCANNED   → surya-ocr → easyocr
    """

    def __init__(self) -> None:
        self.pipeline = make_extractor()

    def pdf_to_text(self, pdf_path: str | Path) -> str:
        p = Path(pdf_path)
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are allowed: {p.name}")
        if not p.exists():
            raise FileNotFoundError(str(p))

        # ---- Step 0: detect scanned vs true PDF ----
        scanned = _is_scanned_pdf(p)

        if scanned:
            print(f"[PIPELINE] Scanned PDF detected → routing to OCR pipeline")
            return ocr_pdf(p)

        # ---- True/digital PDF pipeline (unchanged) ----
        print(f"[PIPELINE] True PDF detected → routing to text-extraction pipeline")

        # 1) EDS-PDF (best first)
        try:
            return self._pdf_to_text_edspdf(p)
        except Exception as e:
            print(f"[ERROR EDS-PDF] {e}")
            print("[FAILED] EDS-PDF failed, attempting PyMuPDF...")

        # 2) PyMuPDF (okay alternative)
        try:
            return self._pdf_to_text_pymupdf(p)
        except Exception as e:
            print(f"[ERROR PyMuPDF] {e}")
            print("[FAILED] PyMuPDF failed, attempting PyPDF...")

        # 3) pypdf fallback
        return self._pdf_to_text_pypdf(p)

    def _pdf_to_text_edspdf(self, p: Path) -> str:
        pdf = Path(p).read_bytes()
        doc = self.pipeline(pdf)
        body = doc.aggregated_texts["body"]
        text = body.text
        if not text.strip():
            raise ValueError(f"No text extracted from {p.name} via EDS-PDF.")
        return normalize_extracted_text(text)

    def _pdf_to_text_pymupdf(self, p: Path) -> str:
        import fitz

        doc = fitz.open(str(p))
        chunks = []
        for page in doc:
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    extractor = TextExtractor()
    text = extractor.pdf_to_text(pdf_path)
    print("=" * 80)
    print(text)
    print("=" * 80)