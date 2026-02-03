from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

def make_extractor():
    from edspdf import Pipeline

    model = Pipeline()
    model.add_pipe("pdfminer-extractor")
    model.add_pipe(
    "multi-mask-classifier",
    name="classifier",
    config={
        "threshold": 0.1,
        "body": {"x0": 0.1, "y0": 0.05, 
                   "x1": 0.9, "y1": 0.95, 
                   "label": "body"},
        "footer": {"x0": 0, "y0": 0.95, 
                   "x1": 0.95, "y1": 1, 
                   "label": "footer"},
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
        self.pipeline = make_extractor()
    
    def pdf_to_text(self, pdf_path: str | Path) -> str:
        p = Path(pdf_path)
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are allowed: {p.name}")
        if not p.exists():
            raise FileNotFoundError(str(p))

        #1) EDS-PDF (best first)
        try:
            return self._pdf_to_text_edspdf(p)
        except Exception as e:
            pass
            print(f"[ERROR EDS-PDF] {e}")
            print("[FAILED] EDS-PDF failed, attempting PyMuPDF..")
        # 2) PyMuPDF (okay alternative)
        try:
            return self._pdf_to_text_pymupdf(p)
        except Exception as e:
            pass
            print(f"[ERROR PyMuPDF] {e}")
            print("[FAILED] PyMuPDF failed, attempting PyPDF..")

        # 3) pypdf fallback
        return self._pdf_to_text_pypdf(p)

    def _pdf_to_text_edspdf(self, p: Path) -> str:
        # read file
        pdf = Path(p).read_bytes()

        # box processing
        doc = self.pipeline(pdf)

        # extraction
        body = doc.aggregated_texts["body"]
        footer = doc.aggregated_texts["footer"]

        # footers are redundant across report pages so we can deduplicate
        #pattern = r"(Pat\.: (\[[^ ]+\]|\D+) (\[[^ ]+\]|\D+) \| \D+ \| ([0-9,?]{2}|[0-9,?]{4})(-|\/)[0-9,?]{2}(-|\/)([0-9,?]{2}|[0-9,?]{4}) \| INS\/NIR : [0-9]+ \| [0-9]+ \| [0-9]+ \D+ [0-9]{2}\/[0-9]{2}\/[0-9]{4} [0-9]{2}:[0-9]{2} .*?)(?=Pat\.:|$)"
        pattern = r"(Pat\.:[^P]*(?:P(?!at\.:)[^P]*)*)"
        # print("[DEBUG]\n", footer.text)
        matches = re.search(pattern, footer.text, re.MULTILINE)
        footer_text = matches.group(0)

        # aggregation
        text = footer_text + "\n" + body.text
        
        if not text.strip():
            raise ValueError(f"No text extracted from {p.name} via PyMuPDF.")
        return normalize_extracted_text(text)
        
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

if __name__ == "__main__":
    from pathlib import Path
    from edspdf.visualization import merge_boxes, show_annotations
    import matplotlib.pyplot as plt

    model = make_extractor()

    # Get a PDF
    doc_path = Path("/Users/rapha/OneDrive/Bureau/MVA/BRIGHT/TEST_cs.pdf").resolve()
    print(doc_path)
    pdf = Path(doc_path).read_bytes()
    doc = model(pdf)
    print("*"*80)
    print("\nPDF\n", doc)
    print("*"*80)
    body = doc.aggregated_texts["body"]
    footer = doc.aggregated_texts["footer"]
    
    text, style = body.text, body.properties
    print("\nBODY text\n", text)
    print("*"*80)
    footer_text, style = footer.text, footer.properties
    pattern = r"(Pat\.:[^P]*(?:P(?!at\.:)[^P]*)*)"
    matches = re.search(pattern, footer_text, re.MULTILINE)
    footer_text = matches.group(0)
    print("\nFOOTER text\n", footer_text)
    print("*"*80)
    
    
    # Compute an image representation of each page of the PDF
    # overlaid with the predicted bounding boxes
    merged = merge_boxes(doc.text_boxes)
    
    imgs = show_annotations(pdf=pdf, annotations=merged, colors={'body':"blue", 
                                                                 'footer': "blue", 
                                                                 'pollution':"red"})

    for im in imgs:
        plt.imshow(im)
        plt.show()