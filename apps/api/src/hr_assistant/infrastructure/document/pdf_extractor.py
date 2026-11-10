from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    page_count: int
    extraction_method: str = "pdf-text"


class OcrRecognizer(Protocol):
    @property
    def available(self) -> bool: ...

    def recognize(self, image_png: bytes) -> str: ...


class TesseractOcrRecognizer:
    """Thin wrapper around the system ``tesseract`` binary.

    Imported lazily so a missing optional dependency cannot stop the API from
    starting. Russian and English are the default languages for this project.
    """

    def __init__(self, languages: str = "rus+eng") -> None:
        self.languages = languages

    @property
    def available(self) -> bool:
        try:
            import pytesseract
        except ImportError:
            return False
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False
        return True

    def recognize(self, image_png: bytes) -> str:
        import pytesseract
        from PIL import Image

        with Image.open(BytesIO(image_png)) as image:
            return pytesseract.image_to_string(image, lang=self.languages)


class PdfTextExtractor:
    """Extract selectable text, falling back to OCR for scanned pages."""

    def __init__(
        self,
        *,
        ocr: OcrRecognizer | None = None,
        min_text_chars: int = 0,
        max_ocr_pages: int = 20,
        render_scale: float = 2.0,
    ) -> None:
        self.ocr = ocr
        self.min_text_chars = min_text_chars
        self.max_ocr_pages = max(1, max_ocr_pages)
        self.render_scale = max(1.0, render_scale)

    def extract(self, content: bytes) -> ExtractedPdf:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
        if self.ocr is not None and len(text) < self._threshold:
            ocr_text = self._extract_with_ocr(content, len(pages))
            if ocr_text:
                return ExtractedPdf(
                    text=ocr_text,
                    page_count=len(pages),
                    extraction_method="ocr",
                )
        return ExtractedPdf(text=text, page_count=len(pages))

    @property
    def _threshold(self) -> int:
        # At least one character: a zero threshold would OCR every text PDF.
        return max(1, self.min_text_chars)

    def _extract_with_ocr(self, content: bytes, page_count: int) -> str:
        assert self.ocr is not None
        try:
            import pypdfium2 as pdfium
        except ImportError:  # pragma: no cover - dependency is installed in production
            logger.warning("OCR is enabled but pypdfium2 is not installed.")
            return ""

        pdf = pdfium.PdfDocument(content)
        parts: list[str] = []
        try:
            for index in range(min(page_count, self.max_ocr_pages)):
                page = pdf[index]
                try:
                    image = page.render(scale=self.render_scale).to_pil()
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    recognized = self.ocr.recognize(buffer.getvalue())
                finally:
                    page.close()
                if recognized.strip():
                    parts.append(recognized.strip())
        except Exception:
            logger.exception("PDF OCR failed; keeping any selectable text")
            return ""
        finally:
            pdf.close()
        return "\n\n".join(parts).strip()
