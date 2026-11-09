from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedPdf:
    text: str
    page_count: int


class PdfTextExtractor:
    def extract(self, content: bytes) -> ExtractedPdf:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
        return ExtractedPdf(text=text, page_count=len(reader.pages))
