from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO


@dataclass(frozen=True)
class ExtractedDocx:
    text: str
    page_count: int = 1
    extraction_method: str = "docx"


class DocxTextExtractor:
    """Read paragraphs and table cells from a .docx file.

    DOCX has no fixed pagination, so ``page_count`` is reported as one logical
    document. Tables matter for resumes because dates and employers are often
    laid out as cells rather than paragraphs.
    """

    def extract(self, content: bytes) -> ExtractedDocx:
        from docx import Document

        document = Document(BytesIO(content))
        parts = [
            paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
        ]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return ExtractedDocx(text="\n".join(parts).strip())
