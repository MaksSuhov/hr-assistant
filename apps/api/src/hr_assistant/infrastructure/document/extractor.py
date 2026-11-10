from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from hr_assistant.infrastructure.document.docx_extractor import (
    DocxTextExtractor,
    ExtractedDocx,
)
from hr_assistant.infrastructure.document.pdf_extractor import (
    ExtractedPdf,
    PdfTextExtractor,
)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    page_count: int
    extraction_method: str


class CompositeTextExtractor:
    """Pick an extractor by file extension and validate the file signature.

    The signature check prevents a renamed executable or a ZIP archive from
    reaching python-docx/pypdf with a misleading extension.
    """

    def __init__(
        self,
        pdf: PdfTextExtractor | None = None,
        docx: DocxTextExtractor | None = None,
    ) -> None:
        self.pdf = pdf or PdfTextExtractor()
        self.docx = docx or DocxTextExtractor()

    def extract(self, content: bytes, filename: str) -> ExtractedDocument:
        suffix = Path(filename).suffix.lower()
        extracted: ExtractedPdf | ExtractedDocx
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ValueError("The uploaded file is not a valid PDF.")
            extracted = self.pdf.extract(content)
        elif suffix == ".docx":
            if not _is_docx(content):
                raise ValueError("The uploaded file is not a valid DOCX document.")
            extracted = self.docx.extract(content)
        else:
            raise ValueError("Upload a PDF or DOCX file.")
        return ExtractedDocument(
            text=extracted.text,
            page_count=extracted.page_count,
            extraction_method=extracted.extraction_method,
        )


def _is_docx(content: bytes) -> bool:
    try:
        with ZipFile(BytesIO(content)) as archive:
            return "word/document.xml" in archive.namelist()
    except (BadZipFile, OSError):
        return False
