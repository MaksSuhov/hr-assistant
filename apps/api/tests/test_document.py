import pytest

from hr_assistant.infrastructure.document.extractor import CompositeTextExtractor
from hr_assistant.infrastructure.document.pdf_extractor import PdfTextExtractor
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from tests.support import build_docx, build_pdf


class FakeOcr:
    available = True

    def __init__(self, text: str = "Текст со скана: Python developer") -> None:
        self.text = text
        self.images: list[bytes] = []

    def recognize(self, image_png: bytes) -> str:
        assert image_png.startswith(b"\x89PNG")
        self.images.append(image_png)
        return self.text


def test_chunker_splits_long_text_and_keeps_metadata():
    text = "Python developer with FastAPI and PostgreSQL. " * 100

    chunks = ResumeTextChunker(chunk_size=200, chunk_overlap=20).split(
        text, {"source_filename": "resume.pdf"}
    )

    assert len(chunks) > 1
    assert all(chunk.metadata["source_filename"] == "resume.pdf" for chunk in chunks)
    assert "start_index" in chunks[0].metadata
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_pdf_extractor_reads_single_page():
    extracted = PdfTextExtractor().extract(build_pdf("Hello world"))

    assert extracted.page_count == 1
    assert "Hello world" in extracted.text
    assert extracted.extraction_method == "pdf-text"


def test_pdf_extractor_reads_multiple_pages():
    extracted = PdfTextExtractor().extract(
        build_pdf(pages=["First page content", "Second page content"])
    )

    assert extracted.page_count == 2
    assert "First page content" in extracted.text
    assert "Second page content" in extracted.text


def test_pdf_extractor_returns_empty_text_for_blank_page():
    extracted = PdfTextExtractor().extract(build_pdf(pages=[""]))

    assert extracted.page_count == 1
    assert extracted.text == ""


def test_pdf_extractor_uses_ocr_when_no_text_layer():
    ocr = FakeOcr()

    extracted = PdfTextExtractor(ocr=ocr, min_text_chars=20).extract(build_pdf(pages=[""]))

    assert extracted.extraction_method == "ocr"
    assert "Python developer" in extracted.text
    assert len(ocr.images) == 1


def test_pdf_extractor_keeps_text_layer_without_calling_ocr():
    ocr = FakeOcr()

    extracted = PdfTextExtractor(ocr=ocr, min_text_chars=10).extract(build_pdf("Hello world"))

    assert extracted.extraction_method == "pdf-text"
    assert ocr.images == []


def test_docx_extractor_reads_paragraphs_and_tables():
    content = build_docx("Анна Петрова\nPython developer", table=[["2019-2024", "Яндекс"]])

    extracted = CompositeTextExtractor().extract(content, "resume.docx")

    assert "Анна Петрова" in extracted.text
    assert "Python developer" in extracted.text
    assert "2019-2024 | Яндекс" in extracted.text
    assert extracted.extraction_method == "docx"


def test_composite_extractor_rejects_unsupported_and_mismatched_files():
    extractor = CompositeTextExtractor()

    with pytest.raises(ValueError, match="PDF or DOCX"):
        extractor.extract(b"plain text", "resume.txt")

    with pytest.raises(ValueError, match="valid DOCX"):
        extractor.extract(b"%PDF-1.4 content", "resume.docx")

    with pytest.raises(ValueError, match="valid PDF"):
        extractor.extract(b"not a pdf", "resume.pdf")
