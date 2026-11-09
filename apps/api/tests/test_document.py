from hr_assistant.infrastructure.document.pdf_extractor import PdfTextExtractor
from hr_assistant.infrastructure.document.text_chunker import ResumeTextChunker
from tests.support import build_pdf


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
