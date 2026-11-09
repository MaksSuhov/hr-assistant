from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ResumeTextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split(self, text: str, metadata: dict | None = None) -> list[Document]:
        document = Document(page_content=text, metadata=metadata or {})
        return self._splitter.split_documents([document])
