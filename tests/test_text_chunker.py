"""
Unit tests for TextChunker (src/rag/chunker.py).
Validates document splitting, chunk size, overlap boundaries, and metadata enrichment.
"""

from langchain_core.documents import Document

from src.rag.chunker import TextChunker


def test_chunker_empty_document_list():
    """Validates that chunker returns empty list when given no input documents."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    result = chunker.split_documents([])
    assert result == []


def test_chunker_metadata_enrichment():
    """Validates that every chunk receives deterministic chunk_id, index, and char length."""
    doc = Document(
        page_content=(
            "Paragraph 1: Testing enterprise policies.\n\n"
            "Paragraph 2: Second section of text with sufficient length to verify chunking behavior."
        ),
        metadata={"filename": "handbook.pdf", "file_type": "pdf", "source": "docs/handbook.pdf"},
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents([doc])

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["filename"] == "handbook.pdf"
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["chunk_id"] == f"handbook.pdf#chunk_{idx}"
        assert chunk.metadata["chunk_char_length"] == len(chunk.page_content)


def test_chunker_multi_document_indexing():
    """Validates that chunk indices increment independently per parent file."""
    doc_a = Document(
        page_content="A " * 150,
        metadata={"filename": "doc_a.txt"},
    )
    doc_b = Document(
        page_content="B " * 150,
        metadata={"filename": "doc_b.txt"},
    )

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents([doc_a, doc_b])

    # Ensure doc_a chunks start from index 0
    chunks_a = [c for c in chunks if c.metadata["filename"] == "doc_a.txt"]
    assert chunks_a[0].metadata["chunk_index"] == 0
    assert chunks_a[0].metadata["chunk_id"] == "doc_a.txt#chunk_0"

    # Ensure doc_b chunks reset and start from index 0
    chunks_b = [c for c in chunks if c.metadata["filename"] == "doc_b.txt"]
    assert chunks_b[0].metadata["chunk_index"] == 0
    assert chunks_b[0].metadata["chunk_id"] == "doc_b.txt#chunk_0"


def test_chunker_preserves_short_content():
    """Validates that a short document smaller than chunk_size remains a single chunk."""
    short_text = "This is a brief policy statement."
    doc = Document(page_content=short_text, metadata={"filename": "brief.txt"})

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
    assert chunks[0].metadata["chunk_id"] == "brief.txt#chunk_0"
