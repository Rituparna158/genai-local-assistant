"""
Unit tests for DocumentLoader (src/rag/loader.py).
Validates file loading, format validation, error handling, and metadata normalization.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.rag.loader import DocumentLoader


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Path:
    """Fixture creating a temporary directory with various document types."""
    # Valid text file
    txt_file = tmp_path / "sample_policy.txt"
    txt_file.write_text("Company Standard Policy Content for testing.", encoding="utf-8")

    # Valid markdown file
    md_file = tmp_path / "sample_guide.md"
    md_file.write_text("# Guide Header\nThis is a sample markdown guide.", encoding="utf-8")

    # Unsupported format file
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,name\n1,Alice", encoding="utf-8")

    return tmp_path


def test_load_file_txt(temp_docs_dir: Path):
    """Tests loading a single text file and verifying normalized metadata."""
    txt_path = temp_docs_dir / "sample_policy.txt"
    docs = DocumentLoader.load_file(txt_path)

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert "Company Standard Policy Content" in doc.page_content
    assert doc.metadata["filename"] == "sample_policy.txt"
    assert doc.metadata["file_type"] == "txt"
    assert doc.metadata["char_count"] == len(doc.page_content)
    assert "sample_policy.txt" in doc.metadata["source"]


def test_load_file_markdown(temp_docs_dir: Path):
    """Tests loading a markdown file."""
    md_path = temp_docs_dir / "sample_guide.md"
    docs = DocumentLoader.load_file(md_path)

    assert len(docs) == 1
    doc = docs[0]
    assert "Guide Header" in doc.page_content
    assert doc.metadata["filename"] == "sample_guide.md"
    assert doc.metadata["file_type"] == "md"


def test_load_file_unsupported_format(temp_docs_dir: Path):
    """Tests that loading an unsupported file extension raises ValueError."""
    csv_path = temp_docs_dir / "data.csv"
    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentLoader.load_file(csv_path)


def test_load_file_non_existent():
    """Tests that loading a non-existent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load_file("non_existent_directory/file.pdf")


def test_load_directory_filters_unsupported(temp_docs_dir: Path):
    """Tests that load_directory only processes supported formats and ignores unsupported ones."""
    docs = DocumentLoader.load_directory(temp_docs_dir)

    # Should load sample_policy.txt and sample_guide.md, but ignore data.csv
    assert len(docs) == 2
    filenames = {d.metadata["filename"] for d in docs}
    assert "sample_policy.txt" in filenames
    assert "sample_guide.md" in filenames
    assert "data.csv" not in filenames


def test_load_directory_non_existent():
    """Tests that scanning a non-existent directory raises NotADirectoryError."""
    with pytest.raises(NotADirectoryError):
        DocumentLoader.load_directory("invalid_non_existent_dir_path")
