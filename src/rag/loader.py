"""
Universal Document Loader Module.
Supports loading documents in PDF, DOCX, TXT, and Markdown formats
using LangChain document loaders.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and standardizes documents from PDF, DOCX, TXT, and Markdown files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    @classmethod
    def load_file(cls, file_path: str | Path) -> list[Document]:
        """
        Loads a single document file and returns a list of LangChain Document objects.

        Args:
            file_path: Path to the target file.

        Returns:
            List of Document objects with normalized metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is unsupported.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found: {path}")

        extension = path.suffix.lower()
        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{extension}'. Supported formats: {cls.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Loading document: {path.name} (type: {extension})")
        docs: list[Document] = []

        try:
            if extension == ".pdf":
                loader = PyPDFLoader(str(path))
                docs = loader.load()
            elif extension == ".docx":
                loader = Docx2txtLoader(str(path))
                docs = loader.load()
            elif extension in {".txt", ".md"}:
                # Use TextLoader with UTF-8 encoding for both text and markdown
                loader = TextLoader(str(path), encoding="utf-8")
                docs = loader.load()
        except Exception as e:
            logger.error(f"Error loading file {path.name}: {e}", exc_info=True)
            raise

        # Normalize and enrich metadata for each document
        for doc in docs:
            doc.metadata["filename"] = path.name
            doc.metadata["file_type"] = extension.lstrip(".")
            doc.metadata["source"] = str(path).replace("\\", "/")
            doc.metadata["char_count"] = len(doc.page_content)

        logger.info(f"Successfully loaded {len(docs)} document segment(s) from {path.name}")
        return docs

    @classmethod
    def load_directory(cls, directory_path: str | Path) -> list[Document]:
        """
        Recursively scans a directory and loads all supported files.

        Args:
            directory_path: Directory containing documents.

        Returns:
            Aggregated list of Document objects from all supported files.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        all_docs: list[Document] = []
        found_files = 0

        for path in sorted(dir_path.rglob("*")):
            # Skip hidden files and directories (e.g. .git, .env)
            if any(part.startswith(".") for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                found_files += 1
                try:
                    file_docs = cls.load_file(path)
                    all_docs.extend(file_docs)
                except Exception as e:
                    logger.warning(f"Skipping problematic file {path.name}: {e}")

        logger.info(
            f"Loaded total {len(all_docs)} segment(s) from {found_files} file(s) in {dir_path.name}"
        )
        return all_docs
