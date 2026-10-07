"""
Text Chunker Module.
Splits raw documents into semantically coherent, overlapping chunks
using LangChain's RecursiveCharacterTextSplitter.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits Document objects into overlapping chunks with enriched metadata."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """
        Initializes the RecursiveCharacterTextSplitter with chunk parameters.

        Args:
            chunk_size: Maximum number of characters in a single chunk.
            chunk_overlap: Number of characters to overlap between sequential chunks.
            separators: Hierarchy of split delimiters (defaults to paragraphs, newlines, spaces).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Document objects into smaller chunks and enriches
        each chunk's metadata with a deterministic chunk_id and index.

        Args:
            documents: List of input Document objects.

        Returns:
            List of chunked Document objects ready for embedding.
        """
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []

        raw_chunks = self.splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        # Group and index chunks per parent file
        file_chunk_counters: dict[str, int] = {}

        for chunk in raw_chunks:
            filename = chunk.metadata.get("filename", "unknown")
            chunk_index = file_chunk_counters.get(filename, 0)
            file_chunk_counters[filename] = chunk_index + 1

            # Enrich metadata with unique identifier and chunk statistics
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = f"{filename}#chunk_{chunk_index}"
            chunk.metadata["chunk_char_length"] = len(chunk.page_content)

            enriched_chunks.append(chunk)

        logger.info(
            f"Split {len(documents)} document(s) into {len(enriched_chunks)} chunk(s) "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return enriched_chunks
