"""
Qdrant Vector Store Management Module.
Handles vector storage, embedding generation, similarity search,
and document lifecycle operations using LangChain and Qdrant.
"""

import atexit
import logging
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector embeddings and Qdrant storage operations."""

    def __init__(self):
        self.collection_name = settings.qdrant.QDRANT_COLLECTION_NAME
        self.embedding_model = settings.llm.EMBEDDING_MODEL
        self.base_url = settings.llm.OLLAMA_BASE_URL

        # Initialize local Ollama embedding generator
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.base_url,
        )

        # Initialize resilient Qdrant client
        self.client = self._initialize_client()
        self._ensure_collection_exists()

        # Wrap with LangChain QdrantVectorStore
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        atexit.register(self.close)

    def close(self) -> None:
        """Gracefully closes the Qdrant client connection upon application shutdown."""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception:
            pass

    def _initialize_client(self) -> QdrantClient:
        """
        Attempts connection to remote/Docker Qdrant server.
        Falls back to local persistent disk storage if server is unavailable.
        """
        server_url = f"http://{settings.qdrant.QDRANT_HOST}:{settings.qdrant.QDRANT_PORT}"
        try:
            client = QdrantClient(url=server_url, timeout=1.5, check_compatibility=False)
            # Ping server to confirm active connection
            client.get_collections()
            logger.info(f"Connected successfully to Qdrant server at {server_url}")
            return client
        except Exception as e:
            fallback_dir = Path("data/qdrant_db").resolve()
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"Could not connect to Qdrant server at {server_url} ({e}). "
                f"Falling back to local embedded vector storage at: {fallback_dir}"
            )
            try:
                return QdrantClient(path=str(fallback_dir))
            except Exception as lock_err:
                logger.warning(
                    f"Local folder {fallback_dir} is locked by a running process ({lock_err}). "
                    "Falling back to in-memory Qdrant instance."
                )
                return QdrantClient(":memory:")

    def _ensure_collection_exists(self) -> None:
        """Creates the target collection in Qdrant if it does not already exist."""
        try:
            collections = [col.name for col in self.client.get_collections().collections]
            if self.collection_name not in collections:
                # nomic-embed-text generates 768-dimensional dense vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=768,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: '{self.collection_name}' (dim=768, metric=Cosine)")
            else:
                logger.info(f"Using existing Qdrant collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection '{self.collection_name}': {e}", exc_info=True)
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Embeds and indexes document chunks into Qdrant.

        Args:
            documents: List of chunked Document objects with metadata.

        Returns:
            List of generated point IDs.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return []

        logger.info(f"Indexing {len(documents)} document chunk(s) into collection '{self.collection_name}'...")
        ids = self.vector_store.add_documents(documents)
        logger.info(f"Successfully indexed {len(ids)} chunk(s).")
        return ids

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Executes semantic similarity search against indexed document chunks.

        Args:
            query: Natural language question or search query.
            k: Number of most relevant chunks to retrieve.

        Returns:
            List of top-k most relevant Document objects.
        """
        logger.info(f"Executing similarity search for query: '{query}' (top_k={k})")
        results = self.vector_store.similarity_search(query=query, k=k)
        logger.info(f"Retrieved {len(results)} relevant chunk(s).")
        return results

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        """
        Executes similarity search and returns documents with cosine similarity scores.

        Args:
            query: Natural language query.
            k: Top-k results limit.

        Returns:
            List of (Document, similarity_score) tuples.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def list_indexed_documents(self) -> List[Dict[str, any]]:
        """
        Scans Qdrant collection points to list all unique indexed files and chunk counts.

        Returns:
            List of dictionary summaries per indexed document file.
        """
        try:
            # Scroll through points to collect metadata
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            file_stats: Dict[str, Dict[str, any]] = {}
            for record in records:
                payload = record.payload or {}
                # LangChain stores metadata inside 'metadata' dictionary
                meta = payload.get("metadata", {})
                filename = meta.get("filename", "unknown")
                file_type = meta.get("file_type", "unknown")
                source = meta.get("source", "")

                if filename not in file_stats:
                    file_stats[filename] = {
                        "filename": filename,
                        "file_type": file_type,
                        "source": source,
                        "chunk_count": 0,
                    }
                file_stats[filename]["chunk_count"] += 1

            return list(file_stats.values())
        except Exception as e:
            logger.error(f"Error listing indexed documents: {e}", exc_info=True)
            return []

    def delete_document(self, filename: str) -> bool:
        """
        Deletes all chunks belonging to a specific filename from Qdrant.

        Args:
            filename: Name of the file whose chunks should be removed.

        Returns:
            True if deletion request succeeded.
        """
        try:
            logger.info(f"Deleting all chunks for file: '{filename}' from collection '{self.collection_name}'")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.filename",
                            match=qmodels.MatchValue(value=filename),
                        )
                    ]
                ),
            )
            logger.info(f"Successfully deleted chunks for '{filename}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for '{filename}': {e}", exc_info=True)
            return False


# Singleton instance of VectorStoreManager
vector_store_manager = VectorStoreManager()
