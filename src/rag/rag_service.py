"""
RAG (Retrieval-Augmented Generation) Service Module.
Handles document retrieval from Qdrant, context formatting,
and grounded answer generation with source document citations.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from src.api.ollama_service import ollama_service
from src.prompts import RAG_PROMPT
from src.rag.vector_store import vector_store_manager

logger = logging.getLogger(__name__)


class RAGService:
    """Coordinates retrieval from Qdrant and grounded response generation via Ollama."""

    def __init__(self):
        self.vector_store = vector_store_manager
        self.llm = ollama_service.llm

        # Grounded system prompt strictly enforcing facts from context and citations
        self.rag_prompt = RAG_PROMPT

    def _format_context(self, documents: list[Document]) -> str:
        """Formats retrieved document chunks into a clean, annotated context block."""
        context_parts = []
        for doc in documents:
            filename = doc.metadata.get("filename", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            context_parts.append(
                f"--- [Document: {filename} | ID: {chunk_id}] ---\n{doc.page_content.strip()}"
            )
        return "\n\n".join(context_parts)

    def _extract_sources(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Extracts unique source document provenance records from retrieved chunks."""
        sources = []
        seen = set()
        for doc in documents:
            filename = doc.metadata.get("filename", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            key = (filename, chunk_id)
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "filename": filename,
                        "chunk_id": chunk_id,
                        "file_type": doc.metadata.get("file_type", ""),
                        "source_path": doc.metadata.get("source", ""),
                    }
                )
        return sources

    async def answer_question(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """
        Retrieves relevant document chunks and generates a grounded response with citations.

        Args:
            question: The user query.
            top_k: Number of most relevant document chunks to retrieve.

        Returns:
            Dictionary containing 'answer', 'sources', and 'retrieved_chunks_count'.
        """
        logger.info(f"RAG query received: '{question}' (retrieving top_{top_k} chunks)")

        # 1. Retrieve top-k most relevant chunks from Qdrant
        retrieved_docs = self.vector_store.similarity_search(query=question, k=top_k)

        if not retrieved_docs:
            logger.warning(f"No relevant document chunks found in Qdrant for query: '{question}'")
            return {
                "answer": "No relevant documents found in the database. Please ingest documents first.",
                "sources": [],
                "retrieved_chunks_count": 0,
            }

        # 2. Format context for prompt
        context_text = self._format_context(retrieved_docs)

        # 3. Format prompt and generate response via local Ollama LLM
        prompt_value = self.rag_prompt.invoke(
            {
                "context": context_text,
                "question": question,
            }
        )

        response = await self.llm.ainvoke(prompt_value)
        answer_text = str(response.content).strip()

        # 4. Extract citations
        sources = self._extract_sources(retrieved_docs)

        logger.info(f"RAG answer generated successfully with {len(sources)} source citation(s).")
        return {
            "answer": answer_text,
            "sources": sources,
            "context": context_text,
            "retrieved_chunks_count": len(retrieved_docs),
        }


# Singleton instance of RAGService
rag_service = RAGService()
