"""
Chat Router.
Handles conversational interactions with RAG-grounded responses and citations.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from src.config import settings
from src.api.schemas import ChatRequest, ChatResponse, SourceCitation
from src.rag.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Assistant (RAG Grounded)",
    description="Processes user query, performs semantic retrieval from Qdrant, and generates an answer with sources.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Processes user query and generates an AI response via RAG with source citations."""
    try:
        rag_result = await rag_service.answer_question(request.message, top_k=3)

        citations = [
            SourceCitation(
                filename=src["filename"],
                chunk_id=src["chunk_id"],
                file_type=src["file_type"],
                source_path=src["source_path"],
            )
            for src in rag_result["sources"]
        ]

        return ChatResponse(
            response=rag_result["answer"],
            model=settings.llm.LLM_MODEL,
            session_id=request.session_id,
            sources=citations if citations else None,
        )
    except Exception as e:
        logger.error(f"Error generating chat response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate assistant response: {str(e)}",
        )
