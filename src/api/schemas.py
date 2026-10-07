"""
Pydantic Schemas for API Request and Response models.
Defines data contracts for /health, /chat, and document management endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for the GET /health response."""
    status: str = Field(default="healthy", description="Current health status of the service")
    app_env: str = Field(description="Active environment (development/production)")
    model: str = Field(description="Currently configured local LLM model name")


class SourceCitation(BaseModel):
    """Schema for document provenance and source citations."""
    filename: str = Field(description="Name of the source document file")
    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    file_type: str = Field(description="File format extension (pdf, docx, txt, md)")
    source_path: str = Field(description="Full path or URL to the source document")


class ChatRequest(BaseModel):
    """Schema for incoming POST /chat requests."""
    message: str = Field(
        ...,
        min_length=1,
        description="The user query or prompt",
        examples=["What is the company's leave policy?"]
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for tracking conversation history"
    )


class ChatResponse(BaseModel):
    """Schema for POST /chat responses."""
    response: str = Field(description="Generated reply from the assistant")
    model: str = Field(description="The model that produced the answer")
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID associated with the interaction"
    )
    sources: Optional[List[SourceCitation]] = Field(
        default=None,
        description="List of cited source documents and chunks (when applicable)"
    )


class DocumentItem(BaseModel):
    """Schema representing an indexed document record."""
    filename: str = Field(description="Name of the document file")
    file_type: str = Field(description="File extension type")
    source: str = Field(description="File path source")
    chunk_count: int = Field(description="Total chunks generated for this file")


class DocumentListResponse(BaseModel):
    """Schema for GET /documents response."""
    total_documents: int = Field(description="Total count of unique indexed documents")
    total_chunks: int = Field(description="Total count of vector chunks in Qdrant")
    documents: List[DocumentItem] = Field(description="List of indexed document summaries")


class DocumentIngestResponse(BaseModel):
    """Schema for POST /documents/ingest response."""
    status: str = Field(default="success", description="Ingestion execution status")
    message: str = Field(description="Human-readable result summary")
    files_processed: int = Field(description="Count of processed document files")
    chunks_indexed: int = Field(description="Total chunks embedded and stored in Qdrant")
    collection_name: str = Field(description="Target Qdrant collection name")


class DocumentDeleteResponse(BaseModel):
    """Schema for DELETE /documents/{id} response."""
    status: str = Field(description="Deletion status (success/not_found/failed)")
    message: str = Field(description="Description of the deletion result")
    document_id: str = Field(description="Filename or identifier of the document removed")
