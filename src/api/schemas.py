"""
Pydantic Schemas for API Request and Response models.
Defines data contracts for /health, /chat, and document management endpoints.
"""

import re

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Schema for the GET /health response."""

    status: str = Field(default="healthy", description="Current health status of the service")
    app_env: str = Field(description="Active environment (development/production)")
    model: str = Field(description="Currently configured local LLM model name")
    dependencies: dict[str, str] | None = Field(
        default=None, description="Connectivity status of backing dependencies"
    )


class SourceCitation(BaseModel):
    """Schema for document provenance and source citations."""

    filename: str = Field(description="Name of the source document file")
    chunk_id: str = Field(description="Unique deterministic chunk identifier")
    file_type: str = Field(description="File format extension (pdf, docx, txt, md)")
    source_path: str = Field(description="Path or filename of the source document")


class ChatRequest(BaseModel):
    """Schema for incoming POST /chat requests."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user query or prompt",
        examples=["What is the company's leave policy?"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional session ID for tracking conversation history",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Ensures the message is not empty or composed solely of whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only.")
        return stripped

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        """Validates session_id characters to prevent injection attacks."""
        if v is None:
            return None
        stripped = v.strip()
        if not stripped:
            return None
        if not re.match(r"^[a-zA-Z0-9_\-]+$", stripped):
            raise ValueError(
                "session_id must contain only alphanumeric characters, dashes, or underscores."
            )
        return stripped


class ChatResponse(BaseModel):
    """Schema for POST /chat responses."""

    response: str = Field(description="Generated reply from the assistant")
    model: str = Field(description="The model that produced the answer")
    session_id: str | None = Field(
        default=None, description="Session ID associated with the interaction"
    )
    route: str | None = Field(
        default=None,
        description="The intelligent routing pathway selected (rag, sql, combined, chitchat, refuse)",
    )
    sources: list[SourceCitation] | None = Field(
        default=None, description="List of cited source documents and chunks (when applicable)"
    )
    sql_query: str | None = Field(
        default=None, description="The executed SQL query (when applicable for database searches)"
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
    documents: list[DocumentItem] = Field(description="List of indexed document summaries")


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
