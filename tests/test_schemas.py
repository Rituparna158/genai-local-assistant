"""
Unit tests for API Pydantic schemas (src/api/schemas.py).
Validates schema validation, constraints, serialization, and edge cases.
"""

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    SourceCitation,
    DocumentItem,
    DocumentListResponse,
    DocumentIngestResponse,
    DocumentDeleteResponse,
)


def test_chat_request_valid():
    """Validates that a well-formed ChatRequest passes validation."""
    req = ChatRequest(message="What is the refund policy?", session_id="session_123")
    assert req.message == "What is the refund policy?"
    assert req.session_id == "session_123"


def test_chat_request_empty_message_fails():
    """Validates that an empty string message fails validation (min_length=1)."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_optional_session_id():
    """Validates that session_id defaults to None when omitted."""
    req = ChatRequest(message="Valid question")
    assert req.session_id is None


def test_chat_response_serialization():
    """Validates that ChatResponse properly nests SourceCitation objects."""
    citation = SourceCitation(
        filename="leave_policy.pdf",
        chunk_id="leave_policy.pdf#chunk_0",
        file_type="pdf",
        source_path="/data/docs/leave_policy.pdf",
    )
    res = ChatResponse(
        response="You have 18 annual leave days.",
        model="llama3.2:3b",
        session_id="sess_1",
        sources=[citation],
    )

    data = res.model_dump()
    assert data["response"] == "You have 18 annual leave days."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["chunk_id"] == "leave_policy.pdf#chunk_0"


def test_document_list_response():
    """Validates DocumentListResponse aggregation."""
    items = [
        DocumentItem(filename="doc1.pdf", file_type="pdf", source="/path/1", chunk_count=3),
        DocumentItem(filename="doc2.txt", file_type="txt", source="/path/2", chunk_count=2),
    ]
    resp = DocumentListResponse(total_documents=2, total_chunks=5, documents=items)
    assert resp.total_documents == 2
    assert resp.total_chunks == 5
    assert len(resp.documents) == 2


def test_document_ingest_response():
    """Validates DocumentIngestResponse default and custom values."""
    resp = DocumentIngestResponse(
        status="success",
        message="Ingested 5 files",
        files_processed=5,
        chunks_indexed=20,
        collection_name="company_documents",
    )
    assert resp.status == "success"
    assert resp.files_processed == 5
    assert resp.chunks_indexed == 20
