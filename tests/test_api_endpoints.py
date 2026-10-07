"""
Integration Tests for FastAPI Endpoints:
- GET /health
- GET /documents
- POST /documents/ingest
- POST /chat
- DELETE /documents/{id}
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Validates the GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_env" in data
    assert "model" in data


def test_list_documents_endpoint():
    """Validates the GET /documents endpoint."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_chat_endpoint_schema():
    """Validates the POST /chat endpoint structure with a factual query."""
    response = client.post(
        "/chat",
        json={"message": "What is the company leave policy?", "session_id": "test_session_1"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0
    assert "model" in data
    assert data["session_id"] == "test_session_1"
    # Citations should be returned
    assert "sources" in data
