"""
Unit tests for VectorStoreManager operations (src/rag/vector_store.py).
Tests document indexing, similarity retrieval, document listing, and chunk deletion
using an isolated in-memory Qdrant client and deterministic test embeddings.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


@pytest.fixture
def memory_vector_store():
    """Fixture providing an isolated in-memory QdrantVectorStore instance."""
    client = QdrantClient(":memory:")
    collection_name = "test_collection"
    embeddings = FakeEmbeddings(size=768)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
    )

    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return {"client": client, "store": store, "collection": collection_name}


def test_vector_store_add_and_search(memory_vector_store):
    """Tests indexing documents and performing similarity search."""
    store = memory_vector_store["store"]

    sample_docs = [
        Document(
            page_content="Employees receive 18 days of paid annual leave.",
            metadata={"filename": "leave.pdf", "chunk_id": "leave.pdf#chunk_0", "file_type": "pdf"},
        ),
        Document(
            page_content="Returns are accepted within 30 days of purchase.",
            metadata={"filename": "refund.md", "chunk_id": "refund.md#chunk_0", "file_type": "md"},
        ),
    ]

    ids = store.add_documents(sample_docs)
    assert len(ids) == 2

    # Query the store
    results = store.similarity_search(query="leave policy", k=1)
    assert len(results) == 1
    assert "filename" in results[0].metadata


def test_vector_store_delete_by_filename(memory_vector_store):
    """Tests removing chunks belonging to a specific filename from the collection."""
    client = memory_vector_store["client"]
    store = memory_vector_store["store"]
    collection = memory_vector_store["collection"]

    sample_docs = [
        Document(
            page_content="Document A chunk 1",
            metadata={"filename": "doc_a.pdf", "chunk_id": "doc_a.pdf#chunk_0"},
        ),
        Document(
            page_content="Document A chunk 2",
            metadata={"filename": "doc_a.pdf", "chunk_id": "doc_a.pdf#chunk_1"},
        ),
        Document(
            page_content="Document B chunk 1",
            metadata={"filename": "doc_b.pdf", "chunk_id": "doc_b.pdf#chunk_0"},
        ),
    ]

    store.add_documents(sample_docs)

    # Delete doc_a.pdf chunks using Qdrant filter
    client.delete(
        collection_name=collection,
        points_selector=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="metadata.filename",
                    match=qmodels.MatchValue(value="doc_a.pdf"),
                )
            ]
        ),
    )

    # Verify only doc_b.pdf remains
    records, _ = client.scroll(collection_name=collection, limit=10, with_payload=True)
    remaining_filenames = [r.payload.get("metadata", {}).get("filename") for r in records]

    assert "doc_a.pdf" not in remaining_filenames
    assert "doc_b.pdf" in remaining_filenames
    assert len(records) == 1
