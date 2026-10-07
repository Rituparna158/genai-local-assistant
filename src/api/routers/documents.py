"""
Documents Router.
Handles document ingestion, vector indexing, listing, and deletion.
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from src.api.schemas import (
    DocumentListResponse,
    DocumentItem,
    DocumentIngestResponse,
    DocumentDeleteResponse,
)
from src.rag.loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.vector_store import vector_store_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest Documents into Vector Store",
    description="Scans data/documents directory, chunks all PDF, DOCX, TXT, and Markdown files, and indexes into Qdrant.",
)
async def ingest_documents() -> DocumentIngestResponse:
    """Ingests all files from data/documents into Qdrant vector database."""
    docs_dir = Path("data/documents").resolve()
    if not docs_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Documents directory 'data/documents' not found.",
        )

    try:
        raw_docs = DocumentLoader.load_directory(docs_dir)
        if not raw_docs:
            return DocumentIngestResponse(
                status="warning",
                message="No supported documents (.pdf, .docx, .txt, .md) found in data/documents.",
                files_processed=0,
                chunks_indexed=0,
                collection_name=vector_store_manager.collection_name,
            )

        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_documents(raw_docs)

        vector_store_manager.add_documents(chunks)

        unique_files = len(set(doc.metadata.get("filename") for doc in raw_docs))

        return DocumentIngestResponse(
            status="success",
            message=f"Successfully ingested {unique_files} file(s) into {len(chunks)} chunks.",
            files_processed=unique_files,
            chunks_indexed=len(chunks),
            collection_name=vector_store_manager.collection_name,
        )
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Indexed Documents",
    description="Returns all document files currently indexed in the Qdrant vector database with chunk statistics.",
)
async def list_documents() -> DocumentListResponse:
    """Lists all indexed documents and chunk statistics in Qdrant."""
    try:
        raw_list = vector_store_manager.list_indexed_documents()
        items = [
            DocumentItem(
                filename=item["filename"],
                file_type=item["file_type"],
                source=item["source"],
                chunk_count=item["chunk_count"],
            )
            for item in raw_list
        ]
        total_chunks = sum(item.chunk_count for item in items)

        return DocumentListResponse(
            total_documents=len(items),
            total_chunks=total_chunks,
            documents=items,
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        )


@router.delete(
    "/{id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document by Identifier",
    description="Deletes all vectors and chunks belonging to the specified document filename from Qdrant.",
)
async def delete_document(id: str) -> DocumentDeleteResponse:
    """Removes a document's chunks from Qdrant by filename or ID."""
    try:
        success = vector_store_manager.delete_document(filename=id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete document '{id}'.",
            )

        return DocumentDeleteResponse(
            status="success",
            message=f"All chunks for document '{id}' were removed from Qdrant.",
            document_id=id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document '{id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document '{id}': {str(e)}",
        )
