"""Ingest knowledge base documents into Pinecone.

Chunks documents, embeds with OpenAI, upserts to Pinecone with metadata.

Usage:
    python -m backend.rag.ingest
"""

import hashlib
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from backend.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    EMBEDDING_MODEL,
    KNOWLEDGE_DOCS,
)
from backend.rag.chunker import chunk_all_documents, Chunk


def get_openai_client() -> OpenAI:
    """Initialize OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_chunks(client: OpenAI, chunks: list[Chunk], batch_size: int = 100) -> list[list[float]]:
    """Embed chunk texts using OpenAI embeddings."""
    texts = [c.text for c in chunks]
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def chunk_id(chunk: Chunk) -> str:
    """Generate a stable ID for a chunk based on content hash."""
    content = chunk.text + str(chunk.metadata.get("source_doc", ""))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ensure_index(pc: Pinecone) -> None:
    """Create Pinecone index if it doesn't exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
    else:
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


def upsert_to_pinecone(
    pc: Pinecone,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    batch_size: int = 50,
) -> None:
    """Upsert embedded chunks to Pinecone."""
    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        # Pinecone metadata values must be strings, numbers, booleans, or lists of strings
        metadata = {
            "source_doc": chunk.metadata["source_doc"],
            "section": chunk.metadata["section"],
            "content_type": chunk.metadata["content_type"],
            "signal_category": chunk.metadata["signal_category"],  # list of strings
            "text": chunk.text[:40000],  # Pinecone metadata limit
        }
        vectors.append({
            "id": chunk_id(chunk),
            "values": embedding,
            "metadata": metadata,
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        print(f"  Upserting batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}...")
        index.upsert(vectors=batch)

    print(f"Upserted {len(vectors)} vectors to '{PINECONE_INDEX_NAME}'.")


def is_already_ingested(pc: Pinecone, expected_count: int) -> bool:
    """Check if the index already has vectors from a previous ingestion."""
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        return False

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    current_count = stats.total_vector_count

    if current_count == 0:
        return False

    # If vector count matches expected chunks, skip
    if current_count == expected_count:
        return True

    # Close enough (within 5) — docs may have changed slightly
    return abs(current_count - expected_count) <= 5


def ingest(force: bool = False):
    """Run the full ingestion pipeline.

    Args:
        force: If True, re-ingest even if vectors already exist.
    """
    print("=" * 60)
    print("SPARC Knowledge Base Ingestion")
    print("=" * 60)

    # Step 1: Chunk documents
    print("\n[1/3] Chunking documents...")
    chunks = chunk_all_documents(KNOWLEDGE_DOCS)
    if not chunks:
        print("No chunks produced. Check that doc files exist.")
        return

    # Check if already ingested
    if not force:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if is_already_ingested(pc, len(chunks)):
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            print(f"\nAlready ingested — {stats.total_vector_count} vectors in Pinecone.")
            print("To re-ingest, run:  python -m backend.rag.ingest --force")
            return

    # Step 2: Embed with OpenAI
    print("\n[2/3] Embedding chunks with OpenAI...")
    client = get_openai_client()
    embeddings = embed_chunks(client, chunks)
    print(f"  Produced {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    # Step 3: Upsert to Pinecone
    print("\n[3/3] Upserting to Pinecone...")
    if not force:
        # pc already initialized above
        pass
    else:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc)
    upsert_to_pinecone(pc, chunks, embeddings)

    print("\nIngestion complete!")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ingest(force=force)
