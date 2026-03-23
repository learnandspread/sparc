"""RAG retriever — query Pinecone and return formatted context.

Given a user query and page context, embeds the query with OpenAI,
searches Pinecone with optional metadata boosting for page-relevant
signal categories, and returns formatted context for the system prompt.
"""

from openai import OpenAI
from pinecone import Pinecone

from backend.config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_BOOST_WEIGHT,
)


_pc: Pinecone | None = None
_openai: OpenAI | None = None


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=OPENAI_API_KEY)
    return _openai


def retrieve(
    query: str,
    signal_focus: list[str] | None = None,
    top_k: int = RAG_TOP_K,
) -> str:
    """Retrieve relevant knowledge base context for a query.

    Args:
        query: The user's question or message.
        signal_focus: Signal categories to boost (e.g. ["visibility", "authority"]).
        top_k: Number of chunks to retrieve.

    Returns:
        Formatted context string for injection into system prompt.
    """
    if not query.strip():
        return ""

    pc = _get_pinecone()
    client = _get_openai()

    # Embed the query
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    query_embedding = response.data[0].embedding

    # Search Pinecone — retrieve more than top_k so we can re-rank
    fetch_k = top_k * 2
    index = pc.Index(PINECONE_INDEX_NAME)
    search_result = index.query(
        vector=query_embedding,
        top_k=fetch_k,
        include_metadata=True,
    )

    if not search_result.matches:
        return ""

    # Re-rank with signal category boost
    scored_matches = []
    for match in search_result.matches:
        score = match.score
        if signal_focus and match.metadata:
            chunk_signals = match.metadata.get("signal_category", [])
            if isinstance(chunk_signals, str):
                chunk_signals = [chunk_signals]
            # Boost if any of the chunk's signal categories match the page focus
            if any(s in signal_focus for s in chunk_signals):
                score *= RAG_BOOST_WEIGHT
        scored_matches.append((score, match))

    # Sort by boosted score, take top_k
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored_matches[:top_k]

    # Format as context
    context_parts = []
    for i, (score, match) in enumerate(top_matches, 1):
        meta = match.metadata or {}
        text = meta.get("text", "")
        source = meta.get("source_doc", "unknown")
        section = meta.get("section", "")

        context_parts.append(
            f"[Source: {source} | Section: {section}]\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)
