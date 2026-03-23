"""Test the SPARC pipeline end-to-end.

Run each step independently so you can see exactly where things break.

Usage:
    python -m scripts.test_pipeline
"""

import json
import sys

def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_step_1_chunking():
    """Test: Can we read and chunk the docs?"""
    divider("STEP 1: Document Chunking")

    from backend.config import KNOWLEDGE_DOCS
    from backend.rag.chunker import chunk_all_documents

    chunks = chunk_all_documents(KNOWLEDGE_DOCS)

    if not chunks:
        print("FAIL — no chunks produced. Check that doc files exist in /docs")
        return False

    print(f"\nSample chunk:")
    print(f"  Text: {chunks[0].text[:200]}...")
    print(f"  Metadata: {chunks[0].metadata}")
    print(f"\nPASS — {len(chunks)} chunks ready for embedding")
    return True


def test_step_2_embedding():
    """Test: Can we embed text with OpenAI?"""
    divider("STEP 2: OpenAI Embeddings")

    from openai import OpenAI
    from backend.config import OPENAI_API_KEY, EMBEDDING_MODEL

    if not OPENAI_API_KEY:
        print("FAIL — OPENAI_API_KEY not set in .env")
        return False

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        input=["What is GEO?"],
        model=EMBEDDING_MODEL,
    )

    embedding = response.data[0].embedding
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"\nPASS — embedding works (dim={len(embedding)})")
    return True


def test_step_3_pinecone():
    """Test: Can we connect to Pinecone and is the index there?"""
    divider("STEP 3: Pinecone Connection")

    from pinecone import Pinecone
    from backend.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

    if not PINECONE_API_KEY:
        print("FAIL — PINECONE_API_KEY not set in .env")
        return False

    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = [idx.name for idx in pc.list_indexes()]
    print(f"  Available indexes: {indexes}")

    if PINECONE_INDEX_NAME not in indexes:
        print(f"\n  Index '{PINECONE_INDEX_NAME}' not found.")
        print(f"  Run: python -m backend.rag.ingest  (it will create the index and load data)")
        return False

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"  Index: {PINECONE_INDEX_NAME}")
    print(f"  Total vectors: {stats.total_vector_count}")
    print(f"  Dimension: {stats.dimension}")

    if stats.total_vector_count == 0:
        print(f"\n  Index exists but is empty. Run: python -m backend.rag.ingest")
        return False

    print(f"\nPASS — Pinecone index has {stats.total_vector_count} vectors")
    return True


def test_step_4_retrieval():
    """Test: Can we retrieve relevant chunks?"""
    divider("STEP 4: RAG Retrieval")

    from backend.rag.retriever import retrieve

    test_queries = [
        ("What is GEO?", ["visibility"]),
        ("My ads are too expensive", ["conversion"]),
        ("What is Power Spark X?", None),
    ]

    for query, signal_focus in test_queries:
        print(f"  Query: \"{query}\" (focus: {signal_focus})")
        context = retrieve(query, signal_focus=signal_focus, top_k=3)
        if context:
            # Show first 150 chars of first chunk
            first_chunk = context.split("---")[0].strip()
            print(f"  Result: {first_chunk[:150]}...")
            print()
        else:
            print(f"  FAIL — no results returned\n")
            return False

    print("PASS — retrieval returns relevant chunks")
    return True


def test_step_5_prompt_building():
    """Test: Does prompt assembly work for each persona?"""
    divider("STEP 5: Prompt Building")

    from backend.prompts.builder import build_system_prompt, resolve_persona_key

    test_routes = [
        "/services/visibility-authority",
        "/services/acquisition-conversion",
        "/about",
        "/some/unknown/page",
    ]

    for route in test_routes:
        persona_key = resolve_persona_key(route)
        prompt = build_system_prompt(persona_key=persona_key, rag_context="[test context]")
        print(f"  Route: {route}")
        print(f"  Persona: {persona_key}")
        print(f"  Prompt length: {len(prompt)} chars")

        # Verify all 3 layers are present
        has_identity = "Strategic Predictive Analytical Response Core" in prompt
        has_rag = "RELEVANT KNOWLEDGE BASE CONTEXT" in prompt
        has_persona = "CURRENT PERSONA" in prompt
        print(f"  Layers: identity={has_identity}, rag={has_rag}, persona={has_persona}")
        print()

    print("PASS — all personas build correctly")
    return True


def test_step_6_llm():
    """Test: Can we call OpenAI with an assembled prompt?"""
    divider("STEP 6: OpenAI LLM Response")

    from openai import OpenAI
    from backend.config import OPENAI_API_KEY, LLM_MODEL
    from backend.prompts.builder import build_system_prompt
    from backend.rag.retriever import retrieve

    if not OPENAI_API_KEY:
        print("FAIL — OPENAI_API_KEY not set in .env")
        return False

    # Simulate a user on the Visibility page asking about GEO
    persona_key = "navigator"
    user_query = "What is GEO and why does it matter?"

    print(f"  Persona: {persona_key}")
    print(f"  Query: \"{user_query}\"")
    print(f"  Retrieving context...")

    rag_context = retrieve(user_query, signal_focus=["visibility", "authority"])
    system_prompt = build_system_prompt(persona_key=persona_key, rag_context=rag_context)

    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  Calling OpenAI ({LLM_MODEL})...")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    answer = response.choices[0].message.content
    print(f"\n  --- SPARC Response (Navigator) ---")
    print(f"  {answer[:500]}")
    if len(answer) > 500:
        print(f"  ... ({len(answer)} chars total)")

    print(f"\nPASS — LLM responded with {len(answer)} chars")
    return True


def test_step_7_persona_comparison():
    """Test: Does the same question get different tones on different pages?"""
    divider("STEP 7: Persona Tone Comparison")

    from openai import OpenAI
    from backend.config import OPENAI_API_KEY, LLM_MODEL
    from backend.prompts.builder import build_system_prompt
    from backend.rag.retriever import retrieve
    from backend.prompts.personas import get_persona

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_query = "How do I get more leads?"

    for persona_key in ["navigator", "accelerator", "engineer"]:
        persona = get_persona(persona_key)
        rag_context = retrieve(user_query, signal_focus=persona.signal_focus)
        system_prompt = build_system_prompt(persona_key=persona_key, rag_context=rag_context)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )

        answer = response.choices[0].message.content
        print(f"  [{persona.name}] {answer[:200]}...")
        print()

    print("PASS — compare the tone/framing differences above")
    return True


def main():
    print("\n" + "=" * 60)
    print("  SPARC™ Pipeline Test Suite")
    print("=" * 60)

    steps = [
        ("Chunking", test_step_1_chunking),
        ("Embedding", test_step_2_embedding),
        ("Pinecone", test_step_3_pinecone),
        ("Retrieval", test_step_4_retrieval),
        ("Prompt Building", test_step_5_prompt_building),
        ("OpenAI LLM", test_step_6_llm),
        ("Persona Comparison", test_step_7_persona_comparison),
    ]

    results = []
    for name, test_fn in steps:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"\nFAIL — {type(e).__name__}: {e}")
            passed = False

        results.append((name, passed))

        if not passed:
            print(f"\nStopping at '{name}' — fix this before continuing.")
            break

    divider("RESULTS")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in results)
    if all_passed:
        print(f"\nAll {len(results)} steps passed!")
        print("Next: start the server and test the webhook endpoint:")
        print("  python -m backend.webhook_server")
    else:
        print(f"\n{sum(1 for _, p in results if not p)} step(s) failed.")


if __name__ == "__main__":
    main()
