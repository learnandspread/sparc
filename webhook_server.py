"""SPARC Voice Agent — FastAPI Server.

Vapi calls OpenAI directly (fast). When the assistant needs knowledge,
Vapi's Custom Knowledge Base feature calls our /kb/search endpoint.
We search Pinecone and return relevant documents.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from backend.config import OPENAI_API_KEY, LLM_MODEL, DEFAULT_PERSONA, VAPI_PUBLIC_KEY
from backend.prompts.builder import build_system_prompt, resolve_persona_key
from backend.prompts.personas import get_persona
from backend.rag.retriever import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sparc")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SPARC Voice Agent backend started")
    yield
    logger.info("SPARC Voice Agent backend shutting down")


app = FastAPI(
    title="SPARC Voice Agent",
    description="SPARC™ voice agent — Vapi.ai + OpenAI + Pinecone RAG",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ─── Test UI & Config ───────────────────────────────────────

@app.get("/")
async def test_ui():
    html_path = STATIC_DIR / "test.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/api/config")
async def get_config():
    vapi_assistant_id = os.getenv("VAPI_ASSISTANT_ID", "")
    return {
        "vapiPublicKey": VAPI_PUBLIC_KEY,
        "assistantId": vapi_assistant_id,
    }


@app.get("/api/persona/{persona_key}")
async def get_persona_prompt(persona_key: str):
    resolved = resolve_persona_key(persona_key) if "/" in persona_key else persona_key
    page = get_persona(resolved)
    system_prompt = build_system_prompt(persona_key=resolved)
    return {
        "persona": resolved,
        "firstMessage": page.opening_line,
        "systemPrompt": system_prompt,
        "promptLength": len(system_prompt),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "SPARC™"}


# ─── Knowledge Base Search (Vapi Custom KB) ─────────────────

@app.post("/kb/search")
async def kb_search(request: Request):
    """Vapi Custom Knowledge Base endpoint.

    Vapi sends the conversation history. We extract the latest user query,
    search Pinecone, and return relevant documents.

    Request format:
    {
        "message": {
            "type": "knowledge-base-request",
            "messages": [{"role": "user", "content": "..."}, ...]
        }
    }

    Response format:
    {
        "documents": [
            {"content": "...", "similarity": 0.9, "uuid": "..."}
        ]
    }
    """
    body = await request.json()

    # Log the full payload to understand Vapi's format
    logger.info(f"KB search raw payload: {json.dumps(body, default=str)[:2000]}")

    # Try multiple possible formats
    message = body.get("message", {})
    messages = message.get("messages", [])

    # Also check top-level messages
    if not messages:
        messages = body.get("messages", [])

    # Extract the latest user message for search
    # Vapi uses "message" field, not "content"
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content") or msg.get("message") or ""
            if user_query:
                break

    if not user_query:
        logger.info("KB search: no user message found in payload")
        return JSONResponse(content={"documents": []})

    # Search Pinecone
    start = time.time()
    context = retrieve(query=user_query, top_k=6)
    elapsed = time.time() - start

    logger.info(f"KB search: '{user_query[:80]}' → {len(context)} chars in {elapsed:.2f}s")

    if not context:
        return JSONResponse(content={"documents": []})

    # Split context into separate documents (they're separated by ---)
    chunks = [c.strip() for c in context.split("---") if c.strip()]

    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "content": chunk,
            "similarity": round(0.95 - (i * 0.05), 2),  # Descending relevance
            "uuid": f"sparc-doc-{i}",
        })

    return JSONResponse(content={"documents": documents})


# ─── Vapi Webhook (logging) ─────────────────────────────────

@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    body = await request.json()
    message = body.get("message", {})
    message_type = message.get("type", "")

    logger.info(f"Vapi webhook: type={message_type}")

    if message_type == "end-of-call-report":
        call = message.get("call", {})
        logger.info(
            f"Call ended — duration: {call.get('duration', 0)}s, "
            f"summary: {message.get('summary', '')[:200]}"
        )

    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    from backend.config import HOST, PORT
    uvicorn.run(app, host=HOST, port=int(PORT))
