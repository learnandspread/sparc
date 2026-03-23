"""Configuration for SPARC Voice Agent Backend."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
VAPI_PUBLIC_KEY = os.getenv("VAPI_PUBLIC_KEY", "")

# --- Models ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Pinecone ---
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sparc-knowledge")
PINECONE_DIMENSION = 1536  # text-embedding-3-small output dimension

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# --- RAG ---
RAG_TOP_K = 8  # number of chunks to retrieve
RAG_BOOST_WEIGHT = 1.5  # multiplier for page-relevant chunks

# --- Paths ---
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

# Knowledge base docs (for vector store ingestion)
KNOWLEDGE_DOCS = [
    os.path.join(DOCS_DIR, "SPARC™ Master Knowledge System.docx"),
    os.path.join(DOCS_DIR, "THE DISCOVERY & VISIBILITY ENCYCLOPEDIA.docx"),
    os.path.join(DOCS_DIR, "SPARC™ RESPONSE TRAINING EXAMPLES.docx"),
]

# Page route to persona mapping
ROUTE_TO_PERSONA = {
    "/services/strategy-intelligence": "analyst",
    "/services/visibility-authority": "navigator",
    "/services/acquisition-conversion": "accelerator",
    "/services/experience-infrastructure": "engineer",
    "/services/retention-growth": "multiplier",
    "/about": "connector",
}

DEFAULT_PERSONA = "connector"
