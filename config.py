"""
config.py — Central configuration for the Legal RAG system.
All settings are loaded from .env, with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini API ─────────────────────────────────────────────────
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
GEMINI_LLM_MODEL     = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")

# ── Local Embeddings (sentence-transformers, no API limits) ────
LOCAL_EMBED_MODEL    = os.getenv("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Dataset ─────────────────────────────────────────────────────
DATASET_PDF_PATH     = os.getenv("DATASET_PDF_PATH", "./dataset")

# ── ChromaDB ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR   = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION    = os.getenv("CHROMA_COLLECTION_NAME", "sc_judgments")

# ── Ingestion ────────────────────────────────────────────────────
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 100))
BATCH_SIZE           = int(os.getenv("BATCH_SIZE", 50))

# ── Retrieval ─────────────────────────────────────────────────────
TOP_K_RESULTS        = int(os.getenv("TOP_K_RESULTS", 8))

# ── Flask ────────────────────────────────────────────────────────
FLASK_PORT           = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG          = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ── Jurisdiction ─────────────────────────────────────────────────
JURISDICTION         = "India"
