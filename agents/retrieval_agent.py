"""
retrieval_agent.py — Agent 3: Semantic search against the ChromaDB vector store.

Uses local sentence-transformers for query embedding, then queries ChromaDB.
Returns ranked document chunks with similarity scores.
"""
import time
from sentence_transformers import SentenceTransformer
import chromadb
from config import (
    LOCAL_EMBED_MODEL,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION
)

# Load local embedding model (same model used for ingestion)
_embed_model = SentenceTransformer(LOCAL_EMBED_MODEL)

# Singleton ChromaDB client (loaded once at import time)
_chroma_client = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def embed_query(text: str) -> list[float]:
    """Generate an embedding for the search query using local model."""
    embedding = _embed_model.encode(text)
    return embedding.tolist()


def run(research_plan: dict) -> dict:
    """
    Execute semantic search against the vector DB.

    Returns:
        {
            "chunks": list[dict],     # retrieved chunks with metadata & scores
            "total_found": int,
            "avg_similarity": float,
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()
    try:
        collection = _get_collection()
        total_docs = collection.count()

        if total_docs == 0:
            return {
                "chunks": [],
                "total_found": 0,
                "avg_similarity": 0.0,
                "timing_ms": int((time.time() - start) * 1000),
                "status": "ERROR: Vector DB is empty. Run ingest.py first."
            }

        search_query = research_plan.get("search_query", "Supreme Court judgment India")
        top_k        = research_plan.get("top_k", 8)

        # Build ChromaDB where filter if needed
        where_filter = None
        if research_plan.get("use_year_filter"):
            y_from = str(research_plan.get("year_from", 1950))
            y_to   = str(research_plan.get("year_to", 2025))
            where_filter = {
                "$and": [
                    {"year": {"$gte": y_from}},
                    {"year": {"$lte": y_to}},
                ]
            }
        if research_plan.get("use_case_filter") and research_plan.get("target_case_name"):
            target = research_plan["target_case_name"]
            if len(target) > 50:
                target = target[:50]
            case_filter = {"case_name": {"$contains": target}}
            where_filter = case_filter  # Prefer case filter over year filter

        # Embed the query (LOCAL - instant, no API call)
        query_embedding = embed_query(search_query)

        # Query ChromaDB
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, total_docs),
            "include":          ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        # Unpack results
        chunks = []
        distances = results["distances"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            # ChromaDB cosine distance: similarity = 1 - distance
            similarity = float(round(1.0 - dist, 4))
            chunks.append({
                "text":        doc,
                "case_name":   meta.get("case_name", "Unknown"),
                "year":        meta.get("year", "Unknown"),
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index", i),
                "similarity":  similarity,
            })

        avg_sim = float(round(sum(c["similarity"] for c in chunks) / len(chunks), 4)) if chunks else 0.0
        elapsed = int((time.time() - start) * 1000)

        return {
            "chunks":         chunks,
            "total_found":    len(chunks),
            "avg_similarity": avg_sim,
            "timing_ms":      elapsed,
            "status":         "SUCCESS"
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return {
            "chunks":         [],
            "total_found":    0,
            "avg_similarity": 0.0,
            "timing_ms":      elapsed,
            "status":         f"ERROR: {str(e)}"
        }
