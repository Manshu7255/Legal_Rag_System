"""
app.py — Flask API for the Legal RAG system.

Endpoints:
    POST /api/query   — Run the full 6-agent RAG pipeline on a legal query
    GET  /api/health  — Health check + vector DB status
    GET  /api/stats   — Detailed vector DB statistics

Usage:
    python app.py
"""
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

from config import FLASK_PORT, FLASK_DEBUG, CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from utils.schema_builder import build_error_response

# ── Import agents ─────────────────────────────────────────────────────────────
import agents.query_analysis_agent      as query_analysis_agent
import agents.research_planning_agent   as research_planning_agent
import agents.retrieval_agent           as retrieval_agent
import agents.cross_verification_agent  as cross_verification_agent
import agents.hallucination_guard_agent as hallucination_guard_agent
import agents.response_formatter_agent  as response_formatter_agent

app = Flask(__name__)
CORS(app)   # Allow requests from the Next.js frontend


# ─────────────────────────────────────────────────────────────────────────────
#  Core Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(query: str) -> dict:
    """
    Execute the full 6-agent RAG pipeline for a legal query.
    Returns a structuredResponse-conformant dict.
    """
    agent_timings = {}

    # ── Agent 1: Query Analysis ──────────────────────────────────
    qa_result = query_analysis_agent.run(query)
    agent_timings["QueryAnalysisAgent"] = {
        "timeMs": qa_result["timing_ms"],
        "status": qa_result["status"]
    }
    query_context = qa_result["query_context"]

    # ── Agent 2: Research Planning ───────────────────────────────
    rp_result = research_planning_agent.run(query_context)
    agent_timings["ResearchPlanningAgent"] = {
        "timeMs": rp_result["timing_ms"],
        "status": rp_result["status"]
    }
    research_plan = rp_result["research_plan"]

    # ── Agent 3: Retrieval ────────────────────────────────────────
    ret_result = retrieval_agent.run(research_plan)
    agent_timings["RetrievalAgent"] = {
        "timeMs": ret_result["timing_ms"],
        "status": ret_result["status"]
    }
    chunks = ret_result["chunks"]

    if not chunks:
        agent_timings["CrossVerificationAgent"]  = {"timeMs": 0, "status": "SKIPPED"}
        agent_timings["HallucinationGuardAgent"] = {"timeMs": 0, "status": "SKIPPED"}
        return response_formatter_agent.run(
            guarded_facts         = {},
            confidence_score      = 0.0,
            analysis_limitations  = ret_result.get("status", "No results found in vector DB."),
            agent_timings         = agent_timings,
        )["final_response"]

    # ── Agent 4: Cross Verification ──────────────────────────────
    cv_result = cross_verification_agent.run(chunks, query)
    agent_timings["CrossVerificationAgent"] = {
        "timeMs": cv_result["timing_ms"],
        "status": cv_result["status"]
    }
    verified_facts = cv_result["verified_facts"]

    # ── Agent 5: Hallucination Guard ─────────────────────────────
    hg_result = hallucination_guard_agent.run(verified_facts, chunks)
    agent_timings["HallucinationGuardAgent"] = {
        "timeMs": hg_result["timing_ms"],
        "status": hg_result["status"]
    }

    # ── Agent 6: Response Formatter ──────────────────────────────
    fmt_result = response_formatter_agent.run(
        guarded_facts        = hg_result["guarded_facts"],
        confidence_score     = hg_result["confidence_score"],
        analysis_limitations = hg_result["analysis_limitations"],
        agent_timings        = agent_timings,
    )

    return fmt_result["final_response"]


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/query", methods=["POST"])
def query():
    """
    POST /api/query
    Body: { "query": "Your legal question here" }
    Returns: full structuredResponse schema
    """
    data = request.get_json(silent=True)
    if not data or not data.get("query"):
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    user_query = str(data["query"]).strip()
    if len(user_query) < 5:
        return jsonify({"error": "Query too short"}), 400
    if len(user_query) > 2000:
        return jsonify({"error": "Query too long (max 2000 characters)"}), 400

    try:
        result = run_pipeline(user_query)
        return jsonify(result), 200
    except Exception as e:
        error_resp = build_error_response(str(e))
        return jsonify(error_resp), 500


@app.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — Check if the API and vector DB are ready."""
    try:
        import chromadb
        client     = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(CHROMA_COLLECTION)
        doc_count  = collection.count()
        status     = "ready" if doc_count > 0 else "empty_db"
        return jsonify({
            "status":          status,
            "vector_db_docs":  doc_count,
            "collection_name": CHROMA_COLLECTION,
            "message":         "Run python ingest.py first if vector_db_docs is 0" if doc_count == 0 else "OK"
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """GET /api/stats — Detailed statistics about the vector DB."""
    try:
        import chromadb
        from config import GEMINI_LLM_MODEL, LOCAL_EMBED_MODEL

        client     = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection(CHROMA_COLLECTION)
        count      = collection.count()

        # Sample a few records to show metadata summary
        sample = {}
        if count > 0:
            sample_data = collection.peek(limit=5)
            sample = {
                "sample_cases": [m.get("case_name") for m in sample_data["metadatas"]],
                "sample_years": [m.get("year") for m in sample_data["metadatas"]],
            }

        return jsonify({
            "total_chunks":    count,
            "collection_name": CHROMA_COLLECTION,
            "chroma_path":     CHROMA_PERSIST_DIR,
            "llm_model":       GEMINI_LLM_MODEL,
            "embed_model":     LOCAL_EMBED_MODEL,
            **sample
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🏛️  Legal RAG API starting on http://localhost:{FLASK_PORT}")
    print(f"   Endpoints:")
    print(f"     POST http://localhost:{FLASK_PORT}/api/query")
    print(f"     GET  http://localhost:{FLASK_PORT}/api/health")
    print(f"     GET  http://localhost:{FLASK_PORT}/api/stats\n")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=FLASK_DEBUG)
