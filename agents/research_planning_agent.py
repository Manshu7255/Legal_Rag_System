"""
research_planning_agent.py — Agent 2: Plan the retrieval strategy.

Decides HOW to search the vector DB:
  - Number of results to fetch (top_k)
  - Metadata filters (year range, specific case)
  - Search query string to use for embedding
  - Whether to do a broad or narrow search
"""
import time
import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_LLM_MODEL, TOP_K_RESULTS

client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are a legal research strategist for an Indian Supreme Court RAG system.
Given a structured query context, define the retrieval plan.

Return ONLY valid JSON with this exact structure:
{
  "search_query": "optimized semantic search string for vector DB (max 200 chars)",
  "top_k": 8,
  "use_year_filter": false,
  "year_from": 1950,
  "year_to": 2025,
  "use_case_filter": false,
  "target_case_name": "",
  "retrieval_strategy": "one of: broad_semantic | case_specific | provision_focused | precedent_chain",
  "reasoning": "one sentence explaining the strategy choice"
}

Rules:
- top_k: use 5 for simple, 8 for moderate, 12 for complex queries
- search_query: combine key legal terms, case names, and relevant provisions into a single semantic search phrase
- use_year_filter: true only if the query explicitly mentions a time period
- use_case_filter: true only if a specific case name is given
"""


def run(query_context: dict) -> dict:
    """
    Plan the retrieval strategy for a given query context.

    Returns:
        {
            "research_plan": dict,
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()
    try:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Query context:\n{json.dumps(query_context, indent=2)}"
        )
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        plan = json.loads(raw)
        elapsed = int((time.time() - start) * 1000)
        return {
            "research_plan": plan,
            "timing_ms":     elapsed,
            "status":        "SUCCESS"
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        # Fallback: build a basic plan from query context keywords
        keywords = query_context.get("search_keywords", [])
        case_names = query_context.get("case_names", [])
        return {
            "research_plan": {
                "search_query":      " ".join(keywords) or query_context.get("original_query", ""),
                "top_k":             TOP_K_RESULTS,
                "use_year_filter":   False,
                "year_from":         1950,
                "year_to":           2025,
                "use_case_filter":   bool(case_names),
                "target_case_name":  case_names[0] if case_names else "",
                "retrieval_strategy": "broad_semantic",
                "reasoning":         "Fallback: using keyword-based search",
            },
            "timing_ms": elapsed,
            "status":    f"FALLBACK: {str(e)[:100]}"
        }
