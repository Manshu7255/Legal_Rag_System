"""
query_analysis_agent.py — Agent 1: Parse and classify the user's legal query.

Uses Gemini to extract:
  - Main legal topics
  - Relevant acts / articles
  - Year range (if mentioned)
  - Query type (case search, provision lookup, precedent search, etc.)
"""
import time
import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_LLM_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """You are a legal query analyst specializing in Indian Supreme Court law.
Analyze the given legal query and extract structured information.

Return ONLY valid JSON with this exact structure:
{
  "intent": "one of: case_search | provision_lookup | precedent_search | general_legal",
  "topics": ["list of main legal topics, max 5"],
  "acts_mentioned": ["any specific Acts mentioned in query"],
  "articles_mentioned": ["any specific Articles of Constitution mentioned"],
  "year_range": {"from": 1950, "to": 2025},
  "case_names": ["any specific case names mentioned"],
  "search_keywords": ["3-8 keywords optimized for document retrieval"],
  "complexity": "simple | moderate | complex"
}

Rules:
- year_range: use 1950-2025 as default if no year specified
- Keep lists empty [] if nothing found
- search_keywords must be highly specific legal terms from query
"""


def run(query: str) -> dict:
    """
    Analyze a user's legal query.

    Returns:
        {
            "query_context": dict,   # parsed query info
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()
    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=f"{SYSTEM_PROMPT}\n\nQuery: {query}",
            config=types.GenerateContentConfig(temperature=0.1)
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        query_context = json.loads(raw)
        query_context["original_query"] = query

        elapsed = int((time.time() - start) * 1000)
        return {
            "query_context": query_context,
            "timing_ms": elapsed,
            "status": "SUCCESS"
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        # Fallback: basic keyword extraction
        words = query.split()
        keywords = [w for w in words if len(w) > 4][:8]
        return {
            "query_context": {
                "original_query":     query,
                "intent":             "general_legal",
                "topics":             [],
                "acts_mentioned":     [],
                "articles_mentioned": [],
                "year_range":         {"from": 1950, "to": 2025},
                "case_names":         [],
                "search_keywords":    keywords,
                "complexity":         "moderate",
            },
            "timing_ms": elapsed,
            "status":    f"FALLBACK: {str(e)[:100]}"
        }
