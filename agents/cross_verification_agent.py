"""
cross_verification_agent.py — Agent 4: Cross-check and reconcile facts
extracted from multiple retrieved chunks.

Uses Gemini to:
  - Remove contradictory statements
  - Identify the most relevant case references
  - Extract legal provisions from retrieved text
  - Build a verified fact sheet
"""
import time
import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_LLM_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a Senior Indian Supreme Court legal analyst.
You will receive multiple text chunks retrieved from SC judgment documents.
Cross-verify the information and extract structured legal facts.

Return ONLY valid JSON with this exact structure:
{
  "issue_summary": "1-2 sentence summary of the legal issue/topic",
  "key_observations": ["list of 3-6 key legal observations from the texts"],
  "legal_interpretation": "2-3 sentence synthesis of the legal reasoning",
  "relevant_legal_provisions": ["list of Acts, Articles, Sections e.g. 'Article 19(1)(a) of Constitution'"],
  "applicable_sections": ["specific section numbers e.g. 'Section 144 CrPC'"],
  "case_references": ["case names found in chunks e.g. 'Kesavananda Bharati vs State of Kerala (1973)'"],
  "precedents": ["earlier cases cited as precedent"],
  "conclusion": "1-2 sentence conclusion on what the law says",
  "citations": [
    {
      "case_name": "full case name",
      "year": "year as string",
      "court": "Supreme Court of India",
      "relevance": "one line on why this case is relevant"
    }
  ]
}

Rules:
- Only include facts that appear in the provided text chunks
- Do not invent case names, section numbers, or provisions
- Keep key_observations concise (1 sentence each)
- citations should be the top 3-5 most relevant cases found
"""


def _build_context(chunks: list[dict]) -> str:
    """Concatenate retrieved chunks into a numbered context string."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | Case: {chunk['case_name']} | Year: {chunk['year']}]\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(parts)


def run(chunks: list[dict], original_query: str) -> dict:
    """
    Cross-verify facts across retrieved chunks.

    Returns:
        {
            "verified_facts": dict,   # structured legal facts
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()

    if not chunks:
        return {
            "verified_facts": {},
            "timing_ms": int((time.time() - start) * 1000),
            "status": "ERROR: No chunks to verify"
        }

    try:
        context = _build_context(chunks[:10])  # Use top 10 chunks max

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User's Query: {original_query}\n\n"
            f"Retrieved Legal Text:\n{context}"
        )
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.15, max_output_tokens=2048)
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        verified_facts = json.loads(raw)
        elapsed = int((time.time() - start) * 1000)
        return {
            "verified_facts": verified_facts,
            "timing_ms":      elapsed,
            "status":         "SUCCESS"
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        # Fallback: build a minimal fact dict from chunk metadata
        case_refs = list({c["case_name"] for c in chunks if c["case_name"] != "Unknown"})
        return {
            "verified_facts": {
                "issue_summary":             f"Query: {original_query}",
                "key_observations":          [],
                "legal_interpretation":      "Unable to fully process. See raw retrieved text.",
                "relevant_legal_provisions": [],
                "applicable_sections":       [],
                "case_references":           case_refs,
                "precedents":               [],
                "conclusion":               "",
                "citations":               [{"case_name": c, "year": "", "court": "Supreme Court of India", "relevance": ""} for c in case_refs[:4]],
            },
            "timing_ms": elapsed,
            "status":    f"FALLBACK: {str(e)[:100]}"
        }
