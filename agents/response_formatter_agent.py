"""
response_formatter_agent.py — Agent 6: Final assembly of structuredResponse.

Validates field types, ensures non-empty required fields, and calls
schema_builder.build_response() to produce the final API output.
"""
import time
from utils.schema_builder import build_response, build_error_response


def run(
    guarded_facts:       dict,
    confidence_score:    float,
    analysis_limitations: str | None,
    agent_timings:       dict,
) -> dict:
    """
    Format the final API response.

    Returns:
        {
            "final_response": dict,  # the full API JSON
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()
    try:
        facts = guarded_facts

        # ── Ensure all list fields are lists ─────────────────────
        def ensure_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val:
                return [val]
            return []

        # ── Ensure strings are strings ───────────────────────────
        def ensure_str(val, fallback=""):
            return str(val).strip() if val else fallback

        issue_summary    = ensure_str(facts.get("issue_summary"), "Legal analysis of the queried topic.")
        legal_interp     = ensure_str(facts.get("legal_interpretation"), "Refer to key observations above.")
        conclusion       = ensure_str(facts.get("conclusion"), "See legal interpretation above.")

        provisions       = ensure_list(facts.get("relevant_legal_provisions"))
        sections         = ensure_list(facts.get("applicable_sections"))
        case_refs        = ensure_list(facts.get("case_references"))
        observations     = ensure_list(facts.get("key_observations"))
        precedents       = ensure_list(facts.get("precedents"))
        citations        = ensure_list(facts.get("citations"))

        # ── Record this agent's timing ────────────────────────────
        elapsed = int((time.time() - start) * 1000)
        agent_timings["ResponseFormatterAgent"] = {
            "timeMs": elapsed,
            "status": "SUCCESS"
        }

        final_response = build_response(
            issue_summary            = issue_summary,
            relevant_legal_provisions= provisions,
            applicable_sections      = sections,
            case_references          = case_refs,
            key_observations         = observations,
            legal_interpretation     = legal_interp,
            precedents               = precedents,
            conclusion               = conclusion,
            citations                = citations,
            confidence_score         = confidence_score,
            analysis_limitations     = analysis_limitations,
            agent_timings            = agent_timings,
        )

        return {
            "final_response": final_response,
            "timing_ms":      elapsed,
            "status":         "SUCCESS"
        }

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        agent_timings["ResponseFormatterAgent"] = {
            "timeMs": elapsed,
            "status": f"ERROR: {str(e)[:100]}"
        }
        return {
            "final_response": build_error_response(str(e), agent_timings),
            "timing_ms":      elapsed,
            "status":         f"ERROR: {str(e)[:100]}"
        }
