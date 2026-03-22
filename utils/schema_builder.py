"""
schema_builder.py — Builds the final structuredResponse + agentLogs JSON
that the Flask API returns, matching the exact schema from the existing system.
"""
import uuid
from datetime import datetime, timezone
from config import JURISDICTION


def build_response(
    issue_summary: str,
    relevant_legal_provisions: list[str],
    applicable_sections: list[str],
    case_references: list[str],
    key_observations: list[str],
    legal_interpretation: str,
    precedents: list[str],
    conclusion: str,
    citations: list[dict],
    confidence_score: float,
    analysis_limitations: str | None,
    agent_timings: dict[str, dict],  # {agentName: {timeMs, status}}
) -> dict:
    """
    Assemble the complete API response matching the structuredResponse schema.

    Args:
        agent_timings: dict like:
            {
              "QueryAnalysisAgent": {"timeMs": 687, "status": "SUCCESS"},
              ...
            }
    Returns:
        Full response dict with structuredResponse, agentLogs, totalExecutionTimeMs
    """
    now = datetime.now(timezone.utc).isoformat()

    structured = {
        "response_id":               str(uuid.uuid4()),
        "issue_summary":             issue_summary,
        "relevant_legal_provisions": relevant_legal_provisions,
        "applicable_sections":       applicable_sections,
        "case_references":           case_references,
        "key_observations":          key_observations,
        "legal_interpretation":      legal_interpretation,
        "precedents":                precedents,
        "conclusion":                conclusion,
        "citations":                 citations,
        "confidence_score":          round(confidence_score, 2),
        "analysis_limitations":      analysis_limitations,
        "generated_at":              now,
        "jurisdiction":              JURISDICTION,
    }

    # Standard agent log order (matches existing system)
    agent_order = [
        "QueryAnalysisAgent",
        "ResearchPlanningAgent",
        "RetrievalAgent",
        "CrossVerificationAgent",
        "HallucinationGuardAgent",
        "ResponseFormatterAgent",
    ]

    agent_logs = []
    total_ms = 0
    for name in agent_order:
        info = agent_timings.get(name, {"timeMs": 0, "status": "SKIPPED"})
        agent_logs.append({
            "agentName":      name,
            "executionTimeMs": info["timeMs"],
            "status":         info["status"],
        })
        total_ms += info["timeMs"]

    return {
        "structuredResponse":   structured,
        "agentLogs":            agent_logs,
        "totalExecutionTimeMs": total_ms,
    }


def build_error_response(error_message: str, agent_timings: dict = None) -> dict:
    """Return a schema-conformant error response."""
    return build_response(
        issue_summary=f"Unable to process query: {error_message}",
        relevant_legal_provisions=[],
        applicable_sections=[],
        case_references=[],
        key_observations=[],
        legal_interpretation="",
        precedents=[],
        conclusion="",
        citations=[],
        confidence_score=0.0,
        analysis_limitations=error_message,
        agent_timings=agent_timings or {},
    )
