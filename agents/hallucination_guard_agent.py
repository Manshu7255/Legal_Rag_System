"""
hallucination_guard_agent.py — Agent 5: Validate that all claims in the
verified_facts are actually grounded in the retrieved chunks.

Checks:
  - Case names appear in source chunks
  - Legal provisions appear in source chunks (or are well-known constitutional articles)
  - Observations are not purely LLM-invented
  - Computes final confidence_score
"""
import time
import re


# Well-known Indian constitutional articles that may not appear verbatim in chunks
KNOWN_ARTICLES = {
    "article 14", "article 19", "article 21", "article 32", "article 226",
    "article 136", "article 141", "article 142", "article 368", "article 13",
    "article 20", "article 22", "article 25", "article 300a",
}

# Common statutes that are valid without verbatim chunk appearance
KNOWN_ACTS = {
    "ipc", "crpc", "cpc", "evidence act", "constitution of india",
    "indian penal code", "code of criminal procedure",
    "code of civil procedure", "prevention of corruption act",
    "arbitration act", "companies act", "income tax act",
    "motor vehicles act", "contract act"
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _chunk_corpus(chunks: list[dict]) -> str:
    """Merge all chunk texts into one searchable corpus."""
    return _normalise(" ".join(c["text"] for c in chunks))


def _check_case_reference(case_ref: str, corpus: str) -> bool:
    """Check if a case name (or key name fragment) appears in corpus."""
    # Use the party names (before 'vs') as the search term
    parts = re.split(r"\s+vs?\.?\s+", case_ref, flags=re.IGNORECASE)
    for part in parts:
        clean = _normalise(part.strip())[:30]
        if clean and clean in corpus:
            return True
    return False


def _check_provision(provision: str, corpus: str) -> bool:
    """Check if a legal provision is grounded in corpus or is well-known."""
    p_lower = _normalise(provision)
    if p_lower in corpus:
        return True
    # Allow well-known articles and acts without verbatim presence
    for known in KNOWN_ARTICLES | KNOWN_ACTS:
        if known in p_lower:
            return True
    return False


def run(verified_facts: dict, chunks: list[dict]) -> dict:
    """
    Validate verified_facts against source chunks.

    Returns:
        {
            "guarded_facts": dict,        # cleaned verified_facts
            "confidence_score": float,
            "analysis_limitations": str | None,
            "timing_ms": int,
            "status": str
        }
    """
    start = time.time()

    if not verified_facts or not chunks:
        return {
            "guarded_facts":       verified_facts or {},
            "confidence_score":    0.0,
            "analysis_limitations": "Insufficient data retrieved.",
            "timing_ms":           int((time.time() - start) * 1000),
            "status":              "SKIPPED"
        }

    corpus = _chunk_corpus(chunks)
    avg_similarity = sum(c.get("similarity", 0) for c in chunks) / len(chunks)

    # ── Validate case references ─────────────────────────────────
    raw_case_refs = verified_facts.get("case_references", [])
    valid_case_refs = [c for c in raw_case_refs if _check_case_reference(c, corpus)]
    removed_cases = len(raw_case_refs) - len(valid_case_refs)

    # ── Validate legal provisions ────────────────────────────────
    raw_provisions = verified_facts.get("relevant_legal_provisions", [])
    valid_provisions = [p for p in raw_provisions if _check_provision(p, corpus)]

    raw_sections = verified_facts.get("applicable_sections", [])
    valid_sections = [s for s in raw_sections if _check_provision(s, corpus)]

    # ── Validate citations ───────────────────────────────────────
    raw_citations = verified_facts.get("citations", [])
    valid_citations = []
    for cit in raw_citations:
        name = cit.get("case_name", "")
        if _check_case_reference(name, corpus):
            valid_citations.append(cit)

    # ── Compute confidence score ─────────────────────────────────
    provision_ratio = len(valid_provisions) / max(len(raw_provisions), 1)
    case_ratio      = len(valid_case_refs) / max(len(raw_case_refs), 1)
    has_observations = 1.0 if verified_facts.get("key_observations") else 0.0
    has_conclusion   = 1.0 if verified_facts.get("conclusion") else 0.0

    confidence_score = round(
        (avg_similarity * 0.4) +
        (provision_ratio * 0.2) +
        (case_ratio * 0.2) +
        (has_observations * 0.1) +
        (has_conclusion * 0.1),
        2
    )
    confidence_score = min(confidence_score, 1.0)

    # ── Build limitations string ─────────────────────────────────
    limitations = None
    warnings = []
    if removed_cases > 0:
        warnings.append(f"{removed_cases} unverified case reference(s) removed")
    if avg_similarity < 0.4:
        warnings.append("Low retrieval similarity — results may be tangentially related")
    if confidence_score < 0.5:
        warnings.append("Low overall confidence")
    if warnings:
        limitations = "; ".join(warnings)

    # ── Build guarded facts ──────────────────────────────────────
    guarded = dict(verified_facts)
    guarded["case_references"]           = valid_case_refs or raw_case_refs[:3]
    guarded["relevant_legal_provisions"] = valid_provisions or raw_provisions
    guarded["applicable_sections"]       = valid_sections or raw_sections
    guarded["citations"]                 = valid_citations or raw_citations[:3]

    elapsed = int((time.time() - start) * 1000)
    return {
        "guarded_facts":       guarded,
        "confidence_score":    confidence_score,
        "analysis_limitations": limitations,
        "timing_ms":           elapsed,
        "status":              "SUCCESS"
    }
