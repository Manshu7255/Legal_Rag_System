"""
pdf_parser.py — Extracts text and metadata from Supreme Court judgment PDFs.
Uses PyMuPDF (fitz) for reliable text extraction.
"""
import re
import fitz  # PyMuPDF
from pathlib import Path


def extract_case_name(text: str, filename: str) -> str:
    """
    Try to extract case name from the first 500 chars of the judgment text.
    Falls back to the filename if not found.
    """
    # SC judgments usually have "X vs Y" or "X v Y" in the opening lines
    match = re.search(
        r"([A-Z][A-Za-z\s\.]+(?:vs?\.?|versus)\s+[A-Z][A-Za-z\s\.]+)",
        text[:500]
    )
    if match:
        return match.group(0).strip()
    # Fall back to cleaned filename
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


def extract_year(text: str, filename: str) -> str:
    """
    Extract the judgment year from the text or filename.
    """
    # Look for 4-digit year between 1950 and 2025 in first 200 chars
    match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", text[:200])
    if match:
        return match.group(0)
    # Try in filename
    match = re.search(r"\b(19[5-9]\d|20[0-2]\d)\b", filename)
    if match:
        return match.group(0)
    return "Unknown"


def extract_bench(text: str) -> list[str]:
    """
    Try to extract judge names from the text (typically appear near top).
    """
    judges = []
    # Patterns like "JUSTICE X Y" or "HON'BLE MR. JUSTICE X"
    pattern = re.compile(
        r"(?:HON'?BLE|JUSTICE|J\.|CJI)[.\s]+([A-Z][A-Z\s\.]+?)(?:\n|,|AND|&)",
        re.IGNORECASE
    )
    matches = pattern.findall(text[:1000])
    for m in matches:
        name = m.strip()
        if 2 < len(name) < 60:
            judges.append(name)
    return list(set(judges))[:5]  # Cap at 5 judges


def parse_pdf(pdf_path: str) -> dict | None:
    """
    Parse a single PDF and return structured metadata + full text.

    Returns:
        {
            "text": str,               # Full extracted text
            "case_name": str,          # Extracted case title
            "year": str,               # Judgment year
            "bench": list[str],        # Judge names
            "source_file": str,        # Original filename
            "page_count": int          # Number of pages
        }
        Returns None if the PDF has no extractable text.
    """
    path = Path(pdf_path)
    try:
        doc = fitz.open(str(path))
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()

        full_text = "\n".join(pages_text).strip()
        if not full_text or len(full_text) < 100:
            return None  # Skip empty / scanned-only PDFs

        case_name = extract_case_name(full_text, path.name)
        year      = extract_year(full_text, path.name)
        bench     = extract_bench(full_text)

        return {
            "text":        full_text,
            "case_name":   case_name,
            "year":        year,
            "bench":       bench,
            "source_file": path.name,
            "page_count":  len(pages_text),
        }

    except Exception as e:
        print(f"  [WARN] Could not parse {path.name}: {e}")
        return None
