"""
chunker.py — Splits judgment text into overlapping chunks for embedding.

Strategy:
  - Split by character count (not token count) to avoid tiktoken overhead at scale
  - Preserve sentence boundaries where possible
  - Attach document metadata to every chunk
"""
import re


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (no NLTK dependency)."""
    # Split on ., !, ? followed by whitespace + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    Tries to break at sentence boundaries within the allowed window.

    Args:
        text:          Full document text string
        chunk_size:    Target max characters per chunk
        chunk_overlap: Characters to repeat between consecutive chunks

    Returns:
        List of text chunk strings
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)
        if current_len + slen > chunk_size and current_chunk:
            # Finalize current chunk
            chunks.append(" ".join(current_chunk))
            # Keep overlap: walk back until we are within overlap limit
            overlap_text = []
            overlap_len = 0
            for sent in reversed(current_chunk):
                if overlap_len + len(sent) > chunk_overlap:
                    break
                overlap_text.insert(0, sent)
                overlap_len += len(sent)
            current_chunk = overlap_text
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += slen

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def create_chunks(parsed_doc: dict, chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    """
    Given a parsed document dict (from pdf_parser.py), produce a list of
    chunk dicts ready for insertion into ChromaDB.

    Returns:
        List of dicts:
        {
            "text":        str,   # chunk content
            "case_name":   str,
            "year":        str,
            "source_file": str,
            "chunk_index": int,   # position within the document
        }
    """
    raw_chunks = chunk_text(
        parsed_doc["text"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result = []
    for i, chunk in enumerate(raw_chunks):
        result.append({
            "text":        chunk,
            "case_name":   parsed_doc["case_name"],
            "year":        parsed_doc["year"],
            "source_file": parsed_doc["source_file"],
            "chunk_index": i,
        })
    return result
