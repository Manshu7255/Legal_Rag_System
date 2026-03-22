"""
ingest.py — One-time ingestion pipeline: PDF -> chunks -> embeddings -> ChromaDB

Usage:
    python ingest.py                      # Ingest all PDFs from DATASET_PDF_PATH
    python ingest.py --sample 10          # Test with only 10 PDFs
    python ingest.py --reset              # Clear DB and re-ingest from scratch

The script is resumable: already-ingested files are tracked and skipped.
"""
import os
import sys
import uuid
import time
import argparse
from typing import Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer

import chromadb

from config import (
    LOCAL_EMBED_MODEL,
    DATASET_PDF_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP, BATCH_SIZE
)
from utils.pdf_parser import parse_pdf
from utils.chunker import create_chunks

# ── Setup ────────────────────────────────────────────────────────────────────
print("  Loading embedding model (first run downloads ~80MB)...")
embed_model = SentenceTransformer(LOCAL_EMBED_MODEL)
print(f"  Embedding model '{LOCAL_EMBED_MODEL}' loaded.\n")


def get_collection(reset: bool = False):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    if reset:
        try:
            chroma_client.delete_collection(CHROMA_COLLECTION)
            print(f"  [RESET] Deleted existing collection '{CHROMA_COLLECTION}'")
        except Exception:
            pass
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using local sentence-transformers. No API limits!"""
    embeddings = embed_model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]



def get_ingested_files(collection) -> set[str]:
    """Return a set of source_file names already in the vector DB."""
    try:
        existing = collection.get(include=["metadatas"])
        return {m.get("source_file", "") for m in existing["metadatas"]}
    except Exception:
        return set()


def ingest(sample: Optional[int] = None, reset: bool = False):
    pdf_dir = Path(DATASET_PDF_PATH)
    if not pdf_dir.exists():
        print(f"\n❌ Dataset path not found: {pdf_dir}")
        print("   Please update DATASET_PDF_PATH in your .env file.")
        sys.exit(1)

    all_pdfs = list(pdf_dir.rglob("*.pdf"))
    if not all_pdfs:
        print(f"\n❌ No PDF files found in: {pdf_dir}")
        sys.exit(1)

    if sample is not None:
        all_pdfs = list(all_pdfs[:sample])
        print(f"\n🔬 Sample mode: processing {sample} PDFs")

    collection       = get_collection(reset=reset)
    ingested_files   = get_ingested_files(collection)
    new_pdfs         = [p for p in all_pdfs if p.name not in ingested_files]

    print(f"\n📚 Dataset: {pdf_dir}")
    print(f"   Total PDFs found:     {len(all_pdfs)}")
    print(f"   Already ingested:     {len(ingested_files)}")
    print(f"   To ingest this run:   {len(new_pdfs)}")
    print(f"   Vector DB location:   {CHROMA_PERSIST_DIR}")
    print(f"   Collection:           {CHROMA_COLLECTION}\n")

    if not new_pdfs:
        print("✅ Nothing new to ingest. Vector DB is up to date.")
        return

    total_chunks_ref = [0]   # use list to allow mutation inside nested function
    failed            = 0

    chunk_buffer_ids   = []
    chunk_buffer_docs  = []
    chunk_buffer_metas = []
    chunk_buffer_embs  = []

    def flush_buffer():
        if chunk_buffer_ids:
            collection.add(
                ids=chunk_buffer_ids,
                documents=chunk_buffer_docs,
                metadatas=chunk_buffer_metas,
                embeddings=chunk_buffer_embs,
            )
            total_chunks_ref[0] += len(chunk_buffer_ids)
            chunk_buffer_ids.clear()
            chunk_buffer_docs.clear()
            chunk_buffer_metas.clear()
            chunk_buffer_embs.clear()

    start_time = time.time()

    for i, pdf_path in enumerate(new_pdfs, 1):
        elapsed = time.time() - start_time
        eta_per = elapsed / i if i > 1 else 0
        eta_rem = int(eta_per * (len(new_pdfs) - i))
        display_name = pdf_path.name
        if len(display_name) > 50:
            display_name = display_name[:50]
        print(f"  [{i:>5}/{len(new_pdfs)}] {display_name:<50}  ETA: {eta_rem}s", end="\r")

        # 1. Parse PDF
        parsed = parse_pdf(str(pdf_path))
        if not parsed:
            failed += 1
            continue

        # 2. Chunk text
        chunks = create_chunks(parsed, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            failed += 1
            continue

        # 3. Embed + buffer each chunk
        texts_to_embed = [c["text"] for c in chunks]
        try:
            embeddings = embed_batch(texts_to_embed)
        except Exception as e:
            print(f"\n  [WARN] Embedding failed for {pdf_path.name}: {e}")
            failed += 1
            continue

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            chunk_buffer_ids.append(chunk_id)
            chunk_buffer_docs.append(chunk["text"])
            chunk_buffer_metas.append({
                "case_name":   chunk["case_name"],
                "year":        chunk["year"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
            })
            chunk_buffer_embs.append(emb)

        # Flush when batch is full
        if len(chunk_buffer_ids) >= BATCH_SIZE:
            flush_buffer()

    # Final flush
    flush_buffer()

    total_time = int(time.time() - start_time)
    total_chunks = total_chunks_ref[0]
    print(f"\n\n Ingestion complete!")
    print(f"   PDFs processed: {len(new_pdfs) - failed} / {len(new_pdfs)}")
    print(f"   Chunks stored:  {total_chunks}")
    print(f"   Failed PDFs:    {failed}")
    print(f"   Time taken:     {total_time}s ({total_time // 60}m {total_time % 60}s)")
    print(f"   Total DB size:  {collection.count()} chunks")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SC judgment PDFs into ChromaDB")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only ingest N PDFs (for testing)")
    parser.add_argument("--reset",  action="store_true",
                        help="Delete existing vector DB and re-ingest from scratch")
    args = parser.parse_args()

    ingest(sample=args.sample, reset=args.reset)
