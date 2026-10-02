"""
Stage 1: ingestion + embeddings.

This module reads the synthetic incident reports, turns each one into an
embedding (a vector that captures its meaning), and stores them in ChromaDB
so they can be searched later.

You do not usually call this file directly. Run `python build_index.py`
from the project root, which calls build_index() below.
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import (
    DATA_PATH, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME,
)


def load_reports(path=DATA_PATH):
    """Read the JSON file and return the list of report dicts."""
    with open(path) as f:
        return json.load(f)


def report_to_text(report):
    """
    Decide what text represents a report for searching.

    We combine the type, location and narrative. This is the text that
    gets embedded, so it should carry the meaning we want to match on.
    """
    return (
        f"Incident type: {report['incident_type']}. "
        f"Location: {report['location']}. "
        f"{report['narrative']}"
    )


def build_index():
    """
    Embed every report and store it in ChromaDB.

    Safe to run more than once: it recreates the collection each time so
    you never get duplicates.
    """
    # 1. Load the raw reports.
    reports = load_reports()
    print(f"Loaded {len(reports)} reports")

    # 2. Load the local embedding model (downloads ~90MB the first time only).
    print(f"Loading embedding model '{EMBED_MODEL}' (first run downloads it)...")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. Turn each report into the text we will embed.
    texts = [report_to_text(r) for r in reports]

    # 4. Embed all texts at once. Returns one 384-number vector per report.
    print("Embedding reports...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # 5. Open ChromaDB, saving to a folder on disk.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate the collection from scratch so re-running is clean.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.create_collection(COLLECTION_NAME)

    # 6. Store everything: the vector, the text, and metadata we may want back.
    collection.add(
        ids=[r["id"] for r in reports],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "location": r["location"],
                "incident_type": r["incident_type"],
                "true_severity": r["true_severity"],
                "true_root_cause": r["true_root_cause"],
                "date": r["date"],
            }
            for r in reports
        ],
    )

    print(f"Stored {collection.count()} reports in ChromaDB at '{CHROMA_DIR}/'")
    return collection


if __name__ == "__main__":
    build_index()
