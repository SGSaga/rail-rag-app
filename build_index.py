"""
Run this to build the searchable index from the incident reports.

    python build_index.py

It embeds all reports into ChromaDB, then runs one test search so you can
see retrieval working before we build the rest.
"""

from sentence_transformers import SentenceTransformer

from rag.config import EMBED_MODEL, TOP_K
from rag.ingest import build_index


def test_search(collection):
    """Embed a made-up query and show the nearest stored reports."""
    query = "a train went past a red signal at a busy station"
    print("\n" + "=" * 60)
    print(f"TEST SEARCH for: '{query}'")
    print("=" * 60)

    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode([query]).tolist()

    results = collection.query(query_embeddings=query_vec, n_results=TOP_K)

    # results is a dict of lists; [0] because we sent one query.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"\nMatch {i}  (distance {dist:.3f}, lower = more similar)")
        print(f"  type: {meta['incident_type']}  |  severity: {meta['true_severity']}")
        print(f"  {doc[:200]}...")


if __name__ == "__main__":
    collection = build_index()
    test_search(collection)
    print("\nDone. If the matches above look related to red signals, retrieval works.")
