"""
Stage 2: retrieval.

Wraps the search logic into one clean, reusable tool. You give it any
incident text and it returns the most similar past incidents from ChromaDB.

The embedding model is loaded once when you create the Retriever, then
reused for every query, which is fast and avoids reloading it each time.

Stage 3 (generation) will import Retriever from here.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from rag.config import EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME, TOP_K


class Retriever:
    """Finds the most similar past incidents for a given query text."""

    def __init__(self):
        # Load the embedding model once (reused for every search).
        self.model = SentenceTransformer(EMBED_MODEL)
        # Connect to the existing ChromaDB store on disk.
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query_text, k=TOP_K):
        """
        Return the top-k most similar incidents to query_text.

        Each result is a dict with the report text, its metadata
        (severity, location, etc.) and the distance score.
        """
        # 1. Turn the query into a vector.
        query_vec = self.model.encode([query_text]).tolist()

        # 2. Ask ChromaDB for the nearest stored vectors.
        results = self.collection.query(query_embeddings=query_vec, n_results=k)

        # 3. Repackage Chroma's raw response into a tidy list of dicts.
        #    Chroma nests everything one level deep (per query), hence [0].
        matches = []
        for doc, meta, dist, rid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            matches.append({
                "id": rid,
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })
        return matches


# Quick manual test: run `python -m rag.retrieve` from the project root.
if __name__ == "__main__":
    retriever = Retriever()
    query = "barrier did not come down at a level crossing"
    print(f"Query: {query}\n")
    for i, m in enumerate(retriever.retrieve(query), 1):
        print(f"{i}. [{m['metadata']['incident_type']}] "
              f"severity={m['metadata']['true_severity']} "
              f"distance={m['distance']:.3f}")
        print(f"   {m['text'][:140]}...\n")
