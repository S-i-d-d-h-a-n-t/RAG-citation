"""
search.py – Step 5: Retrieval
=============================
Concept: Nearest-Neighbour Search

Embed the user's question using the exact same model we used for ingestion.
Then compare that question vector against every chunk vector in our VectorStore
using cosine similarity.

Return the top-k most similar chunks, including their original text and 
provenance metadata (file, page).
"""

import sys
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from vector_store import VectorStore

# Load model globally so we don't reload it for every search when running as a server
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def search(question: str, store_dir: str = "db", k: int = 5) -> list[dict]:
    """
    Search the VectorStore for chunks relevant to the question.
    
    Returns a list of dicts:
    [
        {
            "id": "...",
            "text": "...",
            "metadata": {"file": "...", "page": 1, ...},
            "score": 0.85
        },
        ...
    ]
    """
    store = VectorStore(store_dir)
    if store.count() == 0:
        print("VectorStore is empty! Run ingest.py first.", file=sys.stderr)
        return []
        
    model = get_model()
    
    # Embed the question (must be a flat list of floats for VectorStore)
    q_emb = model.encode(question, convert_to_numpy=True).tolist()
    
    # Query the store
    results = store.query(q_emb, k=k)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to return")
    parser.add_argument("--db", type=str, default="db", help="Path to vector store")
    parser.add_argument("--debug", action="store_true", help="Show raw scores and chunks")
    
    args = parser.parse_args()
    
    print(f"Searching for: '{args.question}'...")
    results = search(args.question, store_dir=args.db, k=args.k)
    
    if not results:
        sys.exit(1)
        
    print(f"\nFound top {len(results)} matches:\n")
    
    for i, res in enumerate(results, 1):
        meta = res["metadata"]
        score = res["score"]
        file_name = meta["file"]
        page = meta["page"]
        
        print(f"--- Result {i} (Score: {score:.3f}) ---")
        print(f"Source: {file_name}, Page: {page}")
        
        if args.debug:
            print(f"Text:\n{res['text']}")
        else:
            # Just print a snippet
            snippet = res['text'][:150].replace("\n", " ")
            print(f"Snippet: {snippet}...")
        print()
