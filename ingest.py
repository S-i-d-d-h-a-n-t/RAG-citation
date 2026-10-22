"""
ingest.py – Step 4: Embedding & Storage
=========================================
Concept: The Ingestion Pipeline

1. Load text from PDFs (with provenance: file, page)
2. Chunk text into smaller overlapping windows
3. Embed each chunk into a vector (using all-MiniLM-L6-v2)
4. Store the vectors and metadata in our VectorStore

We generate a unique, deterministic ID for each chunk. If you run this twice,
it won't store duplicates.
"""

import hashlib
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from pdf_loader import load_folder
from chunker import chunk_records
from vector_store import VectorStore

def generate_id(file_name: str, page: int, chunk_index: int) -> str:
    """Generate a stable, unique ID for a chunk to prevent duplication."""
    unique_string = f"{file_name}-{page}-{chunk_index}"
    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

def ingest(docs_dir: str | Path, store_dir: str | Path) -> None:
    """Run the ingestion pipeline on a folder of PDFs."""
    print("--- Step 1: Loading PDFs ---")
    pages = load_folder(docs_dir)
    if not pages:
        print("No pages to process.")
        return
        
    print("\n--- Step 2: Chunking Text ---")
    chunks = chunk_records(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    
    print("\n--- Step 3: Generating Embeddings ---")
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # We extract just the text for embedding
    texts_to_embed = [chunk["text"] for chunk in chunks]
    
    # embed in one batch (or smaller batches if memory is an issue, but for ~100 chunks, one batch is fine)
    print(f"Embedding {len(texts_to_embed)} chunks... (this might take a moment)")
    embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
    
    print("\n--- Step 4: Storing in VectorStore ---")
    store = VectorStore(str(store_dir))
    
    # Prepare data for VectorStore
    ids = []
    texts = []
    metadatas = []
    embeddings_list = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = generate_id(chunk["file"], chunk["page"], chunk["chunk_index"])
        
        ids.append(chunk_id)
        texts.append(chunk["text"])
        metadatas.append({
            "file": chunk["file"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"]
        })
        # VectorStore expects a list of floats for each embedding
        embeddings_list.append(embeddings[i].tolist())
        
    new_count = store.add(ids=ids, texts=texts, embeddings=embeddings_list, metadatas=metadatas)
    
    print(f"Added {new_count} new chunks to the store (skipped {len(chunks) - new_count} duplicates).")
    print(f"Total chunks in store: {store.count()}")

if __name__ == "__main__":
    docs_path = sys.argv[1] if len(sys.argv) > 1 else "docs"
    db_path = "db"
    
    ingest(docs_path, db_path)
