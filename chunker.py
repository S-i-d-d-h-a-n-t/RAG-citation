"""
chunker.py – Step 3: Text Chunking
===================================
Concept: Context Windows and Overlap

LLMs and embedding models have limited context windows. If we embed an entire 
page, the resulting vector becomes a "mush" of many topics, making it hard to 
match a specific question. We split text into smaller "chunks".

Overlap ensures that if a sentence or concept spans a chunk boundary, it isn't 
cut in half. At least one chunk will capture the full context.

We use simple word-level chunking here (splitting by space).
"""

from __future__ import annotations

CHUNK_SIZE = 500
OVERLAP = 50

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split a string into chunks of roughly `chunk_size` words, with `overlap` words."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap")
        
    step = chunk_size - overlap
    
    for i in range(0, len(words), step):
        # Slice out the window of words
        window = words[i : i + chunk_size]
        chunk = " ".join(window)
        chunks.append(chunk)
        
        # If this window reached the end of the text, stop
        if i + chunk_size >= len(words):
            break
            
    return chunks

def chunk_records(records: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """
    Takes a list of page records: [{"file": str, "page": int, "text": str}]
    Returns a list of chunk records: [{"file": str, "page": int, "chunk_index": int, "text": str}]
    """
    chunked_records = []
    
    for record in records:
        text = record["text"]
        chunks = chunk_text(text, chunk_size, overlap)
        
        for idx, chunk in enumerate(chunks):
            chunked_records.append({
                "file": record["file"],
                "page": record["page"],
                "chunk_index": idx,
                "text": chunk
            })
            
    return chunked_records

# ---------------------------------------------------------------------------
# Quick sanity-check: run this file directly to test the chunker
# Usage: python chunker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The greenhouse effect is a natural process that warms Earth's surface. "
        "Solar radiation passes through the atmosphere and heats the surface. "
        "The surface then emits infrared (heat) radiation back toward space. "
        "Greenhouse gases -- water vapour, carbon dioxide, methane, and nitrous "
        "oxide -- absorb some of this infrared radiation and re-emit it in all "
        "directions, trapping heat."
    )
    
    print(f"Original text length: {len(sample_text.split())} words\n")
    
    # Test with very small chunks to demonstrate overlap
    demo_chunk_size = 15
    demo_overlap = 5
    
    print(f"Chunking with size={demo_chunk_size}, overlap={demo_overlap}...\n")
    chunks = chunk_text(sample_text, chunk_size=demo_chunk_size, overlap=demo_overlap)
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ({len(chunk.split())} words) ---")
        print(chunk)
        print()
