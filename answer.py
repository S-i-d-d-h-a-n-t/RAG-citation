"""
answer.py – Step 6: Answer Generation with Citations
======================================================
Concept: Grounded Generation

Instead of letting the LLM answer from its pre-training data, we give it the 
search chunks and strictly instruct it to use *only* those chunks. 
We ask it to cite its sources using [1], [2] markers, which we then parse.
"""

import os
import sys
import re
import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

from search import search

# Read the Gemini key from the environment
load_dotenv()

def generate_answer(question: str, store_dir: str = "db", k: int = 5) -> dict:
    """
    1. Search for top chunks.
    2. Build a grounded prompt.
    3. Call Gemini.
    4. Parse citations and return a structured response.
    """
    # 1. Retrieve context
    chunks = search(question, store_dir=store_dir, k=k)
    
    if not chunks:
        return {
            "answer": "No documents found in the database. Please ingest some PDFs first.",
            "sources": []
        }
        
    # 2. Build the prompt
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(f"Source [{i}]:\n{chunk['text']}")
        
    context_text = "\n\n".join(context_blocks)
    
    system_instruction = (
        "You are a helpful and precise assistant. You must answer the user's question "
        "using ONLY the provided context.\n"
        "If the context does not contain the answer, you must say exactly: "
        "'I couldn't find this in the documents.'\n"
        "When you use information from a source, you MUST cite it using the source number in brackets, e.g. [1] or [2]."
    )
    
    prompt = f"Context documents:\n{context_text}\n\nQuestion: {question}"
    
    # 3. Call Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2, # Low temperature for factual generation
        )
    )
    
    answer_text = response.text
    
    # 4. Parse citations to return the relevant metadata
    # We find all [n] markers in the answer
    cited_indices = set([int(x) for x in re.findall(r'\[(\d+)\]', answer_text)])
    
    sources = []
    for i in cited_indices:
        if 1 <= i <= len(chunks):
            chunk = chunks[i-1]
            sources.append({
                "n": i,
                "file": chunk["metadata"]["file"],
                "page": chunk["metadata"]["page"],
                "snippet": chunk["text"][:150] + "..."
            })
            
    return {
        "answer": answer_text,
        "sources": sorted(sources, key=lambda x: x["n"])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    args = parser.parse_args()
    
    print(f"Question: {args.question}\n")
    
    try:
        result = generate_answer(args.question)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    print("--- Answer ---")
    print(result["answer"])
    print("\n--- Citations ---")
    if not result["sources"]:
        print("None.")
    for source in result["sources"]:
        print(f"[{source['n']}] {source['file']} (Page {source['page']})")
