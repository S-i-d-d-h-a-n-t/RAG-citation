"""
eval.py – Step 8: Evaluation & Benchmarking
===========================================
Concept: Automated RAG Testing

We evaluate three core metrics:
1. Retrieval Hit Rate: Was the correct page present in the top-K retrieved chunks?
2. Citation Accuracy: Did the LLM cite the correct source for the answer?
3. Refusal Rate: Did the LLM refuse to answer questions not covered in the docs?
"""

import json
import sys
import time
from pathlib import Path

from search import search
from answer import generate_answer

def run_evaluation(questions_file: str = "test_questions.json", k: int = 5):
    path = Path(questions_file)
    if not path.exists():
        print(f"Error: {questions_file} not found.")
        sys.exit(1)
        
    with open(path, "r", encoding="utf-8") as f:
        tests = json.load(f)
        
    total_answerable = 0
    total_unanswerable = 0
    
    retrieval_hits = 0
    citation_hits = 0
    refusal_hits = 0
    
    print(f"Running evaluation on {len(tests)} questions...\n")
    
    for idx, test in enumerate(tests, 1):
        time.sleep(2)  # Pause to avoid Gemini API 503/Rate Limit errors
        question = test["question"]
        expected_file = test["expected_file"]
        expected_page = test["expected_page"]
        is_answerable = test["answerable"]
        
        print(f"[{idx}/{len(tests)}] Q: {question}")
        
        # 1. Test Retrieval
        chunks = search(question, k=k)
        retrieval_success = False
        
        if is_answerable:
            total_answerable += 1
            for chunk in chunks:
                if chunk["metadata"]["file"] == expected_file and chunk["metadata"]["page"] == expected_page:
                    retrieval_success = True
                    break
            
            if retrieval_success:
                retrieval_hits += 1
            else:
                print(f"  [FAIL] Retrieval failed to find {expected_file} (Page {expected_page}) in top-{k}")
                
        # 2. Test End-to-End Answer Generation
        try:
            result = generate_answer(question, k=k)
            answer_text = result["answer"]
            sources = result["sources"]
            
            if is_answerable:
                # Check if the LLM cited the correct source
                citation_success = False
                for source in sources:
                    if source["file"] == expected_file and source["page"] == expected_page:
                        citation_success = True
                        break
                        
                if citation_success:
                    citation_hits += 1
                else:
                    print(f"  [FAIL] Generation failed to cite {expected_file} (Page {expected_page})")
            else:
                total_unanswerable += 1
                # Check if it refused appropriately
                if "I couldn't find this in the documents" in answer_text:
                    refusal_hits += 1
                else:
                    print(f"  [FAIL] Hallucination detected! Answered unanswerable question: {answer_text[:50]}...")
                    
        except Exception as e:
            print(f"  [FAIL] Error generating answer: {e}")
            
    # Print Report
    print("\n" + "="*40)
    print(" [REPORT] EVALUATION REPORT")
    print("="*40)
    
    if total_answerable > 0:
        print(f"Retrieval Hit Rate: {retrieval_hits}/{total_answerable} ({(retrieval_hits/total_answerable)*100:.1f}%)")
        print(f"Citation Accuracy:  {citation_hits}/{total_answerable} ({(citation_hits/total_answerable)*100:.1f}%)")
    
    if total_unanswerable > 0:
        print(f"Refusal Rate:       {refusal_hits}/{total_unanswerable} ({(refusal_hits/total_unanswerable)*100:.1f}%)")
        
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
