"""
embed_demo.py – Step 1 Learning Demo
=====================================
Concept: Embeddings & Semantic Similarity

An embedding model converts text into a list of numbers (a "vector").
Sentences with similar *meaning* end up with vectors that point in
similar directions, even if they use different words.

Cosine similarity measures the angle between two vectors:
  - 1.0  → identical direction (same meaning)
  - 0.0  → perpendicular (unrelated)
  - -1.0 → opposite direction

This is the heart of RAG retrieval: embed the question, embed every
chunk, find the chunks whose vectors are closest to the question vector.
"""

from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------------------------
# 1. Load the model (downloads ~90 MB on first run, cached afterwards)
# ---------------------------------------------------------------------------
print("Loading embedding model…")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# 2. Five sample sentences covering different topics
# ---------------------------------------------------------------------------
sentences = [
    "The mitochondria is the powerhouse of the cell.",          # biology
    "Photosynthesis converts sunlight into glucose in plants.", # biology (related)
    "Python is a popular programming language.",                # tech
    "Machine learning models learn patterns from data.",        # tech (related)
    "The Eiffel Tower is located in Paris, France.",            # geography (unrelated)
]

# ---------------------------------------------------------------------------
# 3. A question – notice it doesn't share keywords with the best match
# ---------------------------------------------------------------------------
question = "How do plants produce energy from light?"

# ---------------------------------------------------------------------------
# 4. Embed everything (each sentence → a 384-dimensional vector)
# ---------------------------------------------------------------------------
print("\nEmbedding sentences…")
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
question_embedding  = model.encode(question,  convert_to_tensor=True)

# ---------------------------------------------------------------------------
# 5. Compute cosine similarity between the question and every sentence
# ---------------------------------------------------------------------------
scores = util.cos_sim(question_embedding, sentence_embeddings)[0]  # shape: (5,)

# ---------------------------------------------------------------------------
# 6. Print results, sorted from most to least similar
# ---------------------------------------------------------------------------
ranked = sorted(
    zip(scores.tolist(), sentences),
    key=lambda x: x[0],
    reverse=True,
)

print(f"\nQuestion: \"{question}\"\n")
print(f"{'Score':>6}  Sentence")
print("-" * 70)
for score, sentence in ranked:
    marker = "  ← best match" if sentence == ranked[0][1] else ""
    print(f"{score:>6.3f}  {sentence}{marker}")

print("\nKey insight:")
print("  The top match is about PHOTOSYNTHESIS, not Python or Eiffel Tower.")
print("  The model understood MEANING, not just keywords.")
print("  This is exactly how RAG finds relevant chunks for your question.")
