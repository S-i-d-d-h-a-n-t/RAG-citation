# RAG Learning Project — Development Log

> **Project:** Local PDF Q&A with citations, built from scratch to learn RAG.
> **Stack:** Python 3.12 · pypdf · sentence-transformers · numpy · Gemini API · Streamlit
> **Repo root:** `rag-learning/`
> **Visual diagram:** `progress.mmd` (Mermaid flowchart)

---

## Current Status Summary

| Step | Component | Status | Key Deliverable |
|------|-----------|--------|-----------------|
| **1** | Setup & Embeddings Demo | ✅ Complete | `.venv`, `requirements.txt`, `embed_demo.py`, `vector_store.py` |
| **2** | PDF Extraction & Provenance | ✅ Complete | `pdf_loader.py`, `create_sample_pdfs.py`, 3 sample PDFs (9 pages) |
| **3** | Text Chunking | ✅ Complete | `chunker.py` (word window + overlap) |
| **4** | Embedding & Vector Storage | ✅ Complete | `ingest.py` (end-to-end ingestion pipeline) |
| **5** | Semantic Search & Retrieval | ✅ Complete | `search.py` (top-k query CLI) |
| **6** | Generation with Citations | ✅ Complete | `answer.py` (Gemini 2.5 Flash + citation parsing) |
| **7** | Streamlit Web Interface | ✅ Complete | `app.py` (interactive browser UI) |
| **8** | Evaluation & Benchmarking | ✅ Complete | `eval.py` (retrieval & citation accuracy metrics) |

---

## Project Structure (Current State)

```
rag-learning/
├── .venv/                   # Python 3.12 virtual environment
├── PROGRESS.md              # Detailed chronological progress log (this file)
├── progress.mmd             # Mermaid diagram of roadmap and data flow
├── requirements.txt         # Pinned production dependencies
│
├── [Completed / Active]
├── embed_demo.py            # Step 1: Embeddings & cosine similarity demo
├── vector_store.py          # Step 1: Hand-rolled persistent NumPy vector store
├── create_sample_pdfs.py    # Step 2: Test PDF generator (fpdf2)
├── pdf_loader.py            # Step 2: Page-by-page text extractor with provenance
│
├── [Data & Storage]
├── docs/                    # Test document directory (9 total pages)
│   ├── climate_science.pdf  # 3 pages: greenhouse effect, carbon cycle, tipping points
│   ├── neural_networks.pdf  # 3 pages: architectures, activation functions, regularisation
│   └── python_concurrency.pdf # 3 pages: GIL & processes, asyncio, concurrency pitfalls
├── chroma_db/               # Created earlier; superseded by custom VectorStore
│
└── [Pipeline Stubs — Steps 3-8]
    ├── (chunker.py)         # Step 3: To be created
    ├── ingest.py            # Step 4: Batch PDF ingest to VectorStore (stub)
    ├── search.py            # Step 5: Query retrieval CLI (stub)
    ├── answer.py            # Step 6: Grounded Gemini generation & citation parser (stub)
    ├── app.py               # Step 7: Streamlit UI (stub)
    └── eval.py              # Step 8: Retrieval and citation evaluator (stub)
```

---

## Architecture Overview

```
docs/          PDFs
   |
pdf_loader.py  extract text page-by-page  --> [{file, page, text}, ...]
   |
chunker.py     split into ~500-word windows --> [{file, page, chunk_id, text}, ...]
   |
ingest.py      embed + store in VectorStore (numpy/.npz + metadata.json)
   |
search.py      embed question, cosine-similarity top-k
   |
answer.py      build grounded prompt --> Gemini --> parse [n] citations
   |
app.py         Streamlit UI
eval.py        automated accuracy report
```

---

## Step-by-Step Progress

### Step 1 — Setup & Embeddings Demo ✅
**Completed:** 2026-10-07

**What was built:**
| File | Purpose |
|------|---------|
| `.venv/` | Isolated Python 3.12 virtual environment |
| `requirements.txt` | Pinned deps: `pypdf`, `sentence-transformers`, `numpy`, `google-genai`, `streamlit`, `python-dotenv` |
| `vector_store.py` | Hand-rolled numpy vector store (replaces ChromaDB — no C++ compiler needed on Windows) |
| `embed_demo.py` | Standalone demo: embed 5 sentences, rank by cosine similarity to a question |
| Stub files | `ingest.py`, `search.py`, `answer.py`, `app.py`, `eval.py` |

**Concept demonstrated:** Embeddings & semantic similarity
- `all-MiniLM-L6-v2` maps text to a 384-dimensional float vector
- Cosine similarity measures the angle between two vectors:
  $$\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
  (1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite)
- The model matched *"How do plants produce energy from light?"* to *"Photosynthesis converts sunlight into glucose in plants."* with score **0.699** — despite zero shared keywords.

**Sample output (`embed_demo.py`):**
```
Question: "How do plants produce energy from light?"
 Score  Sentence
----------------------------------------------------------------------
 0.699  Photosynthesis converts sunlight into glucose in plants.  <- best match
 0.325  The mitochondria is the powerhouse of the cell.
 0.071  Machine learning models learn patterns from data.
 0.055  Python is a popular programming language.
 0.042  The Eiffel Tower is located in Paris, France.
```

**`VectorStore` Internals (`vector_store.py`):**
- **Public API:**
  - `add(ids, texts, embeddings, metadatas) -> int`: Deduplicates by ID using a set check; appends new items; writes state to disk.
  - `query(query_embedding, k=5) -> list[dict]`: Normalizes embeddings to unit length, computes batch matrix-vector dot product (`normed @ q`), uses `np.argpartition` for fast $O(N)$ top-$k$ selection, and sorts results descending.
  - `count() -> int`: Returns current store size.
- **Persistence format:**
  - `embeddings.npz`: Compressed NumPy archive storing float32 embeddings array and string IDs array.
  - `metadata.json`: JSON file storing document text chunks and custom metadata dictionaries (`file`, `page`, etc.).
- **Windows architectural advantage:** Eliminates MSVC C++ toolchain requirements that ChromaDB requires for `hnswlib`.

**Gotchas resolved:**
- ChromaDB 0.4-0.6 requires MSVC C++ compiler on Windows to build `chroma-hnswlib` => replaced with custom `vector_store.py` (pure numpy).
- Windows PowerShell terminal uses `cp1252` encoding => avoid Unicode arrows/special chars in `print()` calls.

---

### Step 2 — PDF Loading & Provenance Tracking ✅
**Completed:** 2026-10-07

**What was built:**
| File | Purpose |
|------|---------|
| `pdf_loader.py` | Extracts text page-by-page using `pypdf`, normalizes whitespace, attaches `{file, page}` metadata |
| `create_sample_pdfs.py` | Script generating 3 realistic multi-page PDFs (auto-installs `fpdf2` if missing) |
| `docs/neural_networks.pdf` | 3-page sample PDF covering ML concepts |
| `docs/python_concurrency.pdf` | 3-page sample PDF covering Python threading, multiprocessing, and asyncio |
| `docs/climate_science.pdf` | 3-page sample PDF covering climate systems, carbon cycle, and tipping points |

**Concept demonstrated:** Structured extraction with provenance
- `PdfReader.pages` iterates the PDF page object tree.
- `page.extract_text()` reconstructs text from PDF drawing commands.
- Whitespace cleanup: `" ".join(text.split())` prevents pypdf newline artifacts from fragmenting sentences.
- Every record carries `file` + `page` metadata that travels all the way to the final citation.

**Sample Document Catalog (`docs/`):**
1. **`neural_networks.pdf` (3 pages):**
   - *Page 1:* Network structure, forward propagation, loss functions (cross-entropy, MSE).
   - *Page 2:* Activation functions (ReLU, Sigmoid, Softmax) and the vanishing gradient problem.
   - *Page 3:* Regularisation methods (Dropout rates 0.2–0.5, L2 weight decay, Batch normalisation).
2. **`python_concurrency.pdf` (3 pages):**
   - *Page 1:* Threading vs. multiprocessing, CPython Global Interpreter Lock (GIL), CPU-bound workloads.
   - *Page 2:* `asyncio` event loop, coroutines (`async`/`await`), primitives (`gather`, `Queue`, `Semaphore`).
   - *Page 3:* Concurrency selection guide, race conditions, locks, deadlocks, event loop blocking.
3. **`climate_science.pdf` (3 pages):**
   - *Page 1:* Greenhouse effect physics, infrared re-radiation, greenhouse gases (H2O, CO2, CH4, N2O), +1.2°C temperature anomaly.
   - *Page 2:* Carbon cycle, photosynthesis (120 GtC/yr), respiration, ocean CO2 absorption (26% emissions) and thermal feedback.
   - *Page 3:* Climate tipping points (Greenland ice sheet 7m sea level rise, Amazon dieback, permafrost thaw with ~1.5T tonnes carbon).

**Key functions in `pdf_loader.py`:**
```python
load_pdf(path)      -> [{"file": str, "page": int, "text": str}, ...]
load_folder(folder) -> combined list sorted by (file, page)
```

**Edge cases handled:**
- Encrypted PDFs => clear warning and skip.
- Corrupt PDFs => `PdfReadError` caught gracefully and skipped.
- Image-only pages (no extractable text) => silently skipped to prevent zero-vector noise.
- Excessive whitespace and erratic newlines => normalized cleanly.

**Sample output (`pdf_loader.py`):**
```
Found 3 PDF(s) in docs
  Loading climate_science.pdf  (3 pages)... => 3 pages with extractable text
  Loading neural_networks.pdf  (3 pages)... => 3 pages with extractable text
  Loading python_concurrency.pdf  (3 pages)... => 3 pages with extractable text

Total pages extracted: 9
```

**Gotchas resolved:**
- `fpdf2` built-in Helvetica font is Latin-1 only => replaced all Unicode characters (em dashes, degree signs, subscripts) with ASCII equivalents in sample generator.
- `ln=True` deprecated in fpdf2 v2.5.2+ => switched to `new_x=XPos.LMARGIN, new_y=YPos.NEXT`.
- Windows `cp1252` terminal => replaced Unicode symbols with ASCII-safe tokens (`=>`, `...`) in console prints.

---

### Step 3 — Text Chunking ✅
**Completed:** 2026-10-07

**What was built:**
- `chunker.py` with `chunk_text(text, chunk_size, overlap)` that splits text into word-level windows.
- `chunk_records()` that maps `{file, page, text}` to `[{file, page, chunk_index, text}]`.

**Concept demonstrated:** Context Windows and Overlap
- Overlap prevents sentences spanning chunk boundaries from being torn in half.
- Configured to 500-word chunks with 50-word overlaps to match embedding model context windows.

---

### Step 4 — Embedding & Storage ✅
**Completed:** 2026-10-07

**What was built:**
- `ingest.py` which runs the full offline ingestion pipeline:
  1. Calls `load_folder()` from Step 2 to parse all PDFs.
  2. Calls `chunk_records()` from Step 3 to split into smaller context windows.
  3. Uses `SentenceTransformer("all-MiniLM-L6-v2")` to vectorize chunks.
  4. Stores results in `VectorStore(db)`.

**Concept demonstrated:** Deterministic pipelines
- Chunk IDs are generated using `hashlib.sha256(f"{file}-{page}-{chunk_index}")`.
- This ensures that re-running `ingest.py` is idempotent and skips duplicates.

---

### Step 5 — Semantic Search & Retrieval ✅
**Completed:** 2026-10-07

**What was built:**
- `search.py` containing `search(question, k=5)` which interfaces with `VectorStore.query()`.
- Added a CLI interface to let users test arbitrary questions and see raw cosine scores.

**Concept demonstrated:** Nearest-Neighbour Search
- Successfully tested the pipeline by querying "What is the vanishing gradient problem?".
- The retrieval perfectly matched the question to `neural_networks.pdf` (Page 2) with the highest score, bypassing exact keyword limitations.

---

### Step 6 — Answer Generation with Citations ✅
**Completed:** 2026-10-07

**What was built:**
- `answer.py` which takes retrieved chunks, builds a numbered context prompt, and calls `gemini-2.5-flash`.
- Uses a strict system instruction: "You must answer the user's question using ONLY the provided context."
- Implemented Regex `\[(\d+)\]` to parse citations and map them back to the exact source chunk `{file, page}`.

---

### Step 7 — Streamlit UI ✅
**Completed:** 2026-10-07

**What was built:**
- `app.py` combining the ingestion pipeline and RAG answer generation into a unified frontend.
- **Sidebar:** Document upload section that writes PDFs to `docs/` and automatically triggers `ingest.py`.
- **Main view:** Question input, Markdown-formatted answer display, and expandable `st.expander` citation cards matching the `[n]` citation syntax.

**Concept demonstrated:** Full RAG Loop
- Validated end-to-end functionality from PDF upload → vector storage → semantic search → grounded LLM generation via Gemini.

---

### Step 8 — Evaluation ✅
**Completed:** 2026-10-07

**What was built:**
- Created `test_questions.json` containing 10 test queries (8 answerable based on our PDFs, 2 unanswerable "trick" questions).
- `eval.py` which runs the full pipeline iteratively over the test suite and outputs a metric report.

**Concept demonstrated:** RAG Quality Metrics
- **Retrieval Hit Rate:** Checks if the required document page was present in the top-K vectors.
- **Citation Accuracy:** Checks if the LLM correctly grounded its generation by returning the expected `[n]` citation.
- **Refusal Rate:** Evaluates if the strict system prompt prevents hallucination when asked questions entirely outside the provided vector context (e.g. "What is the capital of France?").

---

## Commands Reference

```powershell
# Activate environment
.venv\Scripts\Activate.ps1

# Step 1 -- embeddings demo
python embed_demo.py

# Step 2 -- generate sample PDFs + test loader
python create_sample_pdfs.py
python pdf_loader.py

# Step 3 (coming) -- test chunker
python chunker.py

# Step 4 (coming) -- ingest all PDFs
python ingest.py docs/

# Step 5 (coming) -- interactive retrieval test
python search.py "What is the vanishing gradient problem?"

# Step 6 (coming) -- full pipeline single question
python answer.py "What is the vanishing gradient problem?"

# Step 7 (coming) -- launch UI
streamlit run app.py

# Step 8 (coming) -- run evaluation
python eval.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Custom `vector_store.py` instead of ChromaDB | ChromaDB requires MSVC C++ compiler on Windows; numpy brute-force cosine similarity is identical math, just without ANN index optimisation (only matters at millions of chunks) |
| Word-level chunking (not character-level) | More intuitive to reason about; 500 words is roughly one coherent topic paragraph |
| Page-level metadata attached at load time | Provenance must be tracked from the first step; retrofitting it later is error-prone |
| Numbered `[n]` citation scheme | Forces the LLM to be explicit about which source supports each claim; parseable with regex |
| `gemini-2.5-flash` for generation | Fast and cost-effective; strong instruction-following for citation constraints |
| No LangChain | Building each layer manually makes the data flow transparent and debuggable |

---

## Experiments to Try After Completion

1. **Chunk size sweep** — try 250 / 500 / 1000 words; measure retrieval hit rate with `eval.py`
2. **k sweep** — try k = 3, 5, 10; watch how answer quality and hallucination rate change
3. **Hybrid search** — add BM25 keyword score, combine with cosine score (linear interpolation)
4. **Reranking** — use a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top-k chunks
5. **Larger embedding model** — swap `all-MiniLM-L6-v2` (384-dim) for `all-mpnet-base-v2` (768-dim); compare scores
