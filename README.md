# RAG-citation
# 📚 RAG Learning — Local PDF Q&A with Citations

A from-scratch Retrieval-Augmented Generation (RAG) system that lets you upload PDFs, ask questions in natural language, and get answers grounded strictly in your documents — complete with page-level citations. Built with Python, Streamlit, sentence-transformers, NumPy, and the Gemini API.

> **No LangChain, no ChromaDB, no heavy C++ dependencies.** Every layer (PDF loading → chunking → embedding → vector store → search → answer generation) is built manually so the data flow is transparent and debuggable.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF ingestion** | Extract text page-by-page with provenance (file name + page number) |
| ✂️ **Text chunking** | Overlapping word-window chunking (~500 words per chunk) |
| 🔢 **Embeddings** | `all-MiniLM-L6-v2` sentence-transformers model (384-dim vectors) |
| 🗄️ **Vector store** | Hand-rolled persistent NumPy vector store with cosine similarity |
| 🔍 **Semantic search** | Top-k retrieval by cosine similarity |
| 💬 **Grounded Q&A** | Gemini generates answers using only retrieved chunks, with `[n]` citation markers |
| 🖥️ **Streamlit UI** | Browser-based interface for uploading PDFs and asking questions |
| 📊 **Evaluation** | Retrieval hit rate, citation accuracy, and refusal-rate metrics |
| 🧪 **Sample PDFs** | 3 test documents (climate science, neural networks, Python concurrency) |

---

## 📂 Project Structure

```
rag-learning/
├── .gitignore              # Git ignore rules (secrets, venv, runtime data)
├── .env                    # API keys (NOT committed — create your own)
├── README.md               # This file
├── PROGRESS.md             # Development log & design decisions
├── progress.mmd            # Mermaid roadmap diagram
├── requirements.txt        # Pinned Python dependencies
├── test_questions.json     # 10 test queries for evaluation
│
├── app.py                  # Streamlit web interface
├── answer.py               # Grounded answer generation + citation parsing
├── chunker.py              # Text chunking with overlap
├── embed_demo.py           # Embeddings & cosine similarity demo
├── eval.py                 # RAG evaluation (hit rate, citation accuracy)
├── ingest.py               # End-to-end ingestion pipeline
├── pdf_loader.py           # Page-by-page PDF text extraction
├── search.py               # Semantic search CLI
└── vector_store.py         # Hand-rolled NumPy vector store
│
├── docs/                   # Sample PDFs (upload your own here too)
└── db/                     # Generated vector store (created at runtime)
```

## 🏗️ Architecture & Data Flow

```
docs/ (PDFs)
   │
   ▼
pdf_loader.py      Extract text page-by-page  →  [{file, page, text}, ...]
   │
   ▼
chunker.py         Split into ~500-word windows →  [{file, page, chunk_id, text}, ...]
   │
   ▼
ingest.py          Embed (all-MiniLM-L6-v2) + store in VectorStore (NumPy/.npz + metadata.json)
   │
   ▼
search.py          Embed question → cosine-similarity top-k retrieval
   │
   ▼
answer.py          Build grounded prompt → Gemini → parse [n] citations
   │
   ▼
app.py             Streamlit UI (upload PDFs, ask questions, view citations)
```

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-learning.git
cd rag-learning
```

### 2. Create a virtual environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 🖥️ Usage

### Option A: Streamlit Web Interface (Recommended)

```powershell
streamlit run app.py
```

1. Upload one or more PDFs in the sidebar.
2. Click **Process Documents** (this runs chunking + embedding + storage).
3. Type a question in the main area and click **Get Answer**.
4. View the answer with expandable citation cards showing source file, page, and snippet.

### Option B: Command Line (Step-by-Step)

```powershell
# Embeddings demo
python embed_demo.py

# Generate sample PDFs + test loader
python create_sample_pdfs.py
python pdf_loader.py

# Test chunker
python chunker.py

# Ingest all PDFs into the vector store
python ingest.py docs/

# Interactive retrieval test
python search.py "What is the vanishing gradient problem?"

# Full pipeline single question (CLI)
python answer.py "What is the vanishing gradient problem?"

# Launch Streamlit UI
streamlit run app.py

# Run evaluation
python eval.py
```

## ⚙️ Configuration

| File | Purpose |
|------|---------|
| `.env` | `GEMINI_API_KEY` — your Google Gemini API key |
| `requirements.txt` | Pinned Python dependencies |
| `test_questions.json` | 10 test queries (8 answerable, 2 unanswerable "trick" questions) |

Key constants used across the pipeline:
- **Embedding model:** `all-MiniLM-L6-v2` (384 dimensions, fast and lightweight)
- **Chunk size:** ~500 words with overlap
- **Top-k retrieval:** `k=5` (configurable in `search.py` and `answer.py`)
- **LLM:** `gemini-2.5-flash` (temperature 0.2 for factual generation)
- **Vector store:** NumPy brute-force cosine similarity (persisted as `.npz` + `metadata.json`)

## 📊 Evaluation

Run `eval.py` to measure RAG quality:

```bash
python eval.py
```

It reports three metrics:

| Metric | What it measures |
|--------|------------------|
| **Retrieval Hit Rate** | Whether the required document page appears in the top-k retrieved chunks |
| **Citation Accuracy** | Whether the LLM correctly cites the expected `[n]` source markers |
| **Refusal Rate** | Whether the system correctly refuses to answer questions outside the provided documents |

## 🧠 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Custom `vector_store.py` instead of ChromaDB** | ChromaDB requires MSVC C++ compiler on Windows. NumPy brute-force cosine similarity is identical math, just without ANN index optimisation (only matters at millions of chunks). |
| **Word-level chunking** | More intuitive than character-level; ~500 words is roughly one coherent topic paragraph. |
| **Page-level metadata at load time** | Provenance is tracked from the first step; retrofitting it later is error-prone. |
| **Numbered `[n]` citation scheme** | Forces the LLM to be explicit about which source supports each claim; parseable with regex. |
| **`gemini-2.5-flash`** | Fast, cost-effective, strong instruction-following for citation constraints. |
| **No LangChain** | Building each layer manually makes the data flow transparent and debuggable for learning purposes. |

## 🔬 Experiments to Try

1. **Chunk size sweep** — try 250 / 500 / 1000 words; measure retrieval hit rate with `eval.py`
2. **k sweep** — try k = 3, 5, 10; watch how answer quality and hallucination rate change
3. **Hybrid search** — add BM25 keyword score, combine with cosine score (linear interpolation)
4. **Reranking** — use a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L6-v2`) to re-score top-k chunks
5. **Larger embedding model** — swap `all-MiniLM-L6-v2` (384-dim) for `all-mpnet-base-v2` (768-dim)

## 📜 License

This project is for educational purposes. Feel free to use it as a learning reference for building RAG systems from scratch.
