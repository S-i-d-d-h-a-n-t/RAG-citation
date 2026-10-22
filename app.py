"""
app.py – Step 7: Streamlit UI
=============================
Concept: The Full RAG Loop

This script ties everything together into a clean web interface using Streamlit.
- Sidebar: Upload PDFs -> saves them to `docs/` -> runs `ingest.py`.
- Main area: Text input for questions -> calls `answer.py` -> displays results 
  with expandable citation cards.
"""

import os
from pathlib import Path

import streamlit as st

from ingest import ingest
from answer import generate_answer

# --- Page Config & Aesthetics ---
st.set_page_config(
    page_title="RAG Learning",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    .source-card {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Local PDF Q&A")
st.markdown("Ask questions in natural language and get answers grounded *strictly* in your documents, complete with citations.")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("📄 Document Management")
    st.markdown("Upload PDFs here. They will be embedded and added to the local VectorStore.")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            docs_dir = Path("docs")
            docs_dir.mkdir(exist_ok=True)
            
            with st.spinner("Saving and ingesting documents... This may take a moment."):
                # 1. Save uploaded files to the docs/ directory
                for f in uploaded_files:
                    file_path = docs_dir / f.name
                    with open(file_path, "wb") as out_file:
                        out_file.write(f.getbuffer())
                
                # 2. Run the ingestion pipeline (chunking + embedding)
                ingest(docs_dir="docs", store_dir="db")
                
            st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
        else:
            st.warning("Please select at least one PDF to upload.")
            
    st.divider()
    st.caption("Step 7 of the RAG Learning Project.")

# --- Main Area: Question & Answer ---
st.markdown("### Ask a Question")
question = st.text_input("What would you like to know?", placeholder="e.g. What is the vanishing gradient problem?")

if st.button("Get Answer", type="primary"):
    if not question:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching VectorStore and generating answer..."):
            try:
                # Run the full retrieval and generation pipeline
                result = generate_answer(question, store_dir="db", k=5)
                
                # Display Answer
                st.markdown("### Answer")
                st.info(result["answer"])
                
                # Display Citations (if any)
                if result.get("sources"):
                    st.markdown("### Sources")
                    for source in result["sources"]:
                        with st.expander(f"**[{source['n']}] {source['file']}** (Page {source['page']})"):
                            st.write(f"... {source['snippet']} ...")
                            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.markdown("*(Did you set the `GEMINI_API_KEY` environment variable?)*")
