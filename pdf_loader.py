"""
pdf_loader.py – Step 2: PDF Text Extraction
=============================================
Concept: Structured data extraction from PDFs.

Before we can embed anything, we need raw text with provenance — i.e.
we must know *which file* and *which page* each piece of text came from.
That metadata travels all the way to the final citation shown to the user.

pypdf's PdfReader reads the PDF object tree; extract_text() calls the
PDF content-stream parser to reconstruct the text from drawing commands.
It works on text-based PDFs but *cannot* OCR scanned images.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def load_pdf(path: str | Path) -> list[dict]:
    """Extract text page-by-page from a single PDF file.

    Args:
        path: Absolute or relative path to the PDF.

    Returns:
        A list of page records, one per non-empty page::

            [
              {"file": "attention.pdf", "page": 1, "text": "Abstract ..."},
              {"file": "attention.pdf", "page": 2, "text": "1 Introduction ..."},
              ...
            ]

        Pages that yield no extractable text (e.g. scanned image pages)
        are silently skipped — they would produce empty embeddings anyway.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    records: list[dict] = []

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        # Handles corrupt files or encrypted PDFs without a password.
        print(f"[WARN] Could not open {path.name}: {exc}", file=sys.stderr)
        return records

    if reader.is_encrypted:
        # We could try reader.decrypt("") for no-password encryption,
        # but for simplicity we skip and warn the user.
        print(
            f"[WARN] {path.name} is encrypted — skipping. "
            "Decrypt it first with: qpdf --decrypt input.pdf output.pdf",
            file=sys.stderr,
        )
        return records

    total_pages = len(reader.pages)
    print(f"  Loading {path.name}  ({total_pages} pages)…")

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Clean up whitespace: pypdf sometimes returns text with excessive
        # newlines and spaces that would confuse the chunker later.
        text = " ".join(text.split())

        if not text:
            # Image-only page — nothing to embed, skip it.
            continue

        records.append(
            {
                "file": path.name,   # just the filename, not the full path
                "page": page_num,
                "text": text,
            }
        )

    print(f"    => {len(records)} pages with extractable text")
    return records


def load_folder(folder: str | Path) -> list[dict]:
    """Load all PDFs from a folder, returning one record per page.

    Args:
        folder: Path to the directory containing PDFs.

    Returns:
        Combined list of page records sorted by (filename, page number).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {folder}", file=sys.stderr)
        return []

    print(f"Found {len(pdf_files)} PDF(s) in {folder}\n")

    all_records: list[dict] = []
    for pdf_path in pdf_files:
        all_records.extend(load_pdf(pdf_path))

    print(f"\nTotal pages extracted: {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Quick sanity-check: run this file directly to see what gets extracted.
# Usage:  python pdf_loader.py [path/to/folder_or_file]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")

    if target.is_file():
        records = load_pdf(target)
    else:
        records = load_folder(target)

    print("\n--- Preview of first 3 records ---")
    for rec in records[:3]:
        snippet = rec["text"][:200].replace("\n", " ")
        print(f"\nFile : {rec['file']}")
        print(f"Page : {rec['page']}")
        print(f"Text : {snippet}…")
