"""
vector_store.py – Tiny persistent vector store built on numpy.
=================================================================
Why not ChromaDB?  ChromaDB's C++ HNSW index requires MSVC to compile
on Windows, which adds a heavy dev-tools dependency.  Building it
ourselves also makes the internals visible:

  1. Embeddings are stored as a 2-D numpy array (rows = chunks).
  2. A query embedding is compared to every row via cosine similarity.
  3. The top-k row indices are returned together with their metadata.

This is *exactly* what every vector DB does — ours just skips the
ANN (approximate nearest neighbour) optimisation, which only matters
at millions of chunks.  For a learning project it's perfectly fine.

Persistence: one .npz file (embeddings + ids) + one JSON file
(metadata + texts), both written to a directory you choose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    """A dead-simple, file-backed vector store."""

    EMBEDDINGS_FILE = "embeddings.npz"
    METADATA_FILE   = "metadata.json"

    def __init__(self, store_dir: str) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._ids:        list[str]        = []
        self._texts:      list[str]        = []
        self._metadatas:  list[dict]       = []
        self._embeddings: np.ndarray | None = None   # shape (N, dim)

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        ids:        list[str],
        texts:      list[str],
        embeddings: list[list[float]],
        metadatas:  list[dict],
    ) -> int:
        """Add chunks that don't already exist (idempotent on id).

        Returns the number of *new* chunks actually stored.
        """
        new_count = 0
        existing = set(self._ids)

        new_ids, new_texts, new_metas, new_embs = [], [], [], []
        for uid, text, emb, meta in zip(ids, texts, embeddings, metadatas):
            if uid in existing:
                continue          # skip duplicates – makes re-runs safe
            new_ids.append(uid)
            new_texts.append(text)
            new_metas.append(meta)
            new_embs.append(emb)
            new_count += 1

        if not new_ids:
            return 0

        # Append to in-memory store
        arr = np.array(new_embs, dtype=np.float32)   # (k, dim)
        self._embeddings = (
            arr if self._embeddings is None
            else np.vstack([self._embeddings, arr])
        )
        self._ids.extend(new_ids)
        self._texts.extend(new_texts)
        self._metadatas.extend(new_metas)

        self._save()
        return new_count

    def query(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """Return the top-k most similar chunks.

        Each result dict has keys:
          id, text, metadata, score  (cosine similarity, higher = better)
        """
        if self._embeddings is None or len(self._ids) == 0:
            return []

        k = min(k, len(self._ids))

        # Cosine similarity = dot(q, d) when both are unit vectors.
        # Normalise the query and all stored embeddings.
        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms          # (N, dim)

        scores = normed @ q                        # dot product → (N,)

        # Top-k indices (unsorted first, then sort)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sort descending

        return [
            {
                "id":       self._ids[i],
                "text":     self._texts[i],
                "metadata": self._metadatas[i],
                "score":    float(scores[i]),
            }
            for i in top_idx
        ]

    def count(self) -> int:
        """Total number of stored chunks."""
        return len(self._ids)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write current state to disk."""
        # Embeddings → compressed numpy archive
        np.savez_compressed(
            self.store_dir / self.EMBEDDINGS_FILE,
            embeddings=self._embeddings,
            ids=np.array(self._ids),
        )
        # Metadata + texts → plain JSON (easy to inspect)
        payload = {"texts": self._texts, "metadatas": self._metadatas}
        (self.store_dir / self.METADATA_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Restore state from disk, if files exist."""
        emb_path  = self.store_dir / self.EMBEDDINGS_FILE
        meta_path = self.store_dir / self.METADATA_FILE

        if not emb_path.exists() or not meta_path.exists():
            return   # fresh store

        data = np.load(emb_path, allow_pickle=False)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._ids        = data["ids"].tolist()

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self._texts     = payload["texts"]
        self._metadatas = payload["metadatas"]
