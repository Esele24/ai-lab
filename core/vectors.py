"""A vector index in ~60 lines of numpy.

The carousel says FAISS. FAISS earns its keep somewhere past a million vectors,
where an exact scan stops being instant. A 300-page PDF is a few thousand
vectors, and an exact numpy dot product over a few thousand rows is sub-
millisecond -- so FAISS here would be a dependency that buys nothing and an
approximate answer where an exact one was available.

Swap point if that ever changes: `search()` is the only function that touches the
matrix, so a real ANN index replaces one function body.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from core.docs import Chunk


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]):
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors.")
        self.chunks = chunks
        matrix = np.asarray(vectors, dtype=np.float32)
        # Normalise once at build time so search is a plain dot product instead of
        # recomputing magnitudes on every query.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = matrix / norms

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: list[float], k: int = 5) -> list[tuple[Chunk, float]]:
        query = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self.matrix @ (query / norm)
        k = min(k, len(self.chunks))
        # argpartition gets the top k without sorting everything.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.matrix)
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks]), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path) -> "VectorIndex":
        matrix = np.load(folder / "vectors.npy")
        raw = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        index = cls.__new__(cls)
        index.chunks = chunks
        index.matrix = matrix  # already normalised when it was saved
        return index
