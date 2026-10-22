"""Retrieval: denso (FAISS), sparso (BM25) e ibrido (fusione RRF)."""

from dataclasses import dataclass
from typing import Callable

import faiss
import numpy as np

from ai_act_qa.bm25 import BM25
from ai_act_qa.corpus import Chunk
from ai_act_qa.vector_math import normalize

MODES = ("dense", "bm25", "hybrid")

@dataclass
class Hit:
    index: int #posizione del chunk nella lista dei chunk
    score: float
    rank: int # da 1 
 
def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Ogni lista e' una classifica di indici (il migliore per primo).
    Un documento al posto r in una classifica vale 1 / (k + r); i valori
    delle diverse classifiche si sommano. Servono solo le POSIZIONI, non i punteggi.
    """   
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
            
def build_flat_ip_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Indice esatto: prodotto scalare su vettori normalizzati == coseno."""
    data = np.ascontiguousarray(normalize(vectors), dtype=np.float32)
    index = faiss.IndexFlatIP(data.shape[1])
    index.add(data)
    return index


class HybridRetriever:
    def __init__(
        self, 
        chunks: list[Chunk],
        vectors: np.ndarray,
        embed_query: Callable[[str], np.ndarray],
        pool: int=30,
        rrf_k: int=60,
    ):
        if len(chunks) != len(vectors):
            raise ValueError("chunks e vettori devono avere la stessa lunghezza")
        self.chunks = chunks
        self.index = build_flat_ip_index(vectors)
        self.bm25 = BM25([c.index_text() for c in chunks])
        self.embed_query = embed_query
        self.pool = pool
        self.rrf_k = rrf_k
    
    def dense(self, query: str, n: int) -> list[tuple[int, float]]:
        #ricerca per significato
        q = np.ascontiguousarray(normalize(self.embed_query(query)).reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(q, min(n, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def sparse(self, query: str, n: int) -> list[tuple[int, float]]:
        #ricerca per parole
        return self.bm25.top_k(query, n)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        #sceglie la modalità
        if mode not in MODES:
            raise ValueError(f"mode deve essere uno di {MODES}")
        if mode == "dense":
            pairs = self.dense(query, k)
        elif mode == "bm25":
            pairs = self.sparse(query, k)
        else:
            dense_rank = [i for i, _ in self.dense(query, self.pool)]
            sparse_rank = [i for i, _ in self.sparse(query, self.pool)]
            pairs = rrf([dense_rank, sparse_rank], k=self.rrf_k)[:k]
        return [Hit(index=i, score=s, rank=r) for r, (i, s) in enumerate(pairs[:k], start=1)]