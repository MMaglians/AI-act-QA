"""BM25 a mano: ricerca per parole (retrieval sparso)."""

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in into is it its of on or "
    "that the their this to was were which with shall".split()
)

def tokenize(text: str) -> list[str]:
    """Converte tutto in minuscolo e individua solo le sequenze di lettere e numeri (ignorando punteggiatura, spazi e simboli).
    Restituisce una lista delle parole rimanenti, senza alterarle con la radice (nessuno stemming).
    """
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]

class BM25:
    """BM25 Okapi.

    score(q, d) = somma sui termini t di q di
        idf(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
    con idf(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1).
    """
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        """Prepara l'indice: per ogni parola registra in quali documenti compare e quante volte"""
        self.k1, self.b = k1, b
        tokenized = [tokenize(d) for d in docs]
        self.n = len(docs)
        self.doc_len = [len(t) for t in tokenized]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, tokens in enumerate(tokenized):
            for term, freq in Counter(tokens).items():
                self.postings[term].append((i, freq))
        self.idf = {
            term: math.log((self.n - len(p) + 0.5) / (len(p) + 0.5) + 1)
            for term, p in self.postings.items()
        }
        
    
    def scores(self, query: str) -> np.ndarray:
        """Dà un voto ad ogni documento"""
        out = np.zeros(self.n, dtype=np.float64)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, freq in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[i] / self.avgdl
                out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return out

    def top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """I k documenti migliori; i documenti con punteggio 0 non vengono restituiti."""
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(int(i), float(s[i])) for i in order if s[i] > 0]