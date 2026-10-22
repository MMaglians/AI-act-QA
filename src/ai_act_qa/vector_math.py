"""Matematica dei vettori per la ricerca semantica (solo NumPy)."""

import numpy as np

def normalize(vectors: np.ndarray) -> np.ndarray:
   """Divide ogni vettore per la sua norma L2. Accetta forma (d,) o (n, d)."""
   v = np.asarray(vectors, dtype=np.float32)
   norms = np.linalg.norm(v, axis=-1, keepdims=True) 
   if np.any(norms == 0):
       raise ValueError("vettore nullo: impossibile normalizzarlo")
   return v / norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno dell'angolo tra a e b: (a . b) / (|a| |b|)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("vettore nullo: coseno non definito")
    return float(np.dot(a, b)/(na * nb))

def top_k(query: np.ndarray, matrix: np.ndarray, k: int = 3):
    """Ricerca esaustiva (brute force): i k vettori di `matrix` piu' simili a `query`.

    Ritorna (indici, punteggi), dal piu' simile al meno simile.
    """
    q = normalize(query)
    m = normalize(matrix)
    scores = m @ q  # dopo la normalizzazione, prodotto scalare == coseno
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]