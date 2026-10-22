"""Embedding di testi con l'API Gemini, con retry e cache su disco."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from google.genai import errors, types

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM = 768
RETRYABLE_CODES = {429, 500, 503}

#Google raccomanda domanda e documento con formati diversi
def format_document(text: str, title: str | None = None) -> str:
    """Formato documento per gemini-embedding-2 (ricerca asimmetrica)."""
    return f"title: {title or 'none'} | text: {text}"


def format_query(text: str) -> str:
    """Formato domanda per gemini-embedding-2 (question answering)."""
    return f"task: question answering | query: {text}"

class GeminiEmbedder: 
    def __init__(
        self,
        client,
        model: str=EMBED_MODEL,
        dim: int=EMBED_DIM,
        batch_size: int=16,#i testi vengono spezzati in blocchi da 16
        max_retries: int=5,
        sleep=time.sleep,
    ):
        self.client = client
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sleep = sleep
    
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        # Un Content per testo: se passassimo una lista di stringhe,
        # gemini-embedding-2 restituirebbe UN SOLO vettore aggregato.
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        config = types.EmbedContentConfig(output_dimensionality=self.dim)#mappo testo in R^dim
        for attempt in range(self.max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model, contents=contents, config=config
                )
                break
            except errors.APIError as exc:
                last_try = attempt == self.max_retries-1
                if exc.code not in RETRYABLE_CODES or last_try:
                    raise
                self.sleep(2**attempt) #1, 2, 4, 8 secondi
        vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
        if vectors.shape != (len(texts), self.dim):
            raise ValueError(
                f"forma inattesa {vectors.shape}, attesa {(len(texts), self.dim)}"
            )
        return vectors
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embedding di una lista di testi, a blocchi di `batch_size`."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [
            self._embed_batch(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(parts)

def cache_key(model: str, dim: int, text: str) -> str:
    """Chiave hash usata da embed_cached(vedi sotto) per salvare i vettori in .cache/"""
    return hashlib.sha256(f"{model}|{dim}|{text}".encode("utf-8")).hexdigest()

def embed_cached(texts: list[str], embedder: GeminiEmbedder, cache_path: Path) -> np.ndarray:
    """Come embedder.embed, ma riusa i vettori gia' calcolati (salvati su disco)."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    keys = [cache_key(embedder.model, embedder.dim, t) for t in texts]
    missing: dict[str, str] = {}
    for key, text in zip(keys, texts):
        if key not in cache:
            missing[key] = text  # il dizionario elimina i duplicati
    if missing:
        vectors = embedder.embed(list(missing.values()))
        for key, vec in zip(missing, vectors):
            cache[key] = vec.tolist()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return np.array([cache[k] for k in keys], dtype=np.float32)   