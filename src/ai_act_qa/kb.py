"""Knowledge base: embedding dei chunk (a gruppi) e della domanda."""

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ai_act_qa.corpus import Chunk
from ai_act_qa.embed_texts import (
    GeminiEmbedder,
    cache_key,
    embed_cached,
    format_document,
    format_query,
)


def doc_text(chunk: Chunk) -> str:
    """Testo del chunk nel formato 'documento' di gemini-embedding-2."""
    title = f"{chunk.ref} - {chunk.title}" if chunk.title else chunk.ref
    return format_document(chunk.text, title)


def all_cached(texts: list[str], embedder: GeminiEmbedder, cache_path: Path) -> bool:
    """True se tutti i testi hanno gia' un vettore nella cache su disco."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    return all(cache_key(embedder.model, embedder.dim, t) in cache for t in texts)


def embed_chunks(
    chunks: list[Chunk],
    embedder: GeminiEmbedder,
    cache_path: Path,
    group: int = 48,
    pause: float = 0.0,
    log: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Embedding di tutti i chunk, un gruppo alla volta.

    La cache viene salvata a ogni gruppo: se una quota si esaurisce a meta',
    al lancio successivo si riparte dal punto in cui ci si era fermati.
    `pause` e' l'attesa (in secondi) dopo ogni gruppo che ha chiamato l'API.
    """
    texts = [doc_text(c) for c in chunks]
    parts = []
    for start in range(0, len(texts), group):
        batch = texts[start : start + group]
        cached = all_cached(batch, embedder, cache_path)
        parts.append(embed_cached(batch, embedder, cache_path))
        log(f"  embedding {min(start + group, len(texts))}/{len(texts)}")
        if pause and not cached and start + group < len(texts):
            sleep(pause)
    if not parts:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    return np.vstack(parts)


def make_query_embedder(embedder: GeminiEmbedder, cache_path: Path) -> Callable[[str], np.ndarray]:
    """Funzione domanda -> vettore (formato 'domanda'), con cache su disco."""

    def embed_query(query: str) -> np.ndarray:
        return embed_cached([format_query(query)], embedder, cache_path)[0]

    return embed_query