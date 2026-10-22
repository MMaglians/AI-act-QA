"""Comandi da terminale: chunks, embed, search, ask."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from ai_act_qa.corpus import build_chunks, corpus_stats, load_chunks, read_text_file, save_chunks
from ai_act_qa.embed_texts import GeminiEmbedder
from ai_act_qa.eurlex_html import html_to_text
from ai_act_qa.generate import GEN_MODEL, answer_question
from ai_act_qa.kb import embed_chunks, make_query_embedder
from ai_act_qa.retrieval import MODES, HybridRetriever

DATA_FILE = Path("data") / "ai_act_en.html"
CACHE_DIR = Path(".cache")
CHUNKS_FILE = CACHE_DIR / "chunks.json"
DOC_CACHE = CACHE_DIR / "doc_embeddings.json"
QUERY_CACHE = CACHE_DIR / "query_embeddings.json"


def make_embedder() -> GeminiEmbedder:
    load_dotenv()
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY non trovata: controlla il file .env")
    return GeminiEmbedder(genai.Client())


def load_retriever(embedder: GeminiEmbedder) -> HybridRetriever:
    if not CHUNKS_FILE.exists():
        raise SystemExit("Mancano i chunk: lancia prima il comando 'chunks'.")
    chunks = load_chunks(CHUNKS_FILE)
    vectors = embed_chunks(chunks, embedder, DOC_CACHE)
    return HybridRetriever(chunks, vectors, make_query_embedder(embedder, QUERY_CACHE))


def cmd_chunks(args) -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"File non trovato: {DATA_FILE}. Salva li' la pagina HTML del consolidato.")
    chunks = build_chunks(html_to_text(read_text_file(DATA_FILE)), max_words=args.max_words)
    save_chunks(chunks, CHUNKS_FILE)
    print(corpus_stats(chunks))
    print("salvati in", CHUNKS_FILE)


def cmd_embed(args) -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit("Mancano i chunk: lancia prima il comando 'chunks'.")
    chunks = load_chunks(CHUNKS_FILE)
    embedder = make_embedder()
    vectors = embed_chunks(chunks, embedder, DOC_CACHE, pause=args.pause)
    print("vettori:", vectors.shape)


def cmd_search(args) -> None:
    retriever = load_retriever(make_embedder())
    for mode in MODES:
        print(f"\n== {mode} ==")
        for hit in retriever.search(args.question, k=args.k, mode=mode):
            chunk = retriever.chunks[hit.index]
            print(f"{hit.rank}. {chunk.label()}  [{hit.score:.4f}]")
            print("   " + chunk.text[:110].replace("\n", " "))

def cmd_ask(args) -> None:
    embedder = make_embedder()
    retriever = load_retriever(embedder)
    hits = retriever.search(args.question, k=args.k, mode=args.mode)
    sources = [retriever.chunks[h.index] for h in hits]
    answer = answer_question(args.question, sources, embedder.client, model=GEN_MODEL)
    if not answer.found:
        print("Non ho trovato la risposta nei passaggi recuperati.")
    else:
        print(answer.text)
        if answer.uncited:
            print("\n[ATTENZIONE: risposta senza citazioni]")
        if answer.invalid:
            print(f"\n[ATTENZIONE: citazioni a fonti inesistenti: {answer.invalid}]")
    print("\nFonti recuperate:")
    for n, chunk in enumerate(sources, start=1):
        mark = "*" if n in answer.cited else " "
        print(f" {mark} S{n}: {chunk.label()}")
    print("(* = citata nella risposta)")
    print(f"token: prompt={answer.prompt_tokens} output={answer.output_tokens}")
    
def cmd_llama(args) -> None:
    """Stessa domanda con la versione LlamaIndex (stessi chunk e stessi vettori)."""
    from ai_act_qa import llama_version

    embedder = make_embedder()
    retriever = load_retriever(embedder)
    vectors = embed_chunks(retriever.chunks, embedder, DOC_CACHE, log=lambda m: None)
    index = llama_version.build_index(retriever.chunks, vectors, retriever.embed_query)
    print("Recupero con LlamaIndex:")
    by_id = {c.id: c for c in retriever.chunks}
    for chunk_id, score in llama_version.retrieve(index, args.question, k=args.k):
        print(f"  {by_id[chunk_id].label()}  [{score:.4f}]")
    engine = llama_version.make_query_engine(index, llama_version.default_llm(), k=args.k)
    print("\nRisposta:")
    print(engine.query(args.question))

def main() -> None:
    parser = argparse.ArgumentParser(prog="ai_act_qa")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chunks", help="legge la pagina e crea i chunk (nessuna API)")
    p.add_argument("--max-words", type=int, default=220)
    p.set_defaults(func=cmd_chunks)

    p = sub.add_parser("embed", help="calcola gli embedding dei chunk (usa l'API)")
    p.add_argument("--pause", type=float, default=0.0, help="secondi di attesa dopo ogni gruppo")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("search", help="confronta dense, bm25 e ibrido")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)
    
    p = sub.add_parser("ask", help="recupera i passaggi e genera la risposta")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.set_defaults(func=cmd_ask)
    
    p = sub.add_parser("llama", help="stessa domanda con LlamaIndex")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_llama)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()