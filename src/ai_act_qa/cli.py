"""Comandi da terminale: chunks, embed, search, ask, llama, show, eval."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from ai_act_qa.corpus import build_chunks, corpus_stats, load_chunks, read_text_file, save_chunks
from ai_act_qa.embed_texts import GeminiEmbedder, embed_cached, format_query
from ai_act_qa.eurlex_html import html_to_text
from ai_act_qa.evaluation import (
    LANGS,
    LEVELS,
    compare,
    first_gold_rank,
    format_table,
    load_questions,
    run_retrieval,
    save_results,
    subset,
    summarize,
    validate_questions,
)
from ai_act_qa.generate import GEN_MODEL, answer_question
from ai_act_qa.kb import embed_chunks, make_query_embedder
from ai_act_qa.retrieval import MODES, HybridRetriever

DATA_FILE = Path("data") / "ai_act_en.html"
CACHE_DIR = Path(".cache")
CHUNKS_FILE = CACHE_DIR / "chunks.json"
DOC_CACHE = CACHE_DIR / "doc_embeddings.json"
QUERY_CACHE = CACHE_DIR / "query_embeddings.json"

QUESTIONS_FILE = Path("eval") / "questions.json"
RESULTS_FILE = Path("eval") / "results.json"


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
    
def cmd_show(args) -> None:
    """Stampa il testo di un articolo o allegato, per controllare le risposte di riferimento."""
    if not CHUNKS_FILE.exists():
        raise SystemExit("Mancano i chunk: lancia prima il comando 'chunks'.")
    found = [c for c in load_chunks(CHUNKS_FILE) if c.ref.lower() == args.ref.lower()]
    if not found:
        raise SystemExit(f"Riferimento non trovato: {args.ref}")
    print(f"{found[0].ref} - {found[0].title}")
    for chunk in found:
        print(f"\n[{chunk.part}/{chunk.parts}] {chunk.text}")


def cmd_eval(args) -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit("Mancano i chunk: lancia prima il comando 'chunks'.")
    chunks = load_chunks(CHUNKS_FILE)
    questions = load_questions(QUESTIONS_FILE)
    problems = validate_questions(questions, chunks)
    for problem in problems:
        print("PROBLEMA:", problem)
    if problems:
        raise SystemExit(f"{len(problems)} problemi nelle domande: correggi e rilancia.")
    print(f"{len(questions)} domande valide.")
    if args.check:
        return

    embedder = make_embedder()
    # embedding di tutte le domande in poche chiamate, invece di una per domanda
    embed_cached([format_query(q.question) for q in questions], embedder, QUERY_CACHE)
    retriever = load_retriever(embedder)
    runs = run_retrieval(retriever, questions)
    summary = summarize(runs, questions)

    print("\n" + format_table(summary, "TUTTE LE DOMANDE (n=%d)" % len(questions)))
    for lang in LANGS:
        part = subset(questions, lang=lang)
        print("\n" + format_table(summarize(runs, part), f"LINGUA {lang} (n={len(part)})"))
    for level in LEVELS:
        part = subset(questions, difficulty=level)
        print("\n" + format_table(summarize(runs, part), f"DIFFICOLTA {level} (n={len(part)})"))

    only_hybrid, only_dense = compare(runs, questions, "hybrid", "dense", 5)
    print(f"\nA k=5: trovate solo da hybrid {only_hybrid}, solo da dense {only_dense}")
    print("\nDomande NON trovate entro k=5 con hybrid:")
    for q in questions:
        rank = first_gold_rank(runs["hybrid"][q.id], q.gold)
        if rank is None or rank > 5:
            print(f"  {q.id} [{q.lang}, {q.difficulty}] attesi {q.gold}, primo posto utile: {rank}")
    save_results(RESULTS_FILE, summary, runs, questions)
    print("risultati salvati in", RESULTS_FILE)
    

def cmd_why(args) -> None:
    """Per una domanda di valutazione: i primi risultati di ogni modalita', con * sui riferimenti attesi."""
    questions = {q.id: q for q in load_questions(QUESTIONS_FILE)}
    if args.id not in questions:
        raise SystemExit(f"Domanda non trovata: {args.id}")
    q = questions[args.id]
    retriever = load_retriever(make_embedder())
    print(f"{q.id} [{q.lang}, {q.difficulty}] {q.question}")
    print(f"attesi: {q.gold}")
    for mode in MODES:
        print(f"\n== {mode} ==")
        for hit in retriever.search(q.question, k=args.k, mode=mode):
            chunk = retriever.chunks[hit.index]
            mark = "*" if chunk.ref in q.gold else " "
            print(f" {mark} {hit.rank}. {chunk.label()[:90]}  [{hit.score:.4f}]")

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
    
    p = sub.add_parser("show", help="stampa il testo di un articolo o allegato")
    p.add_argument("ref", help='per esempio "Article 5" oppure "Annex III"')
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("eval", help="valuta il retrieval sulle domande di eval/questions.json")
    p.add_argument("--check", action="store_true", help="controlla solo le domande (nessuna API)")
    p.set_defaults(func=cmd_eval)
    
    p = sub.add_parser("why", help="mostra i risultati di una domanda di valutazione")
    p.add_argument("id", help="per esempio q10")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_why)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()