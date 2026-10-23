"""Valutazione del retrieval su domande con risposta nota: hit@k, recall@k, intervalli di Wilson."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ai_act_qa.corpus import Chunk
from ai_act_qa.retrieval import MODES
from ai_act_qa.retrieval_metrics import hit_at_k, recall_at_k

KS = (1, 3, 5, 10)
LANGS = ("en", "it")
LEVELS = ("easy", "medium", "hard")

@dataclass
class Question:
    id: str
    lang: str
    difficulty: str
    question: str
    gold: list[str] #riferimenti attesi
    answer: str #risposte di riferimento, hand-written e controllata
    evidence: list[str] #frasi che DEVONO comparire nel testo dei riferimenti

def load_questions(path: Path) -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(**item) for item in data]

def normalize_text(text: str) -> str:
    """Minuscolo, apostrofi tipografici resi semplici, spazi compattati."""
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\xa0", " ")
    return " ".join(text.lower().split())

def validate_questions(questions: list[Question], chunks: list[Chunk]) -> list[str]:
    """Elenco dei problemi trovati (vuoto = tutto a posto);
    controla che ogni riferimento esista.
    """
    problems: list[str] = []
    text_by_ref: dict[str, str] = {}
    for c in chunks:
        text_by_ref[c.ref] = text_by_ref.get(c.ref, "") + " " + c.text
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: id duplicato")
        seen.add(q.id)
        if q.lang not in LANGS:
            problems.append(f"{q.id}: lingua '{q.lang}' non valida")
        if q.difficulty not in LEVELS:
            problems.append(f"{q.id}: difficolta' '{q.difficulty}' non valida")
        if not q.question.strip() or not q.answer.strip():
            problems.append(f"{q.id}: domanda o risposta vuota")
        if not q.gold:
            problems.append(f"{q.id}: nessun riferimento atteso")
        missing = [g for g in q.gold if g not in text_by_ref]
        for g in missing:
            problems.append(f"{q.id}: il riferimento '{g}' non esiste nel corpus")
        if missing or not q.evidence:
            if not q.evidence:
                problems.append(f"{q.id}: manca l'evidenza")
            continue
        joined = normalize_text(" ".join(text_by_ref[g] for g in q.gold))
        for phrase in q.evidence:
            if normalize_text(phrase) not in joined:
                problems.append(f"{q.id}: evidenza non trovata in {q.gold}: '{phrase}'")
    return problems

def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    #Intervallo di Wilson al 95%
    if n==0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = z * math.sqrt(p*(1-p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

def run_retrieval(retriever, questions: list[Question], modes=MODES, max_k: int = max(KS)) -> dict:
    """Per ogni modalita' e domanda: la lista dei riferimenti dei primi max_k chunk, in ordine."""
    runs: dict[str, dict[str, list[str]]] = {}
    for mode in modes:
        runs[mode] = {}
        for q in questions:
            hits = retriever.search(q.question, k=max_k, mode=mode)
            runs[mode][q.id] = [retriever.chunks[h.index].ref for h in hits]
    return runs


def summarize(runs: dict, questions: list[Question], ks=KS) -> dict:
    """mode -> k -> {n, hits, hit, ci, recall}. k conta CHUNK, non articoli distinti."""
    out: dict[str, dict[int, dict]] = {}
    for mode, per_question in runs.items():
        out[mode] = {}
        for k in ks:
            hits = 0
            recalls = []
            for q in questions:
                ranked = per_question[q.id]
                gold = set(q.gold)
                hits += int(hit_at_k(ranked, gold, k))
                recalls.append(recall_at_k(ranked, gold, k))
            n = len(questions)
            out[mode][k] = {
                "n": n,
                "hits": hits,
                "hit": hits / n if n else 0.0,
                "ci": wilson(hits, n),
                "recall": sum(recalls) / n if n else 0.0,
            }
    return out

def subset(questions: list[Question], **filters) -> list[Question]:
    return [q for q in questions if all(getattr(q, key) == value for key, value in filters.items())]

def first_gold_rank(ranked: list[str], gold: list[str]) -> int | None:
    """Posizione (da 1) del primo chunk che appartiene a un riferimento atteso; None se assente."""
    for position, ref in enumerate(ranked, start=1):
        if ref in gold:
            return position
    return None

def compare(runs: dict, questions: list[Question], mode_a: str, mode_b: str, k: int) -> tuple[list[str], list[str]]:
    """Domande trovate solo da mode_a e solo da mode_b entro i primi k chunk."""
    only_a, only_b = [], []
    for q in questions:
        a = hit_at_k(runs[mode_a][q.id], set(q.gold), k)
        b = hit_at_k(runs[mode_b][q.id], set(q.gold), k)
        if a and not b:
            only_a.append(q.id)
        elif b and not a:
            only_b.append(q.id)
    return only_a, only_b

def format_table(summary: dict, title: str, ks=KS) -> str:
    """Due blocchi stretti: hit@k con intervallo di Wilson, poi recall@k."""
    lines = [title, "hit@k (intervallo di Wilson al 95%)"]
    lines.append(f"{'':<8}" + "".join(f"{'k=' + str(k):<19}" for k in ks))
    for mode, per_k in summary.items():
        cells = ""
        for k in ks:
            s = per_k[k]
            lo, hi = s["ci"]
            cells += f"{s['hit']:.2f} ({lo:.2f}-{hi:.2f})".ljust(19)
        lines.append(f"{mode:<8}" + cells)
    lines.append("recall@k")
    for mode, per_k in summary.items():
        lines.append(f"{mode:<8}" + "".join(f"{per_k[k]['recall']:<19.2f}" for k in ks))
    return "\n".join(lines)

def save_results(path: Path, summary: dict, runs: dict, questions: list[Question]) -> None:
    payload = {
        "n_questions": len(questions),
        "summary": {m: {str(k): v for k, v in per_k.items()} for m, per_k in summary.items()},
        "first_gold_rank": {
            mode: {q.id: first_gold_rank(runs[mode][q.id], q.gold) for q in questions} for mode in runs
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")