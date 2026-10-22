"""Generazione della risposta con citazioni verificabili meccanicamente."""

import re
import time
from dataclasses import dataclass, field

from google.genai import errors, types

from ai_act_qa.corpus import Chunk

GEN_MODEL = "gemini-3.5-flash-lite"
RETRYABLE_CODES = {429, 500, 503}
NOT_FOUND = "NOT_FOUND"

SYSTEM_PROMPT = """You answer questions about the EU AI Act using ONLY the numbered excerpts provided.
Rules:
- Cite the excerpts that support each statement with their label in square brackets, for example [S1] or [S2][S3].
- Use only labels that appear in the excerpts. Never invent a label.
- If the excerpts do not contain enough information to answer, reply with exactly: NOT_FOUND
- Do not use knowledge from outside the excerpts. Do not give legal advice.
- Answer in the language of the question, concisely."""


@dataclass
class Answer:
    text: str
    found: bool
    cited: list[int]  # numeri delle fonti citate (S1 -> 1), senza duplicati
    invalid: list[int]  # citazioni a fonti che non esistono
    uncited: bool  # risposta trovata ma senza nessuna citazione
    sources: list[Chunk] = field(default_factory=list)  # S1..Sn, nell'ordine passato al modello
    prompt_tokens: int | None = None
    output_tokens: int | None = None


def build_prompt(question: str, sources: list[Chunk]) -> str:
    """numera le fonti e aggiunge la domanda"""
    blocks = [f"[S{n}] {c.label()}\n{c.text}" for n, c in enumerate(sources, start=1)]
    return "Excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def extract_citations(answer: str) -> list[int]:
    """Numeri delle fonti citate, nell'ordine di apparizione. Accetta [S1], [S1][S2] e [S1, S2]."""
    numbers: list[int] = []
    for group in re.findall(r"\[([^\]]*)\]", answer):
        for n in re.findall(r"S(\d+)", group):
            if int(n) not in numbers:
                numbers.append(int(n))
    return numbers


def generate_with_retry(client, model: str, prompt: str, max_retries: int = 5, sleep=time.sleep):
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_CODES or attempt == max_retries - 1:
                raise
            sleep(2**attempt)


def answer_question(
    question: str,
    sources: list[Chunk],
    client,
    model: str = GEN_MODEL,
    max_retries: int = 5,
    sleep=time.sleep,
) -> Answer:
    if not sources:
        return Answer("", False, [], [], False, [])
    response = generate_with_retry(client, model, build_prompt(question, sources), max_retries, sleep)
    text = (response.text or "").strip()
    usage = getattr(response, "usage_metadata", None)
    found = bool(text) and NOT_FOUND not in text
    cited_all = extract_citations(text) if found else []
    valid = [n for n in cited_all if 1 <= n <= len(sources)]
    invalid = [n for n in cited_all if n not in valid]
    return Answer(
        text=text,
        found=found,
        cited=valid,
        invalid=invalid,
        uncited=found and not cited_all,
        sources=sources,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
    )