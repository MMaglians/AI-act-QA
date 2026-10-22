"""Lettura del testo dell'AI Act e suddivisione in chunk con metadati."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ai_act_qa.chunking import chunk_words

# Marcatori di modifica della versione consolidata, per esempio "[\u25bcM1]".
BRACKETED_MARKER = re.compile(r"\[\s*[\u25bc\u25ba\u25c4][^\]]*\]")
LOOSE_MARKER = re.compile(r"[\u25bc\u25ba]\s*[A-Z]\d*\b|\u25c4")

ARTICLE_RE = re.compile(r"^Article\s+(\d+[a-z]?)$")
CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)$")
SECTION_RE = re.compile(r"^SECTION\s+\d+$")
ANNEX_RE = re.compile(r"^ANNEX\s+([IVXLC]+)$")


@dataclass
class Chunk:
    id: str  # es. "article-5-1"
    kind: str  # "article" oppure "annex"
    ref: str  # es. "Article 5" o "Annex III"
    title: str
    chapter: str
    text: str
    part: int  # numero del pezzo dentro l'articolo, da 1
    parts: int  # quanti pezzi ha l'articolo

    def label(self) -> str:
        """Etichetta leggibile per le citazioni."""
        base = f"{self.ref} ({self.title})" if self.title else self.ref
        return f"{base}, part {self.part}/{self.parts}" if self.parts > 1 else base

    def index_text(self) -> str:
        """Testo usato dalla ricerca per parole (BM25): titolo e corpo insieme."""
        return f"{self.ref} {self.title} {self.text}"


def clean_lines(raw: str) -> list[str]:
    """Toglie marcatori di modifica, spazi strani e righe vuote."""
    lines = []
    for line in raw.replace("\xa0", " ").splitlines():
        line = BRACKETED_MARKER.sub("", line)
        line = LOOSE_MARKER.sub("", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return lines


def parse_sections(raw: str) -> list[dict]:
    """Divide il testo in articoli e allegati.

    Una riga "Article 5" apre un articolo e la riga dopo e' il titolo.
    Lo stesso vale per "ANNEX III". Le righe "CHAPTER III" aggiornano il
    capitolo corrente. Tutto cio' che viene prima del primo articolo
    (indice, considerando) viene ignorato.
    """
    lines = clean_lines(raw)
    sections: list[dict] = []
    current = None
    chapter = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if m := CHAPTER_RE.match(line):
            title = lines[i + 1] if i + 1 < len(lines) else ""
            chapter = f"Chapter {m.group(1)} - {title}".strip(" -")
            i += 2
        elif SECTION_RE.match(line):
            i += 2  # salta "SECTION 2" e il suo titolo
        elif m := ARTICLE_RE.match(line):
            current = {"kind": "article", "ref": f"Article {m.group(1)}",
                       "title": lines[i + 1] if i + 1 < len(lines) else "",
                       "chapter": chapter, "lines": []}
            sections.append(current)
            i += 2
        elif m := ANNEX_RE.match(line):
            current = {"kind": "annex", "ref": f"Annex {m.group(1)}",
                       "title": lines[i + 1] if i + 1 < len(lines) else "",
                       "chapter": "", "lines": []}
            sections.append(current)
            i += 2
        else:
            if current is not None:
                current["lines"].append(line)
            i += 1
    return _dedupe(sections)


def _dedupe(sections: list[dict]) -> list[dict]:
    """Se lo stesso articolo compare due volte (per esempio in un indice), tiene la copia piu' lunga."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for s in sections:
        words = sum(len(x.split()) for x in s["lines"])
        s["words"] = words
        key = s["ref"]
        if key not in best:
            best[key] = s
            order.append(key)
        elif words > best[key]["words"]:
            best[key] = s
    return [best[k] for k in order if best[k]["words"] > 0]


def split_lines(lines: list[str], max_words: int = 220, overlap: int = 30) -> list[str]:
    """Raggruppa righe consecutive fino a max_words; una riga troppo lunga viene spezzata."""
    pieces: list[str] = []
    buf: list[str] = []
    count = 0

    def flush():
        nonlocal buf, count
        if buf:
            pieces.append(" ".join(buf))
        buf, count = [], 0

    for line in lines:
        n = len(line.split())
        if n > max_words:
            flush()
            pieces.extend(chunk_words(line, size=max_words, overlap=overlap))
            continue
        if buf and count + n > max_words:
            flush()
        buf.append(line)
        count += n
    flush()
    return pieces


def build_chunks(raw: str, max_words: int = 220, overlap: int = 30) -> list[Chunk]:
    """mette insieme i pezzi e assegna id e numero di parte"""
    chunks: list[Chunk] = []
    for s in parse_sections(raw):
        pieces = split_lines(s["lines"], max_words=max_words, overlap=overlap)
        slug = s["ref"].lower().replace(" ", "-")
        for n, text in enumerate(pieces, start=1):
            chunks.append(Chunk(
                id=f"{slug}-{n}", kind=s["kind"], ref=s["ref"], title=s["title"],
                chapter=s["chapter"], text=text, part=n, parts=len(pieces),
            ))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(c) for c in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**d) for d in data]


def read_text_file(path: Path) -> str:
    """Legge il file salvato. encoding esplicito: su Windows il default non e' UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def corpus_stats(chunks: list[Chunk]) -> dict:
    """conta statistiche"""
    refs = {c.ref for c in chunks}
    words = [len(c.text.split()) for c in chunks]
    return {
        "chunks": len(chunks),
        "articles": len({c.ref for c in chunks if c.kind == "article"}),
        "annexes": len({c.ref for c in chunks if c.kind == "annex"}),
        "sections": len(refs),
        "min_words": min(words) if words else 0,
        "max_words": max(words) if words else 0,
        "mean_words": round(sum(words) / len(words), 1) if words else 0.0,
    }