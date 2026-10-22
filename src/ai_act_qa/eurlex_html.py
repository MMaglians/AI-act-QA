"""Dal file HTML di EUR-Lex al testo semplice che legge corpus.py."""

import re

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK_TAGS = {"p", "div", "table", "ul", "ol", "li", "tr"}
EMPTY_PARENS = re.compile(r"[ \t]*\(\s*\)")


def _has_block_child(el: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in el.children)


def _text(el) -> str:
    return " ".join(el.get_text().split())


def _table_lines(table: Tag) -> list[str]:
    lines = []
    for row in table.find_all("tr"):
        cells = [_text(td) for td in row.find_all("td")]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _walk(el: Tag) -> list[str]:
    """Trasforma un blocco HTML in righe di testo, nell'ordine del documento."""
    classes = el.get("class") or []
    if el.name == "table":
        return _table_lines(el)

    # elenco "(a) testo": la colonna 1 e' il numero, la colonna 2 il contenuto
    if "grid-container" in classes:
        cols = [c for c in el.find_all("div", recursive=False)]
        if len(cols) == 2:
            prefix = _text(cols[0])
            body = _walk(cols[1]) if _has_block_child(cols[1]) else [_text(cols[1])]
            body = [b for b in body if b]
            if body:
                body[0] = f"{prefix} {body[0]}"
            return body

    # paragrafo numerato: <span class="no-parag">1.</span> + contenuto
    first = next((c for c in el.children if isinstance(c, Tag)), None)
    if first is not None and "no-parag" in (first.get("class") or []):
        prefix = _text(first)
        first.decompose()
        body = _walk(el) if _has_block_child(el) else [_text(el)]
        body = [b for b in body if b]
        if body:
            body[0] = f"{prefix} {body[0]}"
        return body

    if not _has_block_child(el):
        text = _text(el)
        return [text] if text else []

    lines: list[str] = []
    inline: list[str] = []

    def flush():
        text = " ".join(" ".join(inline).split())
        if text:
            lines.append(text)
        inline.clear()

    for child in el.children:
        if isinstance(child, NavigableString):
            inline.append(str(child))
        elif child.name in BLOCK_TAGS:
            flush()
            lines.extend(_walk(child))
        else:
            inline.append(child.get_text())
    flush()
    return lines


def html_to_text(html: str) -> str:
    """Estrae capitoli, articoli e allegati dal consolidato EUR-Lex (HTML salvato dal browser)."""
    soup = BeautifulSoup(html, "html.parser")

    # rumore: marcatori di modifica, frecce di navigazione, note a pie' di pagina, script
    for sel in ("script", "style", "hr", "p.modref", "p.arrow", "a.anchorarrow", "p.footnote"):
        for el in soup.select(sel):
            el.decompose()
    for br in soup.select("br"):
        br.replace_with(" ")
    for sup in soup.select("span.superscript"):
        link = sup.find_parent("a")
        (link or sup).decompose()

    body = soup.find(id="enc_1")
    annexes = [d for d in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$"))]
    if body is None or not annexes:
        raise ValueError("Struttura non riconosciuta: manca il corpo del testo o gli allegati.")

    lines: list[str] = []
    for root in [body, *annexes]:
        lines.extend(_walk(root))
    text = "\n".join(lines)
    text = EMPTY_PARENS.sub("", text)
    # un apostrofo spaiato in fondo a un titolo (es. "Subject matter'") e' un difetto della fonte
    return re.sub(r"(?m)^(.*[A-Za-z])'$", r"\1", text)