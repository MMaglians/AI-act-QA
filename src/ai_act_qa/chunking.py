"""Chunking: spezza un testo in pezzi di dimensione controllata, con sovrapposizione."""

def chunk_words(text: str, size: int=120, overlap: int=20) -> list[str]:
    """Divide `text` in blocchi di al massimo `size` parole e restuisce la lista dei blocchi di testo.

    Blocchi consecutivi condividono `overlap` parole, cosi' una frase
    tagliata a meta' compare intera in almeno uno dei due blocchi.
    """
    if size <= 0:
        raise ValueError("size deve essere > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap deve essere >= 0 e < size")
    words = text.split()
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks