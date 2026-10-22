"""Metriche di retrieval: hit@k e recall@k."""

def hit_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    """1.0 se almeno un documento rilevante e' tra i primi k risultati, altrimenti 0.0."""
    return 1.0 if relevant & set(ranked_ids[:k]) else 0.0

def recall_at_k(ranked_ids: list, relevant: set, k: int) -> float:
    """Frazione dei documenti rilevanti trovati tra i primi k risultati."""
    if not relevant:
        raise ValueError("serve almeno un documento rilevante")
    return len(relevant & set(ranked_ids[:k])) / len(relevant)

def mean(values: list) -> float:
    return sum(values) / len(values)