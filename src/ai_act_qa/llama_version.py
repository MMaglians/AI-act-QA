"""Lo stesso RAG con LlamaIndex: stessi chunk, stessi vettori, indice FAISS gestito dal framework."""

import os
from typing import Callable

import faiss
import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import PrivateAttr

from ai_act_qa.corpus import Chunk
from ai_act_qa.generate import GEN_MODEL
from ai_act_qa.vector_math import normalize

QA_TEMPLATE = PromptTemplate(
    "You answer questions about the EU AI Act using ONLY the excerpts below.\n"
    "Each excerpt starts with its label. Cite the labels you use in square brackets.\n"
    "If the excerpts are not enough, reply exactly: NOT_FOUND\n"
    "Answer in the language of the question, concisely.\n\n"
    "{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer: "
)


class QueryEmbedding(BaseEmbedding):
    """Collega la nostra funzione 'domanda -> vettore' al framework.

    I vettori dei chunk sono GIA' calcolati: il framework non deve ricalcolarli.
    lancia errore se l'embedding dei chunck viene chiesto.
    """

    _embed_query: Callable[[str], np.ndarray] = PrivateAttr()

    def __init__(self, embed_query: Callable[[str], np.ndarray], **kwargs):
        super().__init__(model_name="gemini-embedding-2", **kwargs)
        self._embed_query = embed_query

    def _get_query_embedding(self, query: str) -> list[float]:
        return normalize(self._embed_query(query)).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        raise RuntimeError("i vettori dei chunk sono gia' calcolati: non dovrei essere chiamato")


def build_index(
    chunks: list[Chunk], vectors: np.ndarray, embed_query: Callable[[str], np.ndarray]
) -> VectorStoreIndex:
    """mette ogni chunk in un nodo con il suo vettore, dentro un indice FAISS esatto gestito da LlamaIndex."""
    unit = normalize(vectors)
    nodes = [
        TextNode(
            id_=chunk.id,
            text=chunk.text,
            metadata={"label": chunk.label()},
            embedding=vec.tolist(),
        )
        for chunk, vec in zip(chunks, unit)
    ]
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(unit.shape[1]))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        nodes, storage_context=storage_context, embed_model=QueryEmbedding(embed_query)
    )


def retrieve(index: VectorStoreIndex, question: str, k: int = 5) -> list[tuple[str, float]]:
    """Coppie (id del chunk, punteggio), dal migliore."""
    found = index.as_retriever(similarity_top_k=k).retrieve(question)
    return [(item.node.node_id, float(item.score)) for item in found]


def default_llm():
    """LLM Gemini di LlamaIndex. Chiave letta da GEMINI_API_KEY (stesso .env del resto)."""
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(model=GEN_MODEL, api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0)


def make_query_engine(index: VectorStoreIndex, llm, k: int = 5):
    #collega retrieval e modello  con nostro prompt
    return index.as_query_engine(llm=llm, similarity_top_k=k, text_qa_template=QA_TEMPLATE)