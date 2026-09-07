"""Vanilla RAG: dense-only retrieval, k=5, one flat prompt, no rewrite.

Deliberately unimproved. If the baseline quietly gets better while you build
the good version, the ablation delta disappears and the report loses its spine.
"""
import dspy

from answer_service import _extract_text
from retrieve import format_context

VANILLA_PROMPT = """Answer the question using the context below.

Context:
{context}

Question: {question}
Answer:"""


class VanillaRAG(dspy.Module):
    def __init__(self, retriever, k=5):
        super().__init__()
        self.retriever, self.k = retriever, k

    def forward(self, question, **kw):
        docs = self.retriever.search(question, mode="dense", rerank=False,
                                     expand=False, k=self.k)
        ctx = format_context(docs)
        raw = dspy.settings.lm(VANILLA_PROMPT.format(context=ctx, question=question))
        answer = _extract_text(raw)  # same shape-handling as answer_service.py
        return dspy.Prediction(answer=answer, citations=[], sufficient=True,
                               context=ctx, retrieved=docs, search_query=question)