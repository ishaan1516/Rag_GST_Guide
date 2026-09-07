"""DSPy signatures and the optimizable QA module."""
import dspy

import config
from retrieve import Retriever, format_context


def setup_lm(model=None, api_key=None):
    """Provider is picked by config.PROVIDER (env var LLM_PROVIDER).
    OpenRouter is OpenAI-compatible and reached through litellm's native
    "openrouter/" prefix - no api_base needed. Free-tier calls are rate
    limited (20/min, 50-1000/day depending on account credit); see
    BUILD_GUIDE.md Stage 3b before running optimize.py or evaluate.py
    on this path, since those make far more calls than a single question."""
    if config.PROVIDER == "openrouter":
        lm = dspy.LM(model or config.OPENROUTER_GEN_MODEL,
                     api_key=api_key or config.OPENROUTER_API_KEY,
                     temperature=0.0, max_tokens=3500)
    else:
        lm = dspy.LM(model or config.GEN_MODEL,
                     api_key=api_key or config.OPENAI_API_KEY,
                     temperature=0.0, max_tokens=3500)
    dspy.configure(lm=lm)
    return lm


class RewriteQuery(dspy.Signature):
    """Rewrite a user question into a search query using Indian GST statutory
    vocabulary - section numbers, form names, and CGST Act terminology - so it
    matches the phrasing used in a legal reference text."""

    question: str = dspy.InputField()
    search_query: str = dspy.OutputField()


class AnswerFromContext(dspy.Signature):
    """Answer strictly from the provided GST reference excerpts. Cite the
    chunk_id supporting each claim. If the excerpts do not contain the answer,
    set sufficient to False and say so rather than inferring."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="numbered excerpts with chapter and section")
    answer: str = dspy.OutputField()
    citations: list[str] = dspy.OutputField(desc="chunk_ids actually used")
    sufficient: bool = dspy.OutputField(desc="whether the context answered the question")


class GSTQA(dspy.Module):
    """Both the rewriter and the answerer are optimizable. The retriever is a
    frozen tool - DSPy tunes the text going into it, not its parameters."""

    def __init__(self, retriever, use_rewrite=True, mode="hybrid", rerank=True, k=5):
        super().__init__()
        self.retriever, self.use_rewrite = retriever, use_rewrite
        self.mode, self.rerank, self.k = mode, rerank, k
        self.rewrite = dspy.ChainOfThought(RewriteQuery)
        self.answer = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question, **kw):
        query = self.rewrite(question=question).search_query if self.use_rewrite else question
        docs = self.retriever.search(query, mode=self.mode, rerank=self.rerank, k=self.k)
        ctx = format_context(docs)
        out = self.answer(question=question, context=ctx)
        return dspy.Prediction(answer=out.answer, citations=out.citations or [],
                               sufficient=out.sufficient, context=ctx,
                               retrieved=docs, search_query=query)