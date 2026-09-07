"""Turn retrieved passages into an answer.

Demo mode returns the passages themselves with their citations - no API key,
no cost, and still genuinely useful for checking whether retrieval works.
Real mode sends them to an LLM with strict grounding instructions.
"""
import json
import re

import config
from guardrails import temporal_conflict
from retrieve import format_context

ANSWER_PROMPT = """Answer the question using ONLY the GST reference excerpts below.

Rules:
- Cite the [chunk_id] for every claim you make.
- If the excerpts do not contain the answer, say "The document does not cover this."
  Do not use outside knowledge.
- Quote exact section numbers, form names and amounts as they appear.
- Keep it under 150 words.

Excerpts:
{context}

Question: {question}

Answer:"""

TEMPORAL_PROMPT = """The excerpts contain more than one version of the same rule,
from different dates. State the position that currently applies, name its effective
date, and say what it replaced.

Excerpts:
{context}

Question: {question}

Answer:"""


def _demo_answer(question, docs):
    """No LLM. Show the best passage and its source - proves retrieval works."""
    if not docs:
        return "The document does not cover this.", [], False
    top = docs[0]
    sentences = re.split(r"(?<=[.;])\s+", top["raw_text"])
    snippet = " ".join(sentences[:3])[:600]
    path = " > ".join(top["section_path"]) if top["section_path"] else ""
    body = (f"[demo mode - no AI key, showing the retrieved passage]\n\n"
            f"{snippet}\n\n"
            f"Source: {top['chapter_kind'].title()} {top['chapter_num']} — "
            f"{top['chapter_title']}" + (f" > {path}" if path else ""))
    return body, [top["chunk_id"]], False


def _extract_text(raw):
    """dspy.LM's legacy call path returns list[str | dict] - not always a
    plain string. Some providers reply with a structured/dict-shaped message
    (e.g. tool-call formatting) instead of plain text, depending on which
    underlying free model actually served the request. Handle every shape
    we might realistically get instead of assuming the friendly one."""
    item = raw[0] if isinstance(raw, list) and raw else raw
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        reasoning = (item.get("reasoning_content") or "").strip()
        for key in ("content", "text"):
            val = (item.get(key) or "").strip()
            if not val:
                continue
            # a reasoning model cut off before separating "thinking" from
            # "answering" duplicates the same unfinished text into both
            # fields - trusting `text` blindly here would surface raw
            # chain-of-thought as if it were the real answer
            if reasoning and (val == reasoning or val in reasoning
                              or reasoning in val):
                continue
            return val
        if item.get("tool_calls"):
            return str(item["tool_calls"])
        if reasoning:
            return ("The model ran out of space while thinking through this "
                    "question and never wrote a final answer. Try asking "
                    "again - a different free model may answer directly, or "
                    "the same one may finish this time.")
        return json.dumps(item)  # true last resort - visible, not a crash
    return str(item)


def answer(question, retriever, llm=None, k=5):
    docs = retriever.search(question, mode="hybrid",
                            rerank=retriever.reranker is not None, k=k)
    flagged = temporal_conflict(docs)

    if llm is None:
        text, cites, _ = _demo_answer(question, docs)
        return {"answer": text, "citations": cites,
                "temporal_flag": flagged, "sources": docs, "demo": True}

    prompt = (TEMPORAL_PROMPT if flagged else ANSWER_PROMPT).format(
        context=format_context(docs), question=question)
    raw = llm(prompt)
    text = _extract_text(raw)

    valid = {d["chunk_id"] for d in docs}
    cites = [c for c in re.findall(r"\[([a-z0-9_]+)\]", text) if c in valid]
    return {"answer": text, "citations": cites, "temporal_flag": flagged,
            "sources": docs, "demo": False}