"""The website. Run with:  uvicorn src.webapp:app --reload"""
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

import config
from answer_service import answer as run_answer
from retrieve import Retriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gstrag")

app = FastAPI(title="GST Answers", version="1.0")
STATIC = Path(__file__).resolve().parents[1] / "static"

_state = {"retriever": None, "llm": None, "demo": True}


@app.on_event("startup")
def load():
    """Load the indexes once when the server starts, not on every question."""
    explicit_demo = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")
    has_key = bool(config.OPENROUTER_API_KEY if config.PROVIDER == "openrouter"
                   else config.OPENAI_API_KEY)
    _state["demo"] = explicit_demo or not has_key
    _state["retriever"] = Retriever(fake=explicit_demo, use_reranker=not explicit_demo)
    if not _state["demo"]:
        from dspy_program import setup_lm
        _state["llm"] = setup_lm()
    log.info(f"ready — demo_mode={_state['demo']} provider={config.PROVIDER}")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    r = _state["retriever"]
    return {"status": "ok", "demo_mode": _state["demo"],
            "chunks_indexed": len(r.chunks) if r else 0}


@app.get("/")
def home():
    index = STATIC / "index.html"
    return FileResponse(index) if index.exists() else {"message": "open /docs"}


@app.post("/api/ask")
def ask(payload: Ask):
    q = payload.question.strip()
    if len(q) < 5:
        raise HTTPException(400, "Please type a longer question.")
    result = run_answer(q, _state["retriever"], _state["llm"])
    return {
        "question": q,
        "answer": result["answer"],
        "citations": result["citations"],
        "temporal_flag": result["temporal_flag"],
        "demo": result["demo"],
        "sources": [{
            "chunk_id": d["chunk_id"],
            "chapter": f"{d['chapter_kind'].title()} {d['chapter_num']} — {d['chapter_title']}",
            "section": " > ".join(d["section_path"]),
            "preview": d["raw_text"][:280],
            "dates": d["effective_dates"][:3],
        } for d in result["sources"]],
    }
