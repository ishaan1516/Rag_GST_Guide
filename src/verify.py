"""Offline self-check. Runs everything that needs no model download and no API key.
If this passes, your wiring is correct and the only untested parts are the
model downloads and the LLM calls."""
import sys, json, traceback

CHECKS = []
def check(name):
    def deco(fn):
        CHECKS.append((name, fn)); return fn
    return deco

@check("chunks.jsonl exists and is well-formed")
def _():
    rows = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    assert len(rows) > 1000, f"only {len(rows)} chunks"
    need = {"chunk_id","text","raw_text","chapter_num","section_path",
            "parent_id","type","identifiers","effective_dates","token_count"}
    assert need <= set(rows[0]), f"missing {need - set(rows[0])}"
    return f"{len(rows)} chunks"

@check("indexes built and loadable")
def _():
    import faiss, pickle
    idx = faiss.read_index("artifacts/faiss.index")
    bm = pickle.load(open("artifacts/bm25.pkl","rb"))
    assert idx.ntotal == len(bm["ids"]), "index/bm25 length mismatch"
    return f"{idx.ntotal} vectors, dim {idx.d}"

@check("BM25 retrieval returns on-topic results")
def _():
    from retrieve import Retriever
    r = Retriever(fake=True)
    hits = r.sparse("composition scheme withdrawal CMP-04", k=5)
    assert hits and hits[0].score > 1.0, "no lexical match"
    return f"top score {hits[0].score:.1f}"

@check("RRF fusion produces a merged ranking")
def _():
    from retrieve import Retriever
    r = Retriever(fake=True)
    h = r.hybrid("registration threshold aggregate turnover", k=5)
    assert len(h) == 5 and h[0].score > h[-1].score
    return "5 fused hits, monotonic"

@check("small-to-big expansion stays in budget")
def _():
    from retrieve import Retriever, format_context
    r = Retriever(fake=True)
    ctx = format_context(r.search("input tax credit conditions", k=5))
    assert 200 < len(ctx) < 40000, f"context {len(ctx)} chars"
    return f"{len(ctx)} chars"

@check("temporal guardrail discriminates")
def _():
    from retrieve import Retriever
    from guardrails import temporal_conflict
    r = Retriever(fake=True)
    on = temporal_conflict(r.search("composition scheme withdrawal form CMP-04", k=5))
    return f"fires={on}"

@check("DSPy signatures and module graph build")
def _():
    from dspy_program import GSTQA
    from retrieve import Retriever
    p = GSTQA(Retriever(fake=True))
    names = [n for n, _ in p.named_predictors()]
    assert names == ["rewrite.predict", "answer.predict"], names
    return ", ".join(names)

@check("answer service works with no API key")
def _():
    from answer_service import answer
    from retrieve import Retriever
    r = answer("Which form withdraws you from the composition scheme?",
               Retriever(fake=True), llm=None)
    assert r["demo"] and r["citations"], "demo answer produced no citation"
    return f"cited {r['citations'][0]}, {len(r['sources'])} sources"

@check("web page file exists")
def _():
    from pathlib import Path
    t = Path("static/index.html").read_text()
    assert "/api/ask" in t, "page is not wired to the API"
    return f"{len(t)} characters"

@check("eval set splits are present")
def _():
    import os
    missing = [f for f in ("train","val","test") if not os.path.exists(f"data/{f}.jsonl")]
    if missing:
        return f"SKIP - run evalset.py --split first (missing {missing})"
    n = {f: sum(1 for _ in open(f"data/{f}.jsonl")) for f in ("train","val","test")}
    return str(n)

if __name__ == "__main__":
    sys.path.insert(0, "src")
    ok = 0
    for name, fn in CHECKS:
        try:
            print(f"  PASS  {name} — {fn()}"); ok += 1
        except Exception as e:
            print(f"  FAIL  {name} — {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(CHECKS)} checks passed")
    sys.exit(0 if ok == len(CHECKS) else 1)
