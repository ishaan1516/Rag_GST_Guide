# GST Answers

Question answering over a 1,300-page Indian GST statutory reference. Ask in
plain English, get an answer grounded in the document with the chapter, section,
and effective date behind every claim. Each question is handled independently —
this is not a chatbot.

Built for a take-home RAG assignment: baseline → DSPy re-implementation and
optimization → hybrid retrieval, reranking, and hallucination guardrails.
Full methodology, results, and honest limitations: **REPORT.md**.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/ingest.py data/manual.docx data/chunks.jsonl
python src/index.py --fake && python src/verify.py   # 10 offline checks, no key needed
DEMO_MODE=true python -m uvicorn src.webapp:app       # http://127.0.0.1:8000
```

For real embeddings and generated answers: `pip install sentence-transformers`,
run `python src/index.py` (no `--fake`), and set `LLM_PROVIDER=openrouter` +
`OPENROUTER_API_KEY` in `.env` (OpenRouter's free tier — no OpenAI key used
anywhere in this project).

On Apple Silicon, add `FORCE_CPU=true` before any command that touches the
reranker or the DSPy optimizer — see REPORT.md §5 for why.

## Pipeline

manual.docx
└─ ingest.py 3-pass segmentation, table forward-fill, currency normalization
└─ chunks.jsonl (3,447 chunks)
├─ index.py FAISS dense + identifier-aware BM25
└─ retrieve.py RRF fusion → cross-encoder rerank → small-to-big
├─ answer_service.py grounded answering + citations (webapp)
├─ guardrails.py citation check, abstention, temporal conflict
├─ dspy_program.py optimizable rewriter + answerer
│ └─ optimize.py BootstrapFewShotWithRandomSearch
└─ evaluate.py 5-arm ablation on held-out test split


## Key decisions

- **Hybrid retrieval is a correctness requirement here, not an add-on** — the
  corpus is dense with exact statutory identifiers that embeddings alone
  confuse (`CMP-04` vs `CMP-08`).
- **RRF over weighted score fusion** — one constant, no cross-scale calibration
  to defend.
- **Temporal guardrail is metadata-driven** — the source book prints superseded
  and current provisions side by side; an answer faithful to a retrieved passage
  can still be wrong in the world if that passage was superseded. Standard
  faithfulness metrics can't see this; shared-identifier + conflicting-date
  detection can.
- **Judge model differs from the generator**, and failures degrade gracefully
  (`judge_safely()`) rather than crashing an evaluation run.
- **OpenRouter free tier throughout** — no paid API used. Tradeoffs (rate
  limits, free-model reliability, a platform-specific MPS crash) are documented
  in REPORT.md rather than hidden.

## Verified

`python src/verify.py` — 10 offline checks, no API key, no network, no
downloads required.

## Scope

Answers questions about the contents of one reference book. Not tax advice; the
book may lag current law — see the temporal guardrail note above for exactly
why that distinction matters here.