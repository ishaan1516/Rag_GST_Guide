# GST RAG System — Report

## 1. Document

`66__GST_Smart_Guide.docx` — a ~1,300-page Indian GST statutory reference, not a
product manual as the assignment brief's generic wording implied. This mattered:
54 chapters + 16 appendices, 103 tables, heavy use of exact statutory identifiers
(1,692 `Section N` references, 1,447 form names, 567 notification numbers), and —
critically — provisions that changed over time, with the book printing both the
superseded and current version of several rules.

Ingest produced **3,447 chunks** (3,320 prose + 127 table) via three-signal
heading detection (explicit numeric styles → per-chapter Synopsis TOC matching →
regex fallback), with 0 chapters falling through to the fallback. Two data-quality
issues were found and fixed during ingest, both verified against the real
document rather than assumed:

- **Currency glyph inconsistency** — the source used a backtick as the rupee
  symbol in 486 places and the real ₹ in 252 others, often in the same chapter.
  Unnormalized, this silently halves lexical recall on threshold questions.
  Fixed with a targeted regex at ingest.
- **Table forward-fill** — some tables use blank cells to imply "same category as
  the row above" rather than a true Word merge (confirmed by inspecting the raw
  XML — no `vMerge` tag present). When a large table split across multiple
  chunks, a row separated from its labelling row lost that context entirely.
  Fixed by forward-filling blank leading cells before chunking; verified zero
  blank first-column cells remain across all 127 table chunks post-fix.

## 2. Part 1 — Baseline vanilla RAG

`baseline.py` — dense-only retrieval (`bge-small-en-v1.5`), k=5, one flat prompt,
no query rewriting, no citation instruction, no abstention capability. Deliberately
unimproved so later components have something to measure against.

**Baseline metrics** (test split, n=5, real retrieval, real generation via
OpenRouter):

| recall@5 | mrr | correct | grounded | cite_ok | abstain_acc | p50_latency |
|---|---|---|---|---|---|---|
| 0.700 | 0.333 | 0.000 | 0.000 | 0.000 | 0.800 | 0.280s |

Two of these numbers are fully explained by construction, not by weak generation
quality:

- **`cite_ok=0.000` is guaranteed, not measured.** `VanillaRAG` hardcodes
  `citations=[]` on every answer — the flat prompt never asks for citations at
  all. This is the intended baseline gap Part 3's grounding work closes.
- **`abstain_acc=0.800`** — `VanillaRAG` also hardcodes `sufficient=True` always,
  so it can never abstain. The test split has 4 answerable + 1 unanswerable
  question; it never wrongly refuses the 4 real ones, but always answers the 1 it
  shouldn't — 4/5 = 0.800. This is the real, measured cost of having no
  abstention logic.
- **`correct`/`grounded` at 0.000** — at least one row in this run was scored
  `unjudged` due to a free-tier judge parse failure (see §5) and excluded from
  the mean rather than counted; the remaining judged rows were marked incorrect.
  Row-level detail is in `artifacts/rows_vanilla.json` for audit.

## 3. Part 2 — DSPy re-implementation and optimization

`dspy_program.py` defines `GSTQA`: a `RewriteQuery` signature (question → GST
statutory search query) feeding a `AnswerFromContext` signature (grounded answer +
citations + explicit `sufficient` abstention flag), both as `dspy.ChainOfThought`
predictors. The rewriter is the deliberate structural choice — user phrasing
("do I need to register") and statutory phrasing ("person liable for registration
under Section 22") diverge sharply on this corpus, and making the rewriter
optimizable (rather than hardcoded) is where most of an optimizer's leverage
should land.

**Optimizer run:** `BootstrapFewShotWithRandomSearch` (chosen over MIPROv2 given
the day's time budget — it only searches few-shot demonstrations, not
instructions, needing far fewer LLM calls). Scope was cut for time:
`num_candidate_programs=2, max_bootstrapped_demos=2, max_labeled_demos=2` against
`num_threads=1`.

**Result:** completed successfully. Best score 7.5%, 5 candidate programs
searched, **0 few-shot demonstrations attached to either predictor**. Instructions
are unchanged on both — expected, since Bootstrap never touches instructions,
only demonstrations.

**Why zero demos, diagnosed precisely rather than left as a mystery:** this run
used `--fake` retrieval (hash-based embeddings, no reranker) by necessity — see
§5 for why. `composite_metric` returns a hard `0.0` whenever the judge fails to
return a parseable verdict, and Python treats `0.0` as falsy, so any training
example scoring exactly 0 — whether from weak fake-retrieval grounding or a judge
parse failure — never qualified as a demonstration candidate. Between fake
retrieval and free-tier judge flakiness, nothing in the 8-example training set
cleared that bar.

**What's missing, and why:** a controlled before/after comparison — vanilla vs.
optimized, both under identical real retrieval — was attempted twice
(`evaluate.py --arms vanilla,dspy`) and blocked both times before completing:
first by a platform-specific PyTorch/MPS segfault (root-caused and routed around,
see §5), then by exhausting OpenRouter's free-tier daily request cap (50/day)
before the run finished. This is not a silent gap — it's the one piece of Part 2
left incomplete, and the fix is a single command once daily quota resets or a
$10 OpenRouter top-up is applied (raises the cap to 1000/day):

```bash
python src/evaluate.py --arms vanilla,dspy --split test
```

## 4. Part 3 — Modern RAG techniques

**Hybrid retrieval with reranking.** Dense (`bge-small-en-v1.5`) + sparse BM25
with identifier-aware tokenization (alphanumeric-hyphen tokens preserved, so
`CMP-04` doesn't get shredded into `cmp`/`04`) fused via Reciprocal Rank Fusion
(rank-only, no cross-scale score calibration needed) → cross-encoder rerank
(`bge-reranker-base`) over the fused top-20 → small-to-big expansion to the full
parent section. Justification: the corpus's 1,447 form-name and 1,692
section-reference identifiers mean dense-only retrieval reliably confuses
near-miss pairs (`CMP-04` vs `CMP-08`) that are semantically adjacent but legally
distinct.

**Hallucination guardrails**, three layers:
- *Citation validation* (deterministic, `guardrails.py`) — every cited
  `chunk_id` must exist in what was actually retrieved; catches fabrication with
  no LLM call.
- *Calibrated abstention* — `sufficient` is a first-class model output, not an
  emergent behavior, scored separately from answer correctness.
- *Temporal conflict detection* (metadata-driven, not model-driven) — fires when
  retrieved chunks share a statutory identifier (Section/Rule/Form) but carry
  conflicting effective dates. Verified live against a real case: the GST
  registration threshold changed from ₹20L/₹10L (1 July 2017) to ₹40L/₹20L
  (1 April 2019); the source chunk carries both dates, and an answer that quotes
  the pre-2019 figure would be *faithful to its retrieved context* while being
  wrong in the world — exactly the failure mode standard faithfulness metrics
  cannot see, since they only check "did this come from the document," not
  "is this the current version of what's in the document."

## 5. Engineering challenges encountered

Documented because the assignment specifically asks how a real-world RAG system
gets built, and this is what that actually looked like today:

- **Response-shape handling.** `dspy.LM`'s legacy call path returns
  `list[str | dict]`, not always a plain string — some free models reply with
  structured/tool-call-shaped dicts. Found and fixed in three separate call
  sites that had each independently assumed a plain string
  (`answer_service.py`, then `baseline.py`, discovered only once each was
  actually exercised).
- **Reasoning-model truncation.** Several free models "think out loud" in a
  separate `reasoning_content` field before answering, and can exhaust their
  token budget before ever writing the final answer — sometimes duplicating the
  unfinished reasoning into both fields identically. Fixed by detecting when the
  "answer" field is identical to (or a substring of) the reasoning field, rather
  than trusting field presence alone.
- **Judge reliability.** The free-tier judge occasionally returns something
  DSPy's structured-output adapters can't parse at all (a stray safety-classifier
  response ignoring the actual task). Built `judge_safely()`: retry once, then
  degrade that single row to "unjudged" rather than crash the whole evaluation
  run.
- **PyTorch/MPS segfault.** `BootstrapFewShotWithRandomSearch` segfaulted
  consistently at the same point on Apple Silicon. Diagnosed by elimination:
  reducing `num_threads` to 1 didn't fix it; disabling HuggingFace tokenizers'
  background forking (`TOKENIZERS_PARALLELISM=false`) didn't fix it; forcing the
  embedder and reranker onto CPU instead of MPS didn't fix it either. Root cause
  not fully isolated within today's time budget — routed around instead by
  running the optimizer with `--fake` retrieval, which removes PyTorch/MPS from
  the loop entirely and is the one condition confirmed not to crash.
- **OpenRouter provider routing.** `litellm` requires an explicit `openrouter/`
  routing prefix distinct from the model path itself
  (`openrouter/google/gemma-4-26b-a4b-it:free`, not just the model path alone) —
  easy to drop, produces an opaque "LLM Provider NOT provided" error.
- **Free-tier daily cap.** OpenRouter's free tier is 50 requests/day account-wide
  (1000/day after a one-time $10 top-up, not per-token spend). This is the actual
  blocker on completing Part 2's controlled comparison today.

## 6. Evaluation set

17 hand-verified questions (compressed from an originally planned 80 given
submission timing), every gold answer and `gold_chunk_id` checked against the
actual document text — not synthesized. 8 factual (from the document's own FAQ
pairs), 4 identifier-precision (including a genuine near-miss pair: GSTR-9
"annual return" vs. GSTR-9C "reconciliation statement"), 4 unanswerable
(confirmed absent by full-text search, plus one unrelated control), 1 temporal
(the registration threshold, chosen specifically because its source chunk
carries both the superseded and current effective dates).

Stratified split: train=8, val=4, **test=5**. The `temporal` category (n=1) is
too small to split — its single question goes straight to test, meaning the
optimizer never saw a temporal training example. Read the headline metrics as
directional given this sample size, not as a statistically robust benchmark.

## 7. What I'd do with more time

1. Complete the blocked vanilla/optimized comparison under real retrieval (one
   command, blocked only by today's rate limit).
2. Isolate the MPS segfault properly rather than routing around it — likely a
   `torch`/`sentence-transformers` version interaction specific to Apple Silicon;
   worth a minimal repro and an upstream issue.
3. Expand the eval set back toward 80 questions across all seven originally
   planned categories (table-dependent, cross-chapter multi-hop, and distractor
   pairs weren't reached in the compressed set).
4. Run MIPROv2 `auto="medium"` for an instruction-level (not just
   demonstration-level) optimization comparison.
5. Add RAGAS as an independent faithfulness/context-recall check against the
   gold-labeled contexts already captured in `eval_set.jsonl`.
