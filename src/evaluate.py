"""Run every ablation arm on a split and print the comparison table."""
import argparse, json, time
from statistics import mean

import dspy

import config
from baseline import VanillaRAG
from dspy_program import GSTQA, setup_lm
from evalset import to_dspy
from guardrails import Guarded, validate_citations
from metrics import composite_metric, judge_safely, mrr, ndcg_at_k, recall_at_k, citations_valid
from retrieve import Retriever

ARMS = {
    "vanilla":     dict(kind="vanilla"),
    "hybrid":      dict(kind="dspy", rewrite=False, mode="hybrid", rerank=False, guard=False),
    "rerank":      dict(kind="dspy", rewrite=False, mode="hybrid", rerank=True,  guard=False),
    "dspy":        dict(kind="dspy", rewrite=True,  mode="hybrid", rerank=True,  guard=False, load=True),
    "guardrails":  dict(kind="dspy", rewrite=True,  mode="hybrid", rerank=True,  guard=True,  load=True),
}


def build_arm(name, retriever):
    cfg = ARMS[name]
    if cfg["kind"] == "vanilla":
        return VanillaRAG(retriever)
    prog = GSTQA(retriever, use_rewrite=cfg["rewrite"], mode=cfg["mode"],
                 rerank=cfg["rerank"])
    if cfg.get("load"):
        try:
            prog.load(f"{config.ARTIFACTS}/optimized_program.json")
        except Exception as e:
            print(f"  [warn] could not load optimized program ({e}); using unoptimized")
    return Guarded(prog, config.ABSTAIN_THRESHOLD) if cfg["guard"] else prog


def run_arm(name, program, examples):
    rows = []
    for ex in examples:
        t0 = time.time()
        pred = program(question=ex.question)
        latency = time.time() - t0
        ids = [d["chunk_id"] for d in pred.retrieved]
        row = {"id": ex.category, "latency": latency,
               "recall@5": recall_at_k(ids, ex.gold_chunk_ids, 5),
               "mrr": mrr(ids, ex.gold_chunk_ids),
               "ndcg@5": ndcg_at_k(ids, ex.gold_chunk_ids, 5),
               "cite_ok": float(citations_valid(pred)),
               "answerable": ex.answerable,
               "abstained": not pred.sufficient,
               "temporal_flag": bool(getattr(pred, "temporal_flag", False)),
               "temporal_qualified": bool(getattr(pred, "effective_date", ""))}
        if ex.answerable:
            j = judge_safely(ex.question, ex.gold_answer, pred.context, pred.answer)
            row["correct"] = float(j.correct) if j else None
            row["grounded"] = float(j.grounded) if j else None
        else:
            row["correct"] = row["grounded"] = None
        rows.append(row)
    return rows


def summarize(rows):
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    temp = [r for r in rows if r["id"] == "temporal"]
    def m(xs, k):
        vals = [x[k] for x in xs if x.get(k) is not None]
        return mean(vals) if vals else 0.0
    abst_correct = [1.0 if r["abstained"] else 0.0 for r in una] + \
                   [0.0 if r["abstained"] else 1.0 for r in ans]
    return {
        "recall@5": m(rows, "recall@5"), "mrr": m(rows, "mrr"),
        "correct": m(ans, "correct"), "grounded": m(ans, "grounded"),
        "cite_ok": m(rows, "cite_ok"),
        "abstain_acc": mean(abst_correct) if abst_correct else 0.0,
        "temporal_qual": m(temp, "temporal_qualified"),
        "p50_latency": sorted(r["latency"] for r in rows)[len(rows) // 2],
    }


def main(arms, split, fake):
    setup_lm()
    retriever = Retriever(fake=fake, use_reranker=not fake)
    examples = to_dspy(f"data/{split}.jsonl")
    print(f"{len(examples)} examples from {split}\n")

    results = {}
    for name in arms:
        print(f"running arm: {name}")
        rows = run_arm(name, build_arm(name, retriever), examples)
        results[name] = summarize(rows)
        json.dump(rows, open(f"{config.ARTIFACTS}/rows_{name}.json", "w"), indent=2)

    cols = ["recall@5", "mrr", "correct", "grounded", "cite_ok",
            "abstain_acc", "temporal_qual", "p50_latency"]
    print("\n| Arm | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, s in results.items():
        print(f"| {name} | " + " | ".join(f"{s[c]:.3f}" for c in cols) + " |")
    json.dump(results, open(f"{config.ARTIFACTS}/ablation.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    arms = list(ARMS) if a.arms == "all" else a.arms.split(",")
    main(arms, a.split, a.fake)