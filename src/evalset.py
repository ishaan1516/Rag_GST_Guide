"""Harvest FAQ seed pairs from the manual, then split the reviewed set."""
import argparse, json, random, re
from collections import Counter

import config
from index import load_chunks

CATEGORIES = ["faq", "identifier", "table", "multihop", "temporal",
              "unanswerable", "distractor"]


def harvest_faqs(chunks, limit=60):
    """The manual contains 411 FAQ-style paragraphs. Chunk boundaries sometimes
    separate a Question from its Answer, so we also look at the next chunk in
    the same parent section."""
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c["parent_id"], []).append(c)
    for v in by_parent.values():
        v.sort(key=lambda c: c["chunk_id"])

    pairs = []
    for parent, group in by_parent.items():
        joined = "\n".join(c["raw_text"] for c in group)
        ids = [c["chunk_id"] for c in group]
        for m in re.finditer(
                r"(?:Question\s*\d*|Q\.)\s*[:.]?\s*(.{15,300}?)\s*"
                r"(?:Answer|Ans\.)\s*[:.]?\s*(.{30,900}?)"
                r"(?=Question\s*\d|Q\.|$)", joined, re.S):
            q, a = m.group(1).strip(), m.group(2).strip()
            if "?" not in q:
                continue
            pairs.append({"id": f"faq{len(pairs):03d}", "question": q,
                          "gold_answer": a, "gold_chunk_ids": ids[:2],
                          "category": "faq", "answerable": True,
                          "reviewed": False, "notes": ""})
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def split(path=config.EVALSET_PATH, seed=13):
    """Stratified split. The optimizer sees train+val only; test is reported."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    unreviewed = [r for r in rows if not r.get("reviewed")]
    if unreviewed:
        print(f"WARNING: {len(unreviewed)} rows still have reviewed=false")
    rng = random.Random(seed)
    buckets = {}
    for r in rows:
        buckets.setdefault(r["category"], []).append(r)
    train, val, test = [], [], []
    for cat, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_va = int(n * 0.5), int(n * 0.25)
        train += items[:n_tr]; val += items[n_tr:n_tr + n_va]; test += items[n_tr + n_va:]
    for name, part in (("train", train), ("val", val), ("test", test)):
        with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print("by category:", Counter(r["category"] for r in rows).most_common())


def to_dspy(path):
    import dspy
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [dspy.Example(question=r["question"], gold_answer=r["gold_answer"],
                         gold_chunk_ids=r["gold_chunk_ids"],
                         answerable=r["answerable"], category=r["category"]
                         ).with_inputs("question") for r in rows]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    if a.harvest:
        rows = harvest_faqs(load_chunks())
        with open("data/faq_seed.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"harvested {len(rows)} FAQ seeds -> data/faq_seed.jsonl")
        print("Review each one by hand, set reviewed=true, then add the other categories.")
    if a.split:
        split()
