"""Dense / sparse / hybrid retrieval, reranking, and small-to-big expansion."""
import pickle
from dataclasses import dataclass

import faiss
import numpy as np

import config
from embedder import get_embedder
from index import load_chunks, tokenize
from ingest import extract_identifiers, extract_effective_dates


@dataclass
class Hit:
    chunk: dict
    score: float
    source: str


class Retriever:
    def __init__(self, fake=False, use_reranker=False):
        self.chunks = load_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.by_parent = {}
        for c in self.chunks:
            self.by_parent.setdefault(c["parent_id"], []).append(c)

        self.index = faiss.read_index(f"{config.ARTIFACTS}/faiss.index")
        with open(f"{config.ARTIFACTS}/bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)["bm25"]
        self.embedder = get_embedder(config.EMBED_MODEL, fake=fake)

        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            import os as _os
            _device = "cpu" if _os.environ.get("FORCE_CPU", "").lower() in \
                              ("1", "true", "yes") else None
            self.reranker = CrossEncoder(config.RERANK_MODEL, device=_device)

    # ---------------- single-strategy retrieval

    def dense(self, query, k=config.DENSE_K):
        q = np.ascontiguousarray(self.embedder.encode([query]), dtype="float32")
        scores, idx = self.index.search(q, k)
        return [Hit(self.chunks[i], float(s), "dense")
                for i, s in zip(idx[0], scores[0]) if i >= 0]

    def sparse(self, query, k=config.SPARSE_K):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i], float(scores[i]), "bm25") for i in top]

    # ---------------- fusion

    def hybrid(self, query, k=config.FUSE_K, rrf_k=config.RRF_K):
        """Reciprocal Rank Fusion. Uses rank only, so dense cosine scores and
        unbounded BM25 scores never have to be put on a common scale."""
        fused = {}
        for lst in (self.dense(query), self.sparse(query)):
            for rank, h in enumerate(lst):
                cid = h.chunk["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.by_id[cid], s, "hybrid") for cid, s in top]

    # ---------------- reranking

    def rerank(self, query, hits, top_n=config.RERANK_K):
        if self.reranker is None:
            return hits[:top_n]
        scores = self.reranker.predict([(query, h.chunk["text"]) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: -float(x[1]))[:top_n]
        return [Hit(h.chunk, float(s), "rerank") for h, s in ranked]

    # ---------------- small-to-big

    def expand(self, hits, max_tokens=6000):
        """Retrieve on a 400-token target, generate from the full section."""
        seen, out, budget = set(), [], max_tokens
        for h in hits:
            pid = h.chunk["parent_id"]
            if pid in seen:
                continue
            seen.add(pid)
            sibs = sorted(self.by_parent.get(pid, [h.chunk]),
                          key=lambda c: c["chunk_id"])
            body = "\n".join(s["raw_text"] for s in sibs)
            cost = len(body) // 4
            if cost > budget:
                body, cost = h.chunk["raw_text"], h.chunk["token_count"]
            # recompute metadata over the merged body - the parent section may
            # contain identifiers and dates absent from the retrieved fragment,
            # and the temporal guardrail reads these fields
            out.append({**h.chunk, "raw_text": body, "score": h.score,
                        "identifiers": extract_identifiers(body),
                        "effective_dates": extract_effective_dates(body)})
            budget -= cost
            if budget <= 0:
                break
        return out

    # ---------------- one-call pipeline

    def search(self, query, mode="hybrid", rerank=False, expand=True, k=5):
        if mode == "dense":
            hits = self.dense(query, k=k if not rerank else config.FUSE_K)
        elif mode == "sparse":
            hits = self.sparse(query, k=k if not rerank else config.FUSE_K)
        else:
            hits = self.hybrid(query)
        hits = self.rerank(query, hits, top_n=k) if rerank else hits[:k]
        return self.expand(hits) if expand else [
            {**h.chunk, "score": h.score} for h in hits]


def format_context(docs):
    return "\n\n".join(
        f"[{d['chunk_id']}] {d['chapter_kind'].title()} {d['chapter_num']} — "
        f"{d['chapter_title']}"
        + (" > " + " > ".join(d["section_path"]) if d["section_path"] else "")
        + f"\n{d['raw_text']}"
        for d in docs)