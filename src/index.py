"""Build the dense (FAISS) and sparse (BM25) indexes over chunks.jsonl."""
import argparse, json, pickle, re
from pathlib import Path

import faiss
import numpy as np

import config
from embedder import get_embedder

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")


def tokenize(text):
    """Keep alphanumeric-hyphen tokens whole. A default lowercase-split turns
    CMP-04 into ['cmp','04'] - exactly the distinction that matters here."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_chunks(path=config.CHUNKS_PATH):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build(fake=False, out_dir=config.ARTIFACTS):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks")

    emb_model = get_embedder(config.EMBED_MODEL, fake=fake)
    emb = emb_model.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb, dtype="float32"))
    faiss.write_index(index, f"{out_dir}/faiss.index")
    print(f"dense index: {index.ntotal} vectors, dim {emb.shape[1]}")

    from rank_bm25 import BM25Okapi
    corpus = []
    for c in chunks:
        toks = tokenize(c["text"])
        for group in c["identifiers"].values():          # literal identifier tokens
            toks += [t.lower() for t in group]
        corpus.append(toks)
    with open(f"{out_dir}/bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi(corpus),
                     "ids": [c["chunk_id"] for c in chunks]}, f)
    print(f"sparse index: {len(corpus)} docs, "
          f"avg {sum(len(c) for c in corpus)//len(corpus)} tokens")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="use HashEmbedder (no download, wiring test only)")
    build(fake=ap.parse_args().fake)
