"""Embedding backends.

RealEmbedder downloads a model (~130MB) on first use.
HashEmbedder needs no download and no network - use it to verify the
pipeline wiring before committing to the big install. Its retrieval
quality is poor by design; it exists only to prove the plumbing works.
"""
import os

# Must be set before sentence_transformers (and the HF tokenizers it pulls
# in) is imported anywhere. Tokenizers silently forks background processes
# for parallel tokenization; combined with PyTorch's MPS backend that fork
# is a known crash source on Mac - the "leaked semaphore" / multiprocessing
# resource_tracker warning is the classic symptom.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import numpy as np


class HashEmbedder:
    dim = 384

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class RealEmbedder:
    def __init__(self, name, device=None):
        from sentence_transformers import SentenceTransformer
        # FORCE_CPU=true routes around a segfault seen on Apple Silicon
        # during heavy repeated MPS use (DSPy's optimizer bootstrapping
        # phase). Root cause not fully isolated - CPU sidesteps the whole
        # Metal backend rather than continuing to guess at it.
        device = device or (os.environ.get("FORCE_CPU", "").lower() in
                            ("1", "true", "yes") and "cpu") or None
        self.model = SentenceTransformer(name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, **kw):
        return self.model.encode(texts, normalize_embeddings=True,
                                 batch_size=64, **kw).astype("float32")


def get_embedder(name=None, fake=False):
    return HashEmbedder() if fake else RealEmbedder(name)