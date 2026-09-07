"""Central config. Change models/keys here only."""
import os

CHUNKS_PATH   = "data/chunks.jsonl"
EVALSET_PATH  = "data/eval_set.jsonl"
ARTIFACTS     = "artifacts"

EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
RERANK_MODEL  = "BAAI/bge-reranker-base"

# --- Provider switch -------------------------------------------------
# "openai" = paid, most reliable. "openrouter" = free-tier models, rate
# limited (20 req/min, 50/day free or 1000/day after a one-time $10
# top-up - see BUILD_GUIDE.md Stage 3b). Free model IDs rotate; default
# to OpenRouter's own auto-router so you never hardcode one that vanishes.
PROVIDER = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "openrouter"

GEN_MODEL     = "openai/gpt-4.1-mini"     # used when PROVIDER == "openai"
JUDGE_MODEL   = "openai/gpt-4.1"          # MUST differ from GEN_MODEL - avoids self-grading bias

# Two DIFFERENT free model families - keeps the anti-self-grading principle
# even on the free tier. Override with named models if you want to skip
# OpenRouter's auto-router (faster, but the pinned ID may go paid or vanish -
# check https://openrouter.ai/models?max_price=0 for what's currently live).
OPENROUTER_GEN_MODEL   = os.getenv("OPENROUTER_GEN_MODEL", "openrouter/openrouter/free")
OPENROUTER_JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "openrouter/openrouter/free")

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

DENSE_K, SPARSE_K, FUSE_K, RERANK_K, RRF_K = 30, 30, 20, 5, 60
ABSTAIN_THRESHOLD = 0.0
