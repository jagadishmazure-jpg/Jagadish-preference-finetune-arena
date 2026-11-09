"""Embed every (prompt, response) text the reward model needs, once, on CPU.

Texts: all candidates, the reward-hacking probes, and the tuned models' answers. Uses
BAAI/bge-small-en-v1.5 (sentence-transformers==6.2.0), normalised, stored as float16 in
data/emb/embeddings.npz. Re-run after new generations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena import data  # noqa: E402
from prefarena.evaluate import GOOD, generations  # noqa: E402
from prefarena.hacks import probes  # noqa: E402
from prefarena.prefs import scored_candidates  # noqa: E402
from prefarena.reward import EMB_FILE, embed_text, key  # noqa: E402
from prefarena.rubric import words  # noqa: E402


def texts() -> dict[str, str]:
    idx, out = data.prompt_index(), {}
    for pid, cands in scored_candidates().items():
        p = idx[pid]
        for c in cands:
            t = embed_text(p, c["response"])
            out[key(t)] = t
        best = sorted(cands, key=lambda c: (-c["score"]["total"], words(c["response"])))[0]
        if p["split"] == "test" and best["score"]["total"] >= GOOD:
            for h in probes(p, best["response"]).values():
                t = embed_text(p, h)
                out[key(t)] = t
    for g in generations().values():
        for pid, r in g.items():
            t = embed_text(idx[pid], r)
            out[key(t)] = t
    return out


def main() -> int:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", device="cpu")
    t = texts()
    keys = sorted(t)
    vecs = model.encode([t[k] for k in keys], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    EMB_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(EMB_FILE, keys=np.array(keys), vectors=vecs.astype(np.float16))
    print(f"{len(keys)} embeddings written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
