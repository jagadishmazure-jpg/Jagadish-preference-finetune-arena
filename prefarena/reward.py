"""Reward models: learn a score r(prompt, response) from pairwise preferences (Bradley-Terry).

P(a beats b) = sigmoid(r(a) - r(b)) with r linear in features, fitted as logistic regression on
feature differences (both orders, no intercept). Two feature sets:
  * bow:   TF-IDF words and bigrams of the response (a "bag of words" baseline),
  * embed: a small open sentence-embedding model (bge-small, 384 numbers) of policy + question +
           response, computed once on CPU and committed.
A length-only rule (prefer the longer, or the shorter, answer) is the floor to beat.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import data
from .styles import user_message

EMB_FILE = data.DATA / "emb" / "embeddings.npz"


def key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def embed_text(prompt: dict, response: str) -> str:
    return f"{user_message(prompt)}\n\nAssistant: {response}"


@lru_cache(maxsize=1)
def embeddings() -> dict[str, np.ndarray]:
    z = np.load(EMB_FILE)
    return {k: v.astype(np.float32) for k, v in zip(z["keys"].tolist(), z["vectors"])}


class BT:
    def __init__(self, kind: str, C: float = 1.0):
        self.kind, self.C = kind, C

    def _feats(self, items: list[tuple[dict, str]]):
        if self.kind == "bow":
            return self.vec.transform([r for _, r in items])
        e = embeddings()
        return np.stack([e[key(embed_text(p, r))] for p, r in items])

    def fit(self, pairs: list[tuple[dict, str, str]]) -> BT:
        """pairs: (prompt, winner, loser)."""
        if self.kind == "bow":
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.vec.fit([w for _, w, _ in pairs] + [l_ for _, _, l_ in pairs])
        fw = self._feats([(p, w) for p, w, _ in pairs])
        fl = self._feats([(p, l_) for p, _, l_ in pairs])
        diff = fw - fl
        x = diff if self.kind != "bow" else diff.tocsr()
        from scipy.sparse import vstack

        X = vstack([x, -x]) if self.kind == "bow" else np.vstack([x, -x])
        y = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
        self.lr = LogisticRegression(C=self.C, fit_intercept=False, max_iter=5000).fit(X, y)
        return self

    def reward(self, items: list[tuple[dict, str]]) -> np.ndarray:
        f = self._feats(items)
        return np.asarray(f @ self.lr.coef_[0]).ravel()


def length_rule(longer: bool):
    def reward(items: list[tuple[dict, str]]) -> np.ndarray:
        n = np.array([len(r.split()) for _, r in items], dtype=float)
        return n if longer else -n

    return reward
