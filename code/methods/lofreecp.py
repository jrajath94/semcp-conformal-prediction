"""LofreeCP baseline (arXiv 2403.01216, Su et al. 2024).

Logit-FREE Conformal Prediction for LLMs. Operates at the *string* level
without semantic clustering: nonconformity score is

    s(x, y) = -log p_hat(y) + lambda * R(y)

where p_hat(y) = freq(y in samples) / K and R(y) is a length-rank
regularizer (a small tie-breaker for short answers). We follow the paper's
public release (lambda = 0.5, R = log(1 + length / 10)).

Because this is a *string-level* method, we map predicted strings back to
clusters via the same NLI partition for a fair set-size comparison: the
'meaning set' of LofreeCP is the union of clusters touched by any
predicted string.
"""
from __future__ import annotations

from collections import Counter
from typing import List
import numpy as np

from .base import CPMethod, CPPrediction, CPSamplePool, conformal_quantile


class LofreeCP(CPMethod):
    name = "lofreecp"

    def __init__(self, length_lambda: float = 0.5):
        self.q_hat = float("inf")
        self.lam = length_lambda

    def _string_scores(self, pool: CPSamplePool) -> List[float]:
        if not pool.samples:
            return []
        K = len(pool.samples)
        counts = Counter(pool.samples)
        scores = []
        for y in pool.samples:
            p_hat = counts[y] / K
            R = np.log1p(len(y) / 10.0)
            scores.append(-np.log(max(p_hat, 1e-12)) + self.lam * R)
        return scores

    def _cal_score_correct(self, pool: CPSamplePool) -> float:
        if not any(pool.sample_correct):
            return float("inf")
        scores = self._string_scores(pool)
        correct_idxs = [i for i, ok in enumerate(pool.sample_correct) if ok]
        return float(min(scores[i] for i in correct_idxs))

    def calibrate(self, pools: List[CPSamplePool], alpha: float) -> None:
        cal = np.array([self._cal_score_correct(p) for p in pools], dtype=float)
        cal = cal[np.isfinite(cal)]
        if len(cal) == 0:
            self.q_hat = float("inf")
            return
        self.q_hat = conformal_quantile(cal, alpha)

    def predict(self, pool: CPSamplePool, alpha: float) -> CPPrediction:
        scores = self._string_scores(pool)
        if not scores:
            return CPPrediction(qid=pool.qid, selected_clusters=[], set_size=0,
                                abstained=True, correct_in_set=False, score=float("inf"))
        selected_strings = [i for i, s in enumerate(scores) if s <= self.q_hat]
        # Map string indices to cluster indices.
        clusters_in_set = sorted(set(pool.cluster_ids[i] for i in selected_strings))
        abstained = len(clusters_in_set) == 0
        correct_in_set = any(pool.cluster_correct[c] for c in clusters_in_set) if not abstained else False
        return CPPrediction(qid=pool.qid, selected_clusters=clusters_in_set,
                            set_size=len(clusters_in_set), abstained=abstained,
                            correct_in_set=correct_in_set,
                            score=float(min(scores)))
