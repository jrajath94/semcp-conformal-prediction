"""TECP baseline (arXiv 2509.00461, Chen et al. 2025).

Token-Entropy Conformal Prediction. Uses the average per-token negative
log-likelihood under the LLM as the nonconformity score (so this method
requires teacher-forced log-probs of each candidate string). Lower entropy
==> more confident.

We compute teacher-forced logprobs at sample time (in generate_pool.py)
and store them in pool.extra['logprobs']. TECP picks all strings with
mean-token-NLL <= q_hat, then maps to cluster set as in LofreeCP.

If logprobs are unavailable, we fall back to using -log freq (string)
which yields a degenerate special case of LofreeCP without the length
regularizer; we report this fallback explicitly in the paper.
"""
from __future__ import annotations

from collections import Counter
from typing import List
import numpy as np

from .base import CPMethod, CPPrediction, CPSamplePool, conformal_quantile


class TECP(CPMethod):
    name = "tecp"

    def __init__(self):
        self.q_hat = float("inf")

    @staticmethod
    def _string_scores(pool: CPSamplePool) -> List[float]:
        lp = pool.extra.get("mean_token_nll")
        if lp is not None and len(lp) == len(pool.samples):
            return [float(x) for x in lp]
        # fallback: -log freq
        if not pool.samples:
            return []
        K = len(pool.samples)
        counts = Counter(pool.samples)
        return [-float(np.log(max(counts[y] / K, 1e-12))) for y in pool.samples]

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
        clusters_in_set = sorted(set(pool.cluster_ids[i] for i in selected_strings))
        abstained = len(clusters_in_set) == 0
        correct_in_set = any(pool.cluster_correct[c] for c in clusters_in_set) if not abstained else False
        return CPPrediction(qid=pool.qid, selected_clusters=clusters_in_set,
                            set_size=len(clusters_in_set), abstained=abstained,
                            correct_in_set=correct_in_set,
                            score=float(min(scores)))
