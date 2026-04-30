"""ConU baseline (arXiv 2407.00499, Wang et al. 2024).

Conformal Uncertainty for LLMs. Uses semantic clustering (same NLI partition
as SemCP) and a *frequency*-based nonconformity score:

    s(x, [y]) = 1 - freq(y in samples) / K

The conformal threshold q_hat is computed on calibration examples where
at least one correct sample exists. Test prediction set is
    C(x) = { [y] : 1 - freq([y]) / K <= q_hat }

ConU's correctness-aligned construction means it is intrinsically
conditional on "a correct meaning is sampled" and explicitly assumes
this in its theorem (their Assumption 2).
"""
from __future__ import annotations

from typing import List
import numpy as np

from .base import CPMethod, CPPrediction, CPSamplePool, conformal_quantile


class ConU(CPMethod):
    name = "conu"

    def __init__(self):
        self.q_hat = float("inf")

    @staticmethod
    def _freq_scores(pool: CPSamplePool) -> List[float]:
        if not pool.cluster_ids:
            return []
        K = len(pool.cluster_ids)
        n_clusters = max(pool.cluster_ids) + 1
        counts = np.bincount(pool.cluster_ids, minlength=n_clusters)
        return [1.0 - c / K for c in counts]

    def _cal_score_correct(self, pool: CPSamplePool) -> float:
        if not any(pool.cluster_correct):
            return float("inf")
        scores = self._freq_scores(pool)
        correct = next(c for c, ok in enumerate(pool.cluster_correct) if ok)
        return scores[correct]

    def calibrate(self, pools: List[CPSamplePool], alpha: float) -> None:
        cal = np.array([self._cal_score_correct(p) for p in pools], dtype=float)
        cal = cal[np.isfinite(cal)]
        if len(cal) == 0:
            self.q_hat = float("inf")
            return
        self.q_hat = conformal_quantile(cal, alpha)

    def predict(self, pool: CPSamplePool, alpha: float) -> CPPrediction:
        scores = self._freq_scores(pool)
        if not scores:
            return CPPrediction(qid=pool.qid, selected_clusters=[], set_size=0,
                                abstained=True, correct_in_set=False, score=float("inf"))
        selected = [c for c, s in enumerate(scores) if s <= self.q_hat]
        abstained = len(selected) == 0
        correct_in_set = any(pool.cluster_correct[c] for c in selected) if not abstained else False
        return CPPrediction(qid=pool.qid, selected_clusters=selected,
                            set_size=len(selected), abstained=abstained,
                            correct_in_set=correct_in_set,
                            score=float(min(scores)))
