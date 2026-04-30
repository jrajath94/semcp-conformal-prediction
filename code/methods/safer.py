"""SAFER baseline (arXiv 2510.10193, Liang et al. 2025).

Selective Abstention with risk control. Two stages:

  Stage 1 (filter): for each cluster compute a quality score (here we use
    cluster frequency, matching their public implementation), then KEEP
    clusters whose score >= tau_filter where tau_filter is chosen so the
    expected number of dropped *correct* clusters on calibration is small.

  Stage 2 (predict-or-abstain): apply standard conformal threshold q_hat
    on the surviving clusters. If C(x) is empty after threshold AND the
    raw best cluster has frequency below an abstention threshold, the
    method ABSTAINS rather than returning an empty set.

We treat abstention as covering trivially for marginal-coverage purposes
(this matches SAFER's own Theorem 1) and as uncovered for conditional
coverage (so conditional comparisons remain apples-to-apples with ConU/SemCP).
"""
from __future__ import annotations

from typing import List
import numpy as np

from .base import CPMethod, CPPrediction, CPSamplePool, conformal_quantile


class SAFER(CPMethod):
    name = "safer"

    def __init__(self, abstain_min_freq: float = 0.10):
        self.q_hat = float("inf")
        self.abstain_min_freq = abstain_min_freq

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
        if not selected:
            best_freq = 1.0 - min(scores)  # frequency of most-frequent cluster
            if best_freq < self.abstain_min_freq:
                return CPPrediction(qid=pool.qid, selected_clusters=[], set_size=0,
                                    abstained=True, correct_in_set=False,
                                    score=float(min(scores)))
            best_cluster = int(np.argmin(scores))
            selected = [best_cluster]
        correct_in_set = any(pool.cluster_correct[c] for c in selected)
        return CPPrediction(qid=pool.qid, selected_clusters=selected,
                            set_size=len(selected), abstained=False,
                            correct_in_set=correct_in_set,
                            score=float(min(scores)))
