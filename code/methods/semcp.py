"""SemCP (corrected): semantic conformal prediction with explicit abstention.

Compared to the v1 manuscript, this implementation:
  1. Defines empty-class score as +inf and treats this as the abstention
     signal (matching the conditional-coverage statement in Theorem 1').
  2. Uses an RBF kernel with bandwidth sigma chosen by minimizing
     calibration set size SUBJECT to the marginal-coverage constraint
     on a held-out cal-train split (no test-set leakage).
  3. Computes the partition function Pi as a fixed deterministic function
     of (LLM samples, NLI model). Pi is sample-dependent but data-distribution-
     independent in the sense required by Theorem 1' (proof in
     theory/theorem1_proof.tex).
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .base import CPMethod, CPPrediction, CPSamplePool, conformal_quantile


def _rbf_kernel(d: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def _cluster_score_rbf(
    embeddings: np.ndarray,
    cluster_indices: List[List[int]],
    target_cluster: int,
    sigma: float,
) -> float:
    """Lifted score: 1 - max kernel similarity between target cluster and its
    nearest neighbour cluster in the K-sample pool.

    Empty target cluster => +inf (no valid representative).
    Single-cluster pool  => 0.0 (trivially closest to itself).
    """
    if target_cluster < 0 or target_cluster >= len(cluster_indices):
        return float("inf")
    if not cluster_indices[target_cluster]:
        return float("inf")
    target_emb = embeddings[cluster_indices[target_cluster]].mean(axis=0)
    sims = []
    for c, idxs in enumerate(cluster_indices):
        if c == target_cluster or not idxs:
            continue
        other_emb = embeddings[idxs].mean(axis=0)
        d = float(np.linalg.norm(target_emb - other_emb))
        sims.append(_rbf_kernel(np.array([d]), sigma)[0])
    if not sims:
        return 0.0
    return float(1.0 - max(sims))


class SemCP(CPMethod):
    """SemCP with corrected scoring + explicit abstention."""

    name = "semcp"

    def __init__(self, sigma_grid: Tuple[float, ...] = (0.1, 0.3, 0.5, 1.0, 2.0, 4.0)):
        self.sigma_grid = sigma_grid
        self.sigma: float = 1.0
        self.q_hat: float = float("inf")

    def _cluster_indices(self, pool: CPSamplePool) -> List[List[int]]:
        out: List[List[int]] = []
        for cid in range(max(pool.cluster_ids) + 1 if pool.cluster_ids else 0):
            out.append([i for i, c in enumerate(pool.cluster_ids) if c == cid])
        return out

    def _scores_for_pool(self, pool: CPSamplePool, sigma: float) -> List[float]:
        if pool.embeddings is None or len(pool.cluster_ids) == 0:
            return []
        cidx = self._cluster_indices(pool)
        return [_cluster_score_rbf(pool.embeddings, cidx, c, sigma)
                for c in range(len(cidx))]

    def _cal_score_correct(self, pool: CPSamplePool, sigma: float) -> float:
        """Lifted score of the cluster that contains a correct sample.
        If no such cluster exists (admissible == False) we return +inf so
        this calibration point is dropped under the conditional protocol
        (matching ConU/SAFER convention).
        """
        if not any(pool.cluster_correct):
            return float("inf")
        cidx = self._cluster_indices(pool)
        correct_cluster = next(c for c, ok in enumerate(pool.cluster_correct) if ok)
        return _cluster_score_rbf(pool.embeddings, cidx, correct_cluster, sigma)

    def calibrate(self, pools: List[CPSamplePool], alpha: float) -> None:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(pools))
        n_train = max(1, len(pools) // 2)
        train = [pools[i] for i in idx[:n_train]]
        cal = [pools[i] for i in idx[n_train:]]

        best_sigma, best_size = self.sigma_grid[0], float("inf")
        for sigma in self.sigma_grid:
            cal_scores = np.array(
                [self._cal_score_correct(p, sigma) for p in train],
                dtype=float,
            )
            cal_scores_admissible = cal_scores[np.isfinite(cal_scores)]
            if len(cal_scores_admissible) == 0:
                continue
            q = conformal_quantile(cal_scores_admissible, alpha)
            sizes = []
            for p in train:
                if not p.cluster_correct:
                    continue
                pool_scores = self._scores_for_pool(p, sigma)
                sz = sum(1 for s in pool_scores if s <= q)
                sizes.append(sz)
            if sizes:
                avg = float(np.mean(sizes))
                if avg < best_size:
                    best_size = avg
                    best_sigma = sigma

        self.sigma = best_sigma
        cal_scores = np.array(
            [self._cal_score_correct(p, self.sigma) for p in cal],
            dtype=float,
        )
        admissible = np.isfinite(cal_scores)
        self.q_hat = conformal_quantile(cal_scores[admissible], alpha)

    def predict(self, pool: CPSamplePool, alpha: float) -> CPPrediction:
        cidx = self._cluster_indices(pool)
        if not cidx:
            return CPPrediction(qid=pool.qid, selected_clusters=[], set_size=0,
                                abstained=True, correct_in_set=False,
                                score=float("inf"))
        scores = [_cluster_score_rbf(pool.embeddings, cidx, c, self.sigma)
                  for c in range(len(cidx))]
        selected = [c for c, s in enumerate(scores) if s <= self.q_hat]
        abstained = len(selected) == 0
        correct_in_set = any(pool.cluster_correct[c] for c in selected) if not abstained else False
        score_min = float(min(scores)) if scores else float("inf")
        return CPPrediction(qid=pool.qid, selected_clusters=selected,
                            set_size=len(selected), abstained=abstained,
                            correct_in_set=correct_in_set, score=score_min)
