"""Evaluation metrics for conformal prediction over meaning classes.

Definitions follow the conditional-coverage formulation of our Theorem 1:

  - admissible(x): the test point is "admissible" iff at least one sampled
    response y_k is correct (i.e. matches a reference answer). Coverage is
    only meaningful conditional on admissibility (ConU, SAFER).
  - covered(x): the prediction set C(x) contains the cluster of any
    correct sample y_k. We evaluate this BEFORE abstention is applied,
    so an empty C(x) is uncovered.
  - covered_after_abstain(x): same, but treated as "trivially covered"
    when the method abstains. Reported separately.
  - set_size(x): |C(x)| in number of clusters (meanings). For methods that
    return strings we map back to clusters via the same NLI partition.

Bootstrap CIs are produced via 1000-iter percentile bootstrap.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np


def coverage_summary(
    correct_in_set: Sequence[bool],
    set_sizes: Sequence[int],
    admissible: Sequence[bool],
    abstained: Sequence[bool] = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute marginal + conditional coverage and avg set size with 95% CI."""
    correct_in_set = np.asarray(correct_in_set, dtype=bool)
    set_sizes = np.asarray(set_sizes, dtype=float)
    admissible = np.asarray(admissible, dtype=bool)
    if abstained is None:
        abstained = np.zeros_like(admissible)
    else:
        abstained = np.asarray(abstained, dtype=bool)

    n = len(correct_in_set)
    rng = np.random.default_rng(seed)

    def stat(mask: np.ndarray, vec: np.ndarray):
        if mask.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        v = vec[mask].astype(float)
        boots = []
        for _ in range(n_boot):
            idxs = rng.integers(0, len(v), size=len(v))
            boots.append(v[idxs].mean())
        return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    cov_marg, cov_lo, cov_hi = stat(np.ones(n, bool), correct_in_set.astype(float))
    cov_cond, condc_lo, condc_hi = stat(admissible, correct_in_set.astype(float))
    size_active_mask = ~abstained
    sz_avg, sz_lo, sz_hi = stat(size_active_mask, set_sizes)
    abs_rate = float(abstained.mean())
    adm_rate = float(admissible.mean())

    return {
        "coverage_marginal": cov_marg,
        "coverage_marginal_ci_lo": cov_lo,
        "coverage_marginal_ci_hi": cov_hi,
        "coverage_conditional": cov_cond,
        "coverage_conditional_ci_lo": condc_lo,
        "coverage_conditional_ci_hi": condc_hi,
        "set_size_active": sz_avg,
        "set_size_ci_lo": sz_lo,
        "set_size_ci_hi": sz_hi,
        "abstain_rate": abs_rate,
        "admissible_rate": adm_rate,
        "n": n,
    }


def aggregate_seeds(per_seed: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-seed metric dicts, reporting mean and std."""
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"] = float(np.nanstd(vals))
    return out
