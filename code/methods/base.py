"""Shared method interface for conformal prediction over QA outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class CPSamplePool:
    """Container for the K samples generated for one prompt."""
    qid: str
    question: str
    samples: List[str]                 # K LLM responses
    references: List[str]              # gold answers (1 or more)
    sample_correct: List[bool]         # which of the K samples is correct
    cluster_ids: List[int]             # NLI partition over the K samples
    cluster_correct: List[bool]        # one bool per cluster (any sample correct)
    cluster_reps: List[str]            # one representative string per cluster
    embeddings: Optional[np.ndarray] = None   # (K, d), sentence embeddings
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPPrediction:
    """Output of a CP method on one test prompt."""
    qid: str
    selected_clusters: List[int]       # indices into pool.cluster_reps
    set_size: int                      # |C(x)| in clusters
    abstained: bool                    # method declined to return a useful set
    correct_in_set: bool               # any selected cluster contains a correct sample
    score: float                       # method's chosen "score" for the test (debug)


class CPMethod:
    """Abstract base. Concrete methods implement calibrate() and predict()."""

    name: str = "base"

    def calibrate(self, pools: List[CPSamplePool], alpha: float) -> None:
        """Estimate the conformal threshold from a calibration pool list."""
        raise NotImplementedError

    def predict(self, pool: CPSamplePool, alpha: float) -> CPPrediction:
        """Apply the calibrated threshold to a test pool."""
        raise NotImplementedError


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Standard split-CP quantile: q_hat = ceil((1-alpha)(n+1))/n quantile."""
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])
