"""Correctness scoring for QA outputs.

Two correctness criteria are supported and reported separately to match
prior CP-for-LLMs work (ConU, TECP, LofreeCP, SAFER):

  - exact_match: stripped, lowercased, punctuation-removed exact match
    against any reference answer (SQuAD-style EM).
  - f1: token-level F1 against best matching reference.
  - rougeL_f: ROUGE-L F1 (used by some baselines as a softer criterion).

For the marginal/conditional coverage analysis we use exact_match by default,
because the coverage statement is over discrete meanings and EM is the
binary criterion most directly reflected in NLI partitions.
"""
from __future__ import annotations

import re
import string
from typing import List


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, references: List[str]) -> bool:
    p = normalize_text(prediction)
    return any(p == normalize_text(r) for r in references)


def token_f1(prediction: str, reference: str) -> float:
    p_toks = normalize_text(prediction).split()
    r_toks = normalize_text(reference).split()
    if not p_toks or not r_toks:
        return float(p_toks == r_toks)
    common = set(p_toks) & set(r_toks)
    if not common:
        return 0.0
    n_common = sum(min(p_toks.count(t), r_toks.count(t)) for t in common)
    if n_common == 0:
        return 0.0
    prec = n_common / len(p_toks)
    rec = n_common / len(r_toks)
    return 2 * prec * rec / (prec + rec)


def best_f1(prediction: str, references: List[str]) -> float:
    return max((token_f1(prediction, r) for r in references), default=0.0)


def is_correct(prediction: str, references: List[str],
                criterion: str = "exact_match",
                f1_threshold: float = 0.5) -> bool:
    if criterion == "exact_match":
        return exact_match(prediction, references)
    if criterion == "f1":
        return best_f1(prediction, references) >= f1_threshold
    raise ValueError(f"Unknown correctness criterion: {criterion}")
