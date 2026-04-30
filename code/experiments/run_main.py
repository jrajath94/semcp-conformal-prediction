"""Main experiment runner for SemCP and baselines.

Workflow:
  1. Load pool list from <pool_base>.json + .npz.
  2. For each (method, alpha, seed), split into cal/test and run
     calibrate() then predict() on every test pool.
  3. Aggregate metrics with bootstrap CIs and seed averaging.
  4. Save raw predictions and aggregated table.

This file runs on the GPU pod after build_pools.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.base import CPSamplePool, CPMethod  # noqa: E402
from methods.semcp import SemCP  # noqa: E402
from methods.conu import ConU  # noqa: E402
from methods.safer import SAFER  # noqa: E402
from methods.lofreecp import LofreeCP  # noqa: E402
from methods.tecp import TECP  # noqa: E402
from utils.metrics import coverage_summary, aggregate_seeds  # noqa: E402


METHOD_REGISTRY = {
    "semcp": SemCP,
    "conu": ConU,
    "safer": SAFER,
    "lofreecp": LofreeCP,
    "tecp": TECP,
}


def load_pools(base: str) -> List[CPSamplePool]:
    with open(base + ".json") as f:
        meta = json.load(f)
    npz = np.load(base + ".npz")
    pools = []
    for i, m in enumerate(meta):
        emb = npz[f"emb_{i}"]
        pools.append(CPSamplePool(
            qid=m["qid"], question=m["question"], samples=m["samples"],
            references=m["references"], sample_correct=m["sample_correct"],
            cluster_ids=m["cluster_ids"], cluster_correct=m["cluster_correct"],
            cluster_reps=m["cluster_reps"], embeddings=emb, extra=m.get("extra", {}),
        ))
    return pools


def cal_test_split(pools: List[CPSamplePool], frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(pools)))
    rng.shuffle(idx)
    n_cal = int(len(idx) * frac)
    cal = [pools[i] for i in idx[:n_cal]]
    test = [pools[i] for i in idx[n_cal:]]
    return cal, test


def run_one_method(method_name: str, pools: List[CPSamplePool],
                    alpha: float, seed: int, cal_frac: float):
    method_cls = METHOD_REGISTRY[method_name]
    method: CPMethod = method_cls()
    cal, test = cal_test_split(pools, cal_frac, seed)
    method.calibrate(cal, alpha)

    correct, sizes, abstained, admissible = [], [], [], []
    raw = []
    for p in test:
        pred = method.predict(p, alpha)
        correct.append(pred.correct_in_set)
        sizes.append(pred.set_size)
        abstained.append(pred.abstained)
        admissible.append(any(p.cluster_correct))
        raw.append({"qid": p.qid, "set_size": pred.set_size,
                    "abstained": pred.abstained,
                    "correct_in_set": pred.correct_in_set,
                    "score": pred.score, "admissible": admissible[-1]})
    summary = coverage_summary(correct, sizes, admissible, abstained, seed=seed)
    summary["method"] = method_name
    summary["alpha"] = alpha
    summary["seed"] = seed
    summary["q_hat"] = float(getattr(method, "q_hat", float("nan")))
    if hasattr(method, "sigma"):
        summary["sigma"] = float(method.sigma)
    return summary, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.10])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+",
                    default=list(METHOD_REGISTRY.keys()))
    ap.add_argument("--cal_frac", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pools = load_pools(args.pool_base)
    print(f"Loaded {len(pools)} pools.")

    all_summaries = []
    for method_name in args.methods:
        for alpha in args.alphas:
            per_seed = []
            for seed in args.seeds:
                summary, raw = run_one_method(method_name, pools, alpha,
                                               seed, args.cal_frac)
                per_seed.append(summary)
                all_summaries.append(summary)
                with open(os.path.join(args.output_dir,
                          f"raw_{method_name}_a{alpha}_s{seed}.json"), "w") as f:
                    json.dump(raw, f)
                print(f"  [{method_name} a={alpha} s={seed}] "
                      f"cov_marg={summary['coverage_marginal']:.3f} "
                      f"cov_cond={summary['coverage_conditional']:.3f} "
                      f"sz={summary['set_size_active']:.2f} "
                      f"abs={summary['abstain_rate']:.2f}")

    with open(os.path.join(args.output_dir, "summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote summaries to {args.output_dir}/summaries.json")


if __name__ == "__main__":
    main()
