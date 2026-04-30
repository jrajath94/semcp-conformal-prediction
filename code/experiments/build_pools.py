"""Convert raw generation records to CPSamplePool objects.

For each question:
  1. Load the K samples + reference answers from the JSON produced by
     generate_pool.py.
  2. Mark sample correctness via exact_match against any reference.
  3. Compute the NLI bidirectional-entailment partition.
  4. Compute sentence embeddings for each sample (MiniLM-L6-v2).
  5. Aggregate cluster-level correctness ('any sample in cluster correct').
  6. Save: <out>.json (metadata) + <out>.npz (embeddings).

Runs on the GPU pod (NLI is 1.5B params, embeddings need batched GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.correctness import is_correct  # noqa: E402
from utils.partition import NLIPartitioner  # noqa: E402


def save_pools(pools_meta, embeddings_per_pool, out_base: str):
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    with open(out_base + ".json", "w") as f:
        json.dump(pools_meta, f)
    np.savez_compressed(out_base + ".npz",
                        **{f"emb_{i}": e for i, e in enumerate(embeddings_per_pool)})
    print(f"Wrote {len(pools_meta)} pools to {out_base}.json + .npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True,
                    help="Output prefix; .json and .npz will be appended.")
    ap.add_argument("--nli_model", default="microsoft/deberta-v2-xlarge-mnli")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records from {args.input}")

    print(f"Loading NLI partitioner: {args.nli_model}")
    partitioner = NLIPartitioner(args.nli_model, device="cuda")

    print(f"Loading embedder: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model, device="cuda")

    pools_meta = []
    embeddings_per_pool = []
    for i, rec in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  built {i + 1}/{len(records)} pools")
        samples = rec["samples"]
        refs = rec["answers"]
        sample_correct = [is_correct(s, refs, "exact_match") for s in samples]
        cluster_ids = partitioner.partition(samples)
        n_clusters = max(cluster_ids) + 1 if cluster_ids else 0
        cluster_correct = [False] * n_clusters
        cluster_reps = [""] * n_clusters
        for s_idx, c_idx in enumerate(cluster_ids):
            if sample_correct[s_idx]:
                cluster_correct[c_idx] = True
            if not cluster_reps[c_idx]:
                cluster_reps[c_idx] = samples[s_idx]
        embs = embedder.encode(samples, convert_to_numpy=True,
                               normalize_embeddings=False)
        pools_meta.append({
            "qid": rec["qid"], "question": rec["question"], "samples": samples,
            "references": refs, "sample_correct": sample_correct,
            "cluster_ids": cluster_ids, "cluster_correct": cluster_correct,
            "cluster_reps": cluster_reps,
            "extra": {"mean_token_nll": rec.get("mean_token_nll", []),
                       "dataset": rec.get("dataset", "")},
        })
        embeddings_per_pool.append(embs.astype(np.float32))

    save_pools(pools_meta, embeddings_per_pool, args.output_base)

    n = len(pools_meta)
    admissible = sum(1 for p in pools_meta if any(p["cluster_correct"]))
    avg_clusters = np.mean([len(set(p["cluster_ids"])) for p in pools_meta])
    avg_emcorrect = np.mean([sum(p["sample_correct"]) / len(p["samples"])
                              for p in pools_meta])
    print(f"  admissible: {admissible}/{n} ({admissible/n:.1%})")
    print(f"  avg clusters per question: {avg_clusters:.2f}")
    print(f"  avg per-sample EM correct: {avg_emcorrect:.1%}")


if __name__ == "__main__":
    main()
