"""Generate the K-sample pool for each test question, with per-token NLL.

Runs on the GPU pod. Outputs a JSON with one record per question:
  {
    "qid": str, "question": str, "answers": [str, ...],
    "samples": [str, ...],                     # K samples
    "mean_token_nll": [float, ...],            # one per sample
    "dataset": str,
  }

Uses vLLM for fast sampled generation, then a separate forward pass to get
teacher-forced logprobs of each (prompt, sample) pair so TECP has its
nonconformity score input.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Allow running this file directly from the pod / shell.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data import load_dataset_split  # noqa: E402


SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the question with "
    "the shortest factual response. Do not explain. Output only the answer."
)


def build_prompts(examples, tokenizer) -> List[str]:
    msgs_per_q = [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": q.question}]
        for q in examples
    ]
    return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in msgs_per_q]


def sample_with_vllm(model_name: str, prompts: List[str],
                     k: int, max_new_tokens: int) -> List[List[str]]:
    llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.85,
              max_model_len=4096)
    sampling = SamplingParams(n=k, temperature=1.0, top_p=0.95,
                              max_tokens=max_new_tokens, stop=["\n", "</s>"],
                              seed=42)
    outs = llm.generate(prompts, sampling)
    samples_per_prompt = []
    for out in outs:
        s = [c.text.strip() for c in out.outputs]
        samples_per_prompt.append(s)
    del llm
    torch.cuda.empty_cache()
    return samples_per_prompt


def compute_teacher_forced_nll(model_name: str,
                               prompts: List[str],
                               samples_per_prompt: List[List[str]]
                               ) -> List[List[float]]:
    """Mean per-token NLL of each sample given its prompt."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                  device_map="cuda")
    model.train(False)

    out_per_prompt: List[List[float]] = []
    with torch.no_grad():
        for prompt, samples in zip(prompts, samples_per_prompt):
            row = []
            for s in samples:
                full = prompt + s
                ids_full = tok(full, return_tensors="pt").input_ids.to(model.device)
                ids_prompt = tok(prompt, return_tensors="pt").input_ids.to(model.device)
                n_prompt = ids_prompt.size(1)
                if ids_full.size(1) <= n_prompt:
                    row.append(0.0); continue
                logits = model(ids_full).logits[0, n_prompt - 1: -1]
                target = ids_full[0, n_prompt:]
                nll = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
                row.append(float(nll.item()))
            out_per_prompt.append(row)
    del model
    torch.cuda.empty_cache()
    return out_per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_examples", type=int, default=1000)
    ap.add_argument("--k_samples", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    examples = load_dataset_split(args.dataset, n=args.n_examples, seed=args.seed)
    print(f"Loaded {len(examples)} {args.dataset} examples")

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(examples, tok)

    print("Sampling K responses per question with vLLM...")
    samples_per_prompt = sample_with_vllm(args.model, prompts,
                                          args.k_samples, args.max_new_tokens)

    print("Computing teacher-forced NLL for TECP...")
    nll_per_prompt = compute_teacher_forced_nll(args.model, prompts, samples_per_prompt)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    records = []
    for ex, samples, nlls in zip(examples, samples_per_prompt, nll_per_prompt):
        records.append({
            "qid": ex.id,
            "question": ex.question,
            "answers": ex.answers,
            "samples": samples,
            "mean_token_nll": nlls,
            "dataset": ex.dataset,
        })
    with open(args.output, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
