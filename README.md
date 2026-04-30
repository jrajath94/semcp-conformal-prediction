# SemCP: Conformal Prediction over Meaning Classes for Open-Ended LLM Generation

> NeurIPS 2026 submission · Conformal prediction lifted from token / string space into the **quotient space of meanings**.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](artifacts/deliverables/paper.pdf)

---

## TL;DR

Standard conformal prediction (CP) for LLMs operates over strings. So *"Paris"*, *"Paris, France"*, and *"the capital of France is Paris"* count as three separate items in a prediction set even though they are one meaning. **SemCP** partitions LLM samples into bidirectional-NLI semantic classes and runs split-CP on the resulting quotient space, producing prediction sets whose elements are meanings.

Our main theoretical result is **Theorem 1 (Conditional Semantic Coverage)**:

> Conditional on the true meaning being among the `K` sampled responses,
> SemCP returns a prediction set $C(x)$ with
> $\mathbb{P}(\Pi(Y) \in C(X) \mid \text{admissible}) \;\ge\; 1 - \alpha - \tfrac{1}{|I|+1}$.

The conditioning matches the convention used by recent CP-for-LLM methods (ConU, SAFER, LofreeCP, TECP) and is necessary because no sampling-restricted method can deliver marginal coverage when the generator never produces the right answer.

---

## Key results (Qwen3-8B, TriviaQA + SQuAD, $N=500$/dataset, $\alpha=0.10$, 3 seeds)

See `artifacts/deliverables/paper.pdf` Table 2 for the full table including bootstrap CIs. Headline:

| Method | TriviaQA `|C|` | SQuAD `|C|` | Cond. coverage |
|---|---|---|---|
| **SemCP (ours)** | populated from `results/triviaqa/summaries.json` | populated from `results/squad/summaries.json` | ≥ 0.89 (target 0.90) |
| ConU | … | … | ≥ 0.89 |
| SAFER | … | … | ≥ 0.89 |
| LofreeCP | … | … | (no theory at cluster level) |
| TECP | … | … | (no theory at cluster level) |

Numbers in the table get filled in by `scripts/inject_results.py` once `code/run_full.sh` finishes; see "Reproducing" below. The committed PDF reflects the most recent run.

---

## Repository layout

```
semcp-conformal-prediction/
├── README.md                       # this file
├── .gitignore                      # strict ignore for caches, secrets, large data
├── LICENSE                         # MIT
├── artifacts/
│   └── deliverables/
│       ├── paper.tex               # NeurIPS 2025 style, single source of truth
│       ├── paper.pdf               # latest compiled draft
│       ├── references.bib
│       └── charts/                 # final figures (PDF + PNG, 300 dpi)
├── code/
│   ├── methods/                    # 5 conformal methods sharing a common interface
│   │   ├── base.py                 # CPSamplePool, CPMethod, conformal_quantile
│   │   ├── semcp.py                # ours: kernel-based lifted score + abstention
│   │   ├── conu.py                 # ConU (arXiv 2407.00499)
│   │   ├── safer.py                # SAFER (arXiv 2510.10193)
│   │   ├── lofreecp.py             # LofreeCP (arXiv 2403.01216)
│   │   └── tecp.py                 # TECP (arXiv 2509.00461)
│   ├── utils/
│   │   ├── data.py                 # TriviaQA / SQuAD loaders
│   │   ├── correctness.py          # SQuAD-style EM + token F1
│   │   ├── partition.py            # NLI Union-Find bidirectional entailment
│   │   └── metrics.py              # marginal/conditional coverage, bootstrap CIs
│   ├── experiments/
│   │   ├── generate_pool.py        # vLLM sampling + teacher-forced NLL
│   │   ├── build_pools.py          # NLI partition + sentence embeddings
│   │   ├── run_main.py             # calibrate + predict, write summaries.json
│   │   └── make_figures.py         # publication figures
│   ├── theory/
│   │   └── theorem1_proof.tex      # standalone proof (referenced from paper)
│   └── run_full.sh                 # orchestrates the entire run
└── results/                        # tracked: only summaries.json per dataset
    ├── triviaqa/summaries.json
    └── squad/summaries.json
```

---

## Reproducing

### 1. Hardware

Any single GPU with ≥ 24 GB VRAM. Tested on:
- NVIDIA RTX 6000 Ada Generation (48 GB) — community RunPod, ~$0.74/hr
- NVIDIA H100 80 GB HBM3 (faster, more expensive)

Model weights downloaded at runtime (~16 GB for Qwen3-8B).

### 2. Environment

```bash
# Tested combination — see code/install_compatible.sh for the full install script
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.6.3.post1
pip install transformers==4.45.2 sentence-transformers==3.1.1 \
    datasets==3.0.1 accelerate==0.34.2 huggingface_hub==0.25.2 \
    tokenizers==0.20.3 scikit-learn matplotlib seaborn
```

### 3. Authentication

You only need an HF token if you want to swap to a gated model (e.g. Llama-3.1).
Qwen3-8B is open-access:

```bash
export HF_TOKEN=hf_xxx       # only needed for gated models
```

### 4. Run

```bash
cd code
chmod +x run_full.sh
./run_full.sh
```

Wall clock on RTX 6000 Ada (Qwen3-8B, $N=500$, $K=10$, both datasets):

| Stage | TriviaQA | SQuAD |
|---|---|---|
| Sampling (vLLM) | ~5 min | ~6 min |
| Teacher-forced NLL | ~10 min | ~12 min |
| NLI partition + embeddings | ~7 min | ~8 min |
| Methods × 3 seeds | ~2 min | ~2 min |

**Total ≈ 50 min.**

Artifacts land in `data/` (raw samples + pools), `results/<dataset>/` (summaries + raw predictions), and `logs/`.

### 5. Compile the paper

```bash
cd artifacts/deliverables
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Method at a glance

```
prompt x ──► Qwen3-8B ──► K samples {y_1, …, y_K}
                              │
                              ▼
                  bidirectional NLI partition Π
                              │
                              ▼
                clusters c_1, …, c_m  (each = a meaning)
                              │
                              ▼
            kernel mean embedding μ_x = (1/K) Σ φ(y_k)
                              │
                              ▼
       lifted score s̃(x, c) = 1 − max_{c' ≠ c} κ_σ(φ̄_c, φ̄_{c'})
                              │
                              ▼
   q̂ = quantile_(⌈(1−α)(|I|+1)⌉/|I|) over admissible cal scores
                              │
                              ▼
      C(x) = { c : s̃(x, c) ≤ q̂ }   ← prediction set over meanings
```

Implementation lives in `code/methods/semcp.py` (~140 LOC).

---

## Comparison to prior work

| Method | Score space | Conditioning | Abstention | Coverage target |
|---|---|---|---|---|
| Quach et al. 2023 (token-CP) | tokens | none | no | marginal |
| ConU | clusters (NLI) | "correct sample present" | no | conditional, correctness-aligned |
| SAFER | clusters | abstention rule | yes | risk-bounded |
| LofreeCP | strings | implicit | no | marginal |
| TECP | strings | implicit | no | marginal |
| **SemCP (ours)** | **clusters + RBF kernel** | "correct sample present" | implicit (+∞ score) | **conditional over meanings** |

The reviewer-flagged gap in the v1 manuscript (theorem promised marginal coverage that no sampling-restricted method can deliver) is fixed by adopting the conditional formulation explicitly.

---

## Honest limitations

1. **Single open model**: Qwen3-8B is a strong open-weight 8B model but not the strongest available. We expect the set-size ranking to be largely model-independent, but absolute admissibility scales with model strength.
2. **Two QA datasets**: TriviaQA and SQuAD test the closed-form QA regime where bidirectional NLI works well. Open-ended generation (summarization, dialogue, code) is future work — the equivalence relation itself becomes harder to formalize.
3. **No human evaluation of cluster fidelity**: we report cluster counts and ablations, but the partition itself is judged only by NLI agreement, not by human labels.
4. **K = 10 samples**: covers the regime where most CP-for-LLM papers operate, but very high-entropy queries may need K > 10 for the admissibility event to fire reliably.

These are stated explicitly in the paper (Section 7) and are the natural future-work directions.

---

## Citing

If you use SemCP, please cite the paper:

```bibtex
@inproceedings{semcp2026,
  title  = {SemCP: Coverage Guarantees Over Meanings, Not Strings},
  author = {Anonymous},
  booktitle = {Submitted to NeurIPS 2026},
  year   = {2026}
}
```

---

## License

[MIT](LICENSE). See `LICENSE` for terms. Datasets and pretrained models retain their original licenses (Apache 2.0 for Qwen3, Wikipedia/CC for TriviaQA, CC-BY-SA for SQuAD).

---

## Acknowledgements

Compute via [RunPod](https://runpod.io). Thanks to the maintainers of vLLM, Hugging Face Transformers, sentence-transformers, and the conformal-prediction community for foundational software.
