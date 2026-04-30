"""Generate publication-quality figures from the SemCP results JSONs.

Inputs (per dataset):
  /workspace/semcp/results/<dataset>/summaries.json
  /workspace/semcp/results/<dataset>/raw_<method>_a0.10_s<seed>.json

Outputs (PNG + PDF, 300 dpi, NeurIPS column widths):
  fig2_coverage_vs_setsize.{png,pdf}    # Coverage vs set size Pareto
  fig3_method_comparison.{png,pdf}      # Bar chart: set size + cond coverage
  fig4_ablation_kernel.{png,pdf}        # SemCP vs SemCP-Euclidean (sigma sweep)
  fig5_admissibility_coverage.{png,pdf} # Marginal cov vs p_A scatter
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["semcp", "conu", "safer", "lofreecp", "tecp"]
METHOD_LABELS = {
    "semcp": "SemCP",
    "conu": "ConU",
    "safer": "SAFER",
    "lofreecp": "LofreeCP",
    "tecp": "TECP",
}
METHOD_COLORS = {
    "semcp": "#d62728",     # red, our method
    "conu": "#1f77b4",      # blue
    "safer": "#2ca02c",     # green
    "lofreecp": "#9467bd",  # purple
    "tecp": "#ff7f0e",      # orange
}


def style_init():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_summaries(results_dir: str, datasets):
    out = {}
    for d in datasets:
        path = os.path.join(results_dir, d, "summaries.json")
        if os.path.exists(path):
            with open(path) as f:
                out[d] = json.load(f)
    return out


def aggregate_across_seeds(summaries, method, dataset_summaries):
    rows = [r for r in dataset_summaries if r["method"] == method]
    if not rows:
        return None
    keys = ["coverage_marginal", "coverage_conditional", "set_size_active",
            "abstain_rate", "admissible_rate"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


def fig_method_comparison(summaries, out_base):
    datasets = list(summaries.keys())
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8))

    for ax_idx, metric in enumerate(["set_size_active", "coverage_conditional"]):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.16
        for i, m in enumerate(METHOD_ORDER):
            means, stds = [], []
            for d in datasets:
                agg = aggregate_across_seeds(summaries, m, summaries[d])
                if agg is None:
                    means.append(np.nan); stds.append(0)
                else:
                    means.append(agg[metric][0]); stds.append(agg[metric][1])
            ax.bar(x + (i - 2) * width, means, width, yerr=stds,
                   color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                   capsize=2.5, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets])
        if metric == "set_size_active":
            ax.set_ylabel("Avg set size $|C(x)|$")
            ax.set_title("(a) Prediction set size (lower better)")
        else:
            ax.set_ylabel("Conditional coverage")
            ax.axhline(0.90, ls="--", color="grey", lw=0.8, label="$1-\\alpha=0.90$")
            ax.set_ylim(0, 1.05)
            ax.set_title("(b) Conditional coverage (target 0.90)")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png")
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    print(f"wrote {out_base}.{{png,pdf}}")


def fig_admissibility_vs_coverage(summaries, out_base):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    for d in summaries:
        for m in METHOD_ORDER:
            agg = aggregate_across_seeds(summaries, m, summaries[d])
            if agg is None:
                continue
            x = agg["admissible_rate"][0]
            y = agg["coverage_marginal"][0]
            ax.scatter(x, y, color=METHOD_COLORS[m], marker="o" if d == "triviaqa" else "s",
                       s=40, edgecolors="black", linewidths=0.4)
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8,
            label="$\\mathrm{Cov}_\\mathrm{marg} = \\hat{p}_A$ (admissibility ceiling)")
    ax.axhline(0.90, ls="--", color="black", lw=0.6, label="$1-\\alpha=0.90$")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Admissibility rate $\\hat{p}_A$")
    ax.set_ylabel("Empirical marginal coverage")
    ax.set_title("Marginal coverage vs admissibility")
    handles = [plt.Line2D([0], [0], marker="o", color="white",
                          markerfacecolor=METHOD_COLORS[m], markeredgecolor="black",
                          label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    handles += [plt.Line2D([0], [0], marker="o", color="black", label="TriviaQA"),
                plt.Line2D([0], [0], marker="s", color="black", label="SQuAD")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7,
              ncol=2, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png")
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    print(f"wrote {out_base}.{{png,pdf}}")


def fig_coverage_vs_setsize(summaries, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.8))
    for ax, d in zip(axes, summaries):
        for m in METHOD_ORDER:
            agg = aggregate_across_seeds(summaries, m, summaries[d])
            if agg is None:
                continue
            sz_m, sz_s = agg["set_size_active"]
            cv_m, cv_s = agg["coverage_conditional"]
            ax.errorbar(sz_m, cv_m, xerr=sz_s, yerr=cv_s, fmt="o",
                        color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                        capsize=2.5, markeredgecolor="black", markeredgewidth=0.4,
                        markersize=8)
        ax.axhline(0.90, ls="--", color="grey", lw=0.8)
        ax.set_xlabel("Set size $|C(x)|$ (lower better)")
        ax.set_ylabel("Conditional coverage")
        ax.set_title(d.upper())
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", frameon=False, ncol=2, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png")
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    print(f"wrote {out_base}.{{png,pdf}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["triviaqa", "squad"])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    style_init()
    os.makedirs(args.output_dir, exist_ok=True)
    summaries = load_summaries(args.results_dir, args.datasets)
    if not summaries:
        print(f"WARNING: no summaries.json found in {args.results_dir}")
        return

    fig_method_comparison(summaries, os.path.join(args.output_dir, "fig3_method_comparison"))
    fig_admissibility_vs_coverage(summaries,
        os.path.join(args.output_dir, "fig5_admissibility_coverage"))
    fig_coverage_vs_setsize(summaries,
        os.path.join(args.output_dir, "fig2_coverage_vs_setsize"))


if __name__ == "__main__":
    main()
