"""Replace TODO_NUM placeholders in paper.tex with values from summaries.json.

Reads /workspace/semcp/results/<dataset>/summaries.json (one row per
method/alpha/seed), aggregates over seeds, and substitutes the appropriate
numbers into the LaTeX table cells. This keeps the paper synchronised with
real experimental output and avoids manual transcription errors.

Usage:
    python3 inject_results.py \
        --paper artifacts/deliverables/paper.tex \
        --results_dir results/ \
        --datasets triviaqa squad
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

METHODS_IN_TABLE = ["semcp", "conu", "safer", "lofreecp", "tecp"]


def load_summaries(results_dir: str, dataset: str) -> List[Dict]:
    p = os.path.join(results_dir, dataset, "summaries.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def aggregate(rows: List[Dict], method: str) -> Dict[str, str]:
    sub = [r for r in rows if r["method"] == method]
    if not sub:
        return {k: "--" for k in
                ["cov_marg", "cov_cond", "set_size", "abst", "p_a"]}

    def stat(key: str, fmt: str = "{:.3f}") -> str:
        import numpy as np
        vals = np.array([r[key] for r in sub], dtype=float)
        m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
        return f"{fmt.format(m)} $\\pm$ {fmt.format(s)}"

    return {
        "cov_marg": stat("coverage_marginal", "{:.3f}"),
        "cov_cond": stat("coverage_conditional", "{:.3f}"),
        "set_size": stat("set_size_active", "{:.2f}"),
        "abst": stat("abstain_rate", "{:.3f}"),
        "p_a": stat("admissible_rate", "{:.3f}"),
    }


def build_replacement_block(stats: Dict[str, Dict[str, str]], dataset: str) -> List[str]:
    """One LaTeX row per method, in METHODS_IN_TABLE order."""
    pretty = {"semcp": "SemCP (ours)",
              "conu": "ConU", "safer": "SAFER",
              "lofreecp": "LofreeCP", "tecp": "TECP"}
    rows = []
    for m in METHODS_IN_TABLE:
        s = stats[m]
        row = (f"{pretty[m]:<13} & {s['cov_marg']} & {s['cov_cond']} & "
               f"{s['set_size']} & {s['abst']} & {s['p_a']} \\\\")
        rows.append(row)
    return rows


def replace_table_block(tex: str, dataset: str, replacement_rows: List[str]) -> str:
    """Replace the placeholder rows in the table for `dataset`.

    The original paper.tex has placeholder rows of the form:
        SemCP (ours)  & TODO_NUM & TODO_NUM & TODO_NUM & TODO_NUM & TODO_NUM \\
        ConU          & TODO_NUM & ...
        ...

    grouped under \multicolumn{6}{c}{\textit{<dataset>}}.
    """
    # The placeholder block looks like:
    #   \multicolumn{6}{c}{\textit{TriviaQA}} \\
    #   \midrule
    #   SemCP (ours)  & TODO\_NUM & ... \\
    #   ConU          & TODO\_NUM & ... \\
    #   ... (one row per method) ...
    label = f"\\multicolumn{{6}}{{c}}{{\\textit{{{dataset}}}}}"
    pattern = re.compile(
        re.escape(label) + r" \\\\[ \t]*\n\\midrule[ \t]*\n"
        + r"(?:[^\n]*\\\\[ \t]*\n){" + str(len(replacement_rows)) + r"}",
        re.MULTILINE,
    )
    new_block = label + " \\\\\n\\midrule\n" + "\n".join(replacement_rows) + "\n"
    return pattern.sub(new_block, tex, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tex = Path(args.paper).read_text()

    for ds in args.datasets:
        rows = load_summaries(args.results_dir, ds)
        if not rows:
            print(f"[skip] no summaries for {ds}")
            continue
        stats = {m: aggregate(rows, m) for m in METHODS_IN_TABLE}
        print(f"[{ds}] aggregated stats:")
        for m, s in stats.items():
            print(f"  {m}: {s}")
        repl = build_replacement_block(stats, ds)
        # Capitalised dataset name as it appears in the LaTeX label
        ds_label = "TriviaQA" if ds == "triviaqa" else "SQuAD"
        tex = replace_table_block(tex, ds_label, repl)

    out = args.output or args.paper
    Path(out).write_text(tex)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
