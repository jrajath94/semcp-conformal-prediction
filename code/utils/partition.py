"""Bidirectional-entailment semantic partition (Kuhn et al. 2023, Farquhar 2024).

For each prompt we sample K responses from the LLM and partition them into
meaning equivalence classes via:
    y_i ~ y_j  iff  NLI(y_i, y_j) = entailment AND NLI(y_j, y_i) = entailment

Union-Find provides transitive closure when NLI judgments are noisy.
The partition is a function of the K samples, computed deterministically
from a fixed NLI model and a fixed pair-ordering rule. This is a
sample-dependent partition (Farquhar 2024 calls this a 'semantic
clustering'); the corresponding coverage guarantee is conditional, treated
rigorously in theory/theorem1_proof.tex.
"""
from __future__ import annotations

from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


_NLI_MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        a, b = self.find(x), self.find(y)
        if a != b:
            self.parent[a] = b


class NLIPartitioner:
    """Bidirectional-entailment partitioning of LLM samples."""

    def __init__(self, model_name: str = _NLI_MODEL_NAME, device: str = "cuda"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.model.train(False)
        # MNLI label index for "entailment" varies; check config:
        # roberta-large-mnli: {0:CONTRADICTION, 1:NEUTRAL, 2:ENTAILMENT}
        # deberta-v2-xlarge-mnli: {0:CONTRADICTION, 1:NEUTRAL, 2:ENTAILMENT}
        self.entail_idx = self.model.config.label2id.get(
            "ENTAILMENT", self.model.config.label2id.get("entailment", 2)
        )

    @torch.no_grad()
    def _entails_batch(self, premises: List[str], hypotheses: List[str],
                       context: str = "") -> List[bool]:
        if not premises:
            return []
        prems = [f"{context} {p}".strip() for p in premises]
        hyps = [f"{context} {h}".strip() for h in hypotheses]
        enc = self.tok(prems, hyps, padding=True, truncation=True,
                       max_length=256, return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)
        ent_probs = probs[:, self.entail_idx].cpu().tolist()
        return [p > 0.5 for p in ent_probs]

    def partition(self, samples: List[str], context: str = "") -> List[int]:
        """Return cluster_id for each of the K samples; ids are 0..n_clusters-1.

        Two samples i, j are in the same cluster iff
          entails(i->j, given context) AND entails(j->i, given context).
        Empty / whitespace-only samples form a singleton cluster each so they
        do not bridge unrelated answers via vacuous entailment.
        """
        n = len(samples)
        if n == 0:
            return []
        uf = UnionFind(n)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if samples[i].strip() and samples[j].strip()]
        if pairs:
            prems = [samples[i] for i, j in pairs]
            hyps = [samples[j] for i, j in pairs]
            forward = self._entails_batch(prems, hyps, context)
            backward = self._entails_batch(hyps, prems, context)
            for (i, j), f, b in zip(pairs, forward, backward):
                if f and b:
                    uf.union(i, j)
        roots = {uf.find(i) for i in range(n)}
        root_to_id = {r: idx for idx, r in enumerate(sorted(roots))}
        return [root_to_id[uf.find(i)] for i in range(n)]


def cluster_predictions(samples: List[str], cluster_ids: List[int]
                         ) -> List[List[str]]:
    """Group sample strings by their cluster id."""
    by_id: dict[int, List[str]] = {}
    for s, c in zip(samples, cluster_ids):
        by_id.setdefault(c, []).append(s)
    return [by_id[c] for c in sorted(by_id)]
