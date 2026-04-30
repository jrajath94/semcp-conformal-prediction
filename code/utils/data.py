"""Dataset loaders for SemCP experiments.

Loads TriviaQA, SQuAD, and (optionally) CoQA into a unified format:
    {"id": str, "question": str, "answers": List[str]}

Multiple correct answers per question are preserved (TriviaQA aliases,
SQuAD acceptable answers); evaluating against any of them counts as correct,
matching the convention in conformal-prediction literature for QA.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict

from datasets import load_dataset


@dataclass
class QAExample:
    id: str
    question: str
    answers: List[str]
    dataset: str

    def to_dict(self) -> Dict:
        return {"id": self.id, "question": self.question,
                "answers": self.answers, "dataset": self.dataset}


def load_triviaqa(n: int = 1500, seed: int = 42) -> List[QAExample]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idxs:
        row = ds[i]
        answers = list(set(
            [row["answer"]["value"]] + row["answer"].get("aliases", [])
        ))
        answers = [a for a in answers if a and len(a.strip()) > 0]
        if not answers:
            continue
        out.append(QAExample(
            id=row["question_id"],
            question=row["question"],
            answers=answers,
            dataset="triviaqa",
        ))
    return out


def load_squad(n: int = 1500, seed: int = 42) -> List[QAExample]:
    ds = load_dataset("squad", split="validation")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idxs:
        row = ds[i]
        answers = list(set(row["answers"]["text"]))
        answers = [a for a in answers if a and len(a.strip()) > 0]
        if not answers:
            continue
        out.append(QAExample(
            id=row["id"],
            question=f"{row['context']}\n\nQuestion: {row['question']}",
            answers=answers,
            dataset="squad",
        ))
    return out


def load_dataset_split(name: str, n: int = 1500, seed: int = 42) -> List[QAExample]:
    if name == "triviaqa":
        return load_triviaqa(n, seed)
    if name == "squad":
        return load_squad(n, seed)
    raise ValueError(f"Unknown dataset: {name}")


def calibration_test_split(examples: List[QAExample],
                            cal_frac: float = 0.5,
                            seed: int = 42):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_cal = int(len(shuffled) * cal_frac)
    return shuffled[:n_cal], shuffled[n_cal:]


def save_examples(examples: List[QAExample], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in examples], f, indent=2)


def load_examples(path: str) -> List[QAExample]:
    with open(path) as f:
        return [QAExample(**d) for d in json.load(f)]
