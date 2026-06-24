from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import yaml
from pathlib import Path


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class Category(str, Enum):
    single_hop = "single_hop"
    multi_hop = "multi_hop"
    unanswerable = "unanswerable"
    adversarial = "adversarial"
    tool_required = "tool_required"


@dataclass
class GoldenTrajectoryStep:
    goal: str
    expected_tool: Optional[str]  # None = no tool call expected


@dataclass
class GoldenItem:
    id: str
    question: str
    ground_truth: str
    contexts: list[str]       # arxiv_ids or chunk content that SHOULD be retrieved
    difficulty: Difficulty
    category: Category
    failure_mode_targeted: str
    corpus_hash: str
    golden_trajectory: Optional[list[GoldenTrajectoryStep]] = None


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    items = []
    for r in raw["items"]:
        gt = None
        if r.get("golden_trajectory"):
            gt = [GoldenTrajectoryStep(**s) for s in r["golden_trajectory"]]
        items.append(GoldenItem(
            id=r["id"],
            question=r["question"],
            ground_truth=r["ground_truth"],
            contexts=r["contexts"],
            difficulty=Difficulty(r["difficulty"]),
            category=Category(r["category"]),
            failure_mode_targeted=r["failure_mode_targeted"],
            corpus_hash=r["corpus_hash"],
            golden_trajectory=gt,
        ))
    return items
