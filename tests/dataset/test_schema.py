# tests/dataset/test_schema.py

import pytest
import tempfile
import yaml
from pathlib import Path
from dataset.schema import load_golden_set, Category, Difficulty, GoldenTrajectoryStep


MINIMAL_YAML = {
    "corpus_hash": "abc123",
    "items": [
        {
            "id": "gold_001",
            "question": "What is attention?",
            "ground_truth": "A weighted sum mechanism.",
            "contexts": [],
            "difficulty": "easy",
            "category": "single_hop",
            "failure_mode_targeted": "sanity",
            "corpus_hash": "abc123",
        },
        {
            "id": "gold_002",
            "question": "Compare BERT and GPT.",
            "ground_truth": "BERT is bidirectional, GPT is causal.",
            "contexts": [],
            "difficulty": "hard",
            "category": "multi_hop",
            "failure_mode_targeted": "multi-hop bait",
            "corpus_hash": "abc123",
            "golden_trajectory": [
                {"goal": "retrieve BERT", "expected_tool": "vector_search"},
                {"goal": "retrieve GPT", "expected_tool": "vector_search"},
                {"goal": "synthesize", "expected_tool": None},
            ],
        },
    ]
}


def test_load_golden_set_parses_items():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(MINIMAL_YAML, f)
        path = f.name
    items = load_golden_set(path)
    assert len(items) == 2
    assert items[0].id == "gold_001"
    assert items[0].category == Category.single_hop
    assert items[0].difficulty == Difficulty.easy
    assert items[0].golden_trajectory is None


def test_load_golden_set_parses_trajectory():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(MINIMAL_YAML, f)
        path = f.name
    items = load_golden_set(path)
    assert items[1].golden_trajectory is not None
    assert len(items[1].golden_trajectory) == 3
    assert items[1].golden_trajectory[0].expected_tool == "vector_search"
    assert items[1].golden_trajectory[2].expected_tool is None
