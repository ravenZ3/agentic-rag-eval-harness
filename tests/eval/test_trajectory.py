# tests/eval/test_trajectory.py
import pytest
from eval.contracts import TrajectoryRecord, TrajectoryStep
from eval.trajectory import (
    score_trajectory,
    _score_tool_selection,
    _score_retrieval_strategy,
    _score_step_efficiency,
    _score_planning_quality,
)
from dataset.schema import GoldenItem, GoldenTrajectoryStep, Category, Difficulty


def _make_golden(category=Category.multi_hop, with_trajectory=True) -> GoldenItem:
    traj = [
        GoldenTrajectoryStep(goal="retrieve BERT", expected_tool="vector_search"),
        GoldenTrajectoryStep(goal="retrieve GPT", expected_tool="vector_search"),
        GoldenTrajectoryStep(goal="synthesize", expected_tool=None),
    ] if with_trajectory else None
    return GoldenItem(
        id="gold_001",
        question="Compare BERT and GPT.",
        ground_truth="BERT is bidirectional.",
        contexts=["bert_passage", "gpt_passage"],
        difficulty=Difficulty.hard,
        category=category,
        failure_mode_targeted="multi-hop bait",
        corpus_hash="abc123",
        golden_trajectory=traj,
    )


def _make_record(tool_calls: list[str | None], retrieved: list[str]) -> TrajectoryRecord:
    steps = []
    for i, tool in enumerate(tool_calls):
        steps.append(TrajectoryStep(
            thought=f"step {i}",
            tool_called=tool,
            tool_args={"query": "test"} if tool else None,
            tool_result=retrieved[i] if tool and i < len(retrieved) else None,
            observation="done",
        ))
    return TrajectoryRecord(question="Compare BERT and GPT.", steps=steps, final_answer="BERT is bidirectional.")


def test_tool_selection_perfect():
    steps = [
        TrajectoryStep("", "vector_search", {}, "r", ""),
        TrajectoryStep("", "vector_search", {}, "r", ""),
        TrajectoryStep("", None, None, None, ""),
    ]
    golden = [
        GoldenTrajectoryStep("retrieve BERT", "vector_search"),
        GoldenTrajectoryStep("retrieve GPT", "vector_search"),
        GoldenTrajectoryStep("synthesize", None),
    ]
    assert _score_tool_selection(steps, golden) == 1.0


def test_tool_selection_partial():
    steps = [
        TrajectoryStep("", None, None, None, ""),   # wrong: expected vector_search
        TrajectoryStep("", "vector_search", {}, "r", ""),
        TrajectoryStep("", None, None, None, ""),
    ]
    golden = [
        GoldenTrajectoryStep("retrieve BERT", "vector_search"),
        GoldenTrajectoryStep("retrieve GPT", "vector_search"),
        GoldenTrajectoryStep("synthesize", None),
    ]
    assert _score_tool_selection(steps, golden) == pytest.approx(2 / 3)


def test_tool_selection_no_golden():
    assert _score_tool_selection([], []) is None


def test_retrieval_strategy_perfect():
    assert _score_retrieval_strategy(["bert_passage", "gpt_passage"], ["bert_passage", "gpt_passage"]) == 1.0


def test_retrieval_strategy_partial_recall():
    assert _score_retrieval_strategy(["bert_passage"], ["bert_passage", "gpt_passage"]) == 0.5


def test_retrieval_strategy_no_expected():
    assert _score_retrieval_strategy(["anything"], []) == 1.0


def test_step_efficiency_perfect():
    assert _score_step_efficiency(2, 2) == 1.0


def test_step_efficiency_too_many_steps():
    assert _score_step_efficiency(4, 2) == 0.5


def test_step_efficiency_optimal_capped_at_1():
    assert _score_step_efficiency(1, 2) == 1.0


def test_planning_quality_multi_hop_decomposed():
    golden = _make_golden(category=Category.multi_hop, with_trajectory=True)
    steps = [
        TrajectoryStep("Decomposed question into 2 sub-goals", None, None, None, "sub-goals: [...]"),
        TrajectoryStep("retrieve", "vector_search", {}, "bert", ""),
        TrajectoryStep("retrieve", "vector_search", {}, "gpt", ""),
        TrajectoryStep("synthesize", None, None, None, ""),
    ]
    record = TrajectoryRecord(question="Compare BERT and GPT.", steps=steps, final_answer="...")
    assert _score_planning_quality(record, golden) == 1.0


def test_planning_quality_multi_hop_one_shotted():
    golden = _make_golden(category=Category.multi_hop, with_trajectory=True)
    steps = [
        TrajectoryStep("Decomposed question into 1 sub-goals", None, None, None, "sub-goals: [...]"),
        TrajectoryStep("retrieve", "vector_search", {}, "bert+gpt", ""),
        TrajectoryStep("synthesize", None, None, None, ""),
    ]
    record = TrajectoryRecord(question="Compare BERT and GPT.", steps=steps, final_answer="...")
    assert _score_planning_quality(record, golden) == 0.0


def test_planning_quality_single_hop_returns_none():
    golden = _make_golden(category=Category.single_hop, with_trajectory=False)
    steps = [TrajectoryStep("retrieve", "vector_search", {}, "r", "")]
    record = TrajectoryRecord(question="What is attention?", steps=steps, final_answer="...")
    assert _score_planning_quality(record, golden) is None


def test_score_trajectory_full():
    golden = _make_golden()
    steps = [
        TrajectoryStep("Decomposed question into 2 sub-goals", None, None, None, "sub-goals: [...]"),
        TrajectoryStep("retrieve BERT", "vector_search", {}, "bert_passage", "got 4"),
        TrajectoryStep("retrieve GPT", "vector_search", {}, "gpt_passage", "got 4"),
        TrajectoryStep("synthesize", None, None, None, "answer"),
    ]
    record = TrajectoryRecord(question="Compare BERT and GPT.", steps=steps, final_answer="BERT is bidirectional.")
    result = score_trajectory(record, golden, goal_completion_score=4.0)
    assert result.planning_quality == 1.0
    assert result.tool_selection_correctness == 1.0
    assert result.retrieval_strategy_quality == 1.0
    assert result.step_efficiency == 1.0
    assert result.goal_completion == pytest.approx(4.0 / 5.0)
