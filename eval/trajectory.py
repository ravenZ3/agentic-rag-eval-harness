import numpy as np
from typing import Optional
from eval.contracts import TrajectoryRecord, TrajectoryScore
from dataset.schema import GoldenItem


def _score_tool_selection(
    steps: list,
    golden_steps: list,
) -> Optional[float]:
    """Fraction of steps where tool_called matches expected_tool. None if no golden trajectory."""
    if not golden_steps:
        return None
    # Filter out planning steps from actual steps to align with golden trajectory
    non_planning_steps = [
        s for s in steps
        if not (s.tool_called is None and any(word in s.thought.lower() for word in ["sub-goal", "plan", "decomposed"]))
    ]
    tool_steps = [(s.tool_called, g.expected_tool) for s, g in zip(non_planning_steps, golden_steps)]
    if not tool_steps:
        return 0.0
    correct = sum(1 for actual, expected in tool_steps if actual == expected)
    return correct / len(golden_steps)



def _score_retrieval_strategy(
    retrieved_contexts: list[str],
    expected_contexts: list[str],
) -> float:
    """Recall: fraction of expected contexts that appear in retrieved contexts."""
    if not expected_contexts:
        return 1.0  # nothing expected = no retrieval failure
    retrieved_set = set(retrieved_contexts)
    expected_set = set(expected_contexts)
    intersection = retrieved_set & expected_set
    return len(intersection) / len(expected_set)


def _score_step_efficiency(actual_steps: int, optimal_steps: int) -> Optional[float]:
    """optimal/actual — 1.0 means perfect efficiency, lower = too many steps."""
    if actual_steps == 0 or optimal_steps == 0:
        return None
    return min(1.0, optimal_steps / actual_steps)


def _score_planning_quality(record: TrajectoryRecord, golden: GoldenItem) -> Optional[float]:
    """
    Heuristic: for multi_hop/adversarial, check whether the agent decomposed into ≥2 sub-goals.
    Returns 1.0 if it decomposed as expected, 0.0 if it one-shotted. None for single_hop.
    For cases with a golden_trajectory, checks if planned step count matches golden step count.
    """
    from dataset.schema import Category
    if golden.category == Category.single_hop:
        return None  # single-hop one-shotting is correct behavior

    # Count planner steps (tool_called is None) in trajectory
    planner_step = next(
        (s for s in record.steps if s.tool_called is None and "sub-goal" in s.thought.lower()),
        None,
    )
    retrieval_steps = [s for s in record.steps if s.tool_called == "vector_search"]

    if golden.golden_trajectory:
        expected_retrieval_count = sum(
            1 for s in golden.golden_trajectory if s.expected_tool == "vector_search"
        )
        return 1.0 if len(retrieval_steps) >= expected_retrieval_count else 0.0

    # No golden trajectory: multi-hop/adversarial should have ≥2 retrieval steps
    return 1.0 if len(retrieval_steps) >= 2 else 0.0


def score_trajectory(
    record: TrajectoryRecord,
    golden: GoldenItem,
    goal_completion_score: float = 0.5,   # injected from judge; default neutral
) -> TrajectoryScore:
    """
    Score a trajectory against its golden item.
    goal_completion_score comes from the LLM judge (Track C) and is injected here.
    """
    retrieved_contexts = [
        s.tool_result for s in record.steps
        if s.tool_called == "vector_search" and s.tool_result
    ]

    golden_steps = golden.golden_trajectory or []
    tool_selection = _score_tool_selection(record.steps, golden_steps)
    retrieval_quality = _score_retrieval_strategy(retrieved_contexts, golden.contexts)
    optimal_steps = len(golden_steps) if golden_steps else None
    actual_retrieval_steps = len([s for s in record.steps if s.tool_called == "vector_search"])
    efficiency = _score_step_efficiency(actual_retrieval_steps, optimal_steps) if optimal_steps else None
    planning = _score_planning_quality(record, golden)

    judge_used_for = []
    if tool_selection is None:
        judge_used_for.append("tool_selection_correctness")
    if planning is None:
        judge_used_for.append("planning_quality")
    if efficiency is None:
        judge_used_for.append("step_efficiency")

    return TrajectoryScore(
        planning_quality=planning,
        tool_selection_correctness=tool_selection,
        retrieval_strategy_quality=retrieval_quality,
        step_efficiency=efficiency,
        goal_completion=goal_completion_score / 5.0,  # normalize judge 1–5 to 0–1
        judge_used_for=judge_used_for,
    )
