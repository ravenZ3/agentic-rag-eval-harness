from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnswerRecord:
    """Contract for RAGAS evaluation (Track A). Eval engine consumes this; agent produces it."""
    question: str
    answer: str
    contexts: list[str]        # passages actually retrieved by agent
    ground_truth: str


@dataclass
class TrajectoryStep:
    """One step in an agent trajectory. Mirrors agent.state.TrajectoryStep but owned by eval."""
    thought: str
    tool_called: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[str]
    observation: str


@dataclass
class TrajectoryRecord:
    """Contract for trajectory scoring (Track B) and judge (Track C)."""
    question: str
    steps: list[TrajectoryStep]
    final_answer: str


@dataclass
class JudgeResult:
    safety: int           # 1–5
    tone: int             # 1–5
    hallucination: int    # 1–5 (5 = no hallucination)
    reasoning: str
    failing_claims: list[str]
    variance: dict[str, float]  # run-to-run std dev per dimension
    goal_completion: int = 3    # added in Task 8 / gap fixed vs original plan


@dataclass
class RagasResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    per_item: list[dict]   # per-question breakdown


@dataclass
class TrajectoryScore:
    planning_quality: Optional[float]       # None = judge-scored (no golden trajectory)
    tool_selection_correctness: Optional[float]
    retrieval_strategy_quality: float       # recall of expected contexts
    step_efficiency: Optional[float]        # optimal_steps / actual_steps
    goal_completion: float                  # judge score 1–5, normalized to 0–1
    judge_used_for: list[str]               # which dimensions used judge fallback
