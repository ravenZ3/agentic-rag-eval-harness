from typing import TypedDict, Optional
from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    thought: str
    tool_called: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[str]
    observation: str


class AgentState(TypedDict):
    question: str
    sub_goals: list[str]
    current_goal_idx: int
    contexts: list[str]
    trajectory: list[TrajectoryStep]
    final_answer: str
    step_count: int
    max_steps: int


def initial_state(question: str, max_steps: int = 6) -> AgentState:
    return AgentState(
        question=question,
        sub_goals=[],
        current_goal_idx=0,
        contexts=[],
        trajectory=[],
        final_answer="",
        step_count=0,
        max_steps=max_steps,
    )
