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
