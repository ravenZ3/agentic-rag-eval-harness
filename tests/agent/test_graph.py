# tests/agent/test_graph.py
import pytest
from unittest.mock import patch, MagicMock
from agent.state import AgentState, TrajectoryStep
from agent.nodes import planner_node, retriever_node, synthesizer_node, should_continue


MINIMAL_STATE: AgentState = {
    "question": "What is attention in transformers?",
    "sub_goals": [],
    "current_goal_idx": 0,
    "contexts": [],
    "trajectory": [],
    "final_answer": "",
    "step_count": 0,
    "max_steps": 6,
}


def test_planner_returns_sub_goals():
    mock_llm_response = MagicMock()
    mock_llm_response.content = '["What is attention?", "transformer architecture"]'
    with patch("agent.nodes._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_llm_response
        result = planner_node(MINIMAL_STATE)
    assert isinstance(result["sub_goals"], list)
    assert len(result["sub_goals"]) >= 1
    assert result["current_goal_idx"] == 0
    assert len(result["trajectory"]) == 1


def test_planner_handles_invalid_json():
    mock_llm_response = MagicMock()
    mock_llm_response.content = "not valid json"
    with patch("agent.nodes._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_llm_response
        result = planner_node(MINIMAL_STATE)
    # Falls back to original question as single sub-goal
    assert result["sub_goals"] == [MINIMAL_STATE["question"]]


def test_retriever_appends_contexts():
    state = {**MINIMAL_STATE, "sub_goals": ["attention mechanism"], "current_goal_idx": 0}
    mock_results = ["passage A", "passage B"]
    with patch("agent.tools.vector_search") as mock_tool:
        mock_tool.invoke.return_value = mock_results
        result = retriever_node(state)
    assert result["contexts"] == mock_results
    assert result["current_goal_idx"] == 1
    assert result["step_count"] == 1
    step = result["trajectory"][-1]
    assert step.tool_called == "vector_search"


def test_should_continue_synthesize_when_goals_exhausted():
    state = {**MINIMAL_STATE, "sub_goals": ["goal1"], "current_goal_idx": 1, "step_count": 1}
    assert should_continue(state) == "synthesize"


def test_should_continue_retrieve_when_goals_remain():
    state = {**MINIMAL_STATE, "sub_goals": ["goal1", "goal2"], "current_goal_idx": 0, "step_count": 0}
    assert should_continue(state) == "retrieve"


def test_should_continue_synthesize_when_max_steps_hit():
    state = {**MINIMAL_STATE, "sub_goals": ["g1", "g2", "g3"], "current_goal_idx": 1, "step_count": 6}
    assert should_continue(state) == "synthesize"


def test_synthesizer_returns_answer():
    state = {**MINIMAL_STATE, "contexts": ["passage A"], "trajectory": []}
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Attention is a mechanism that..."
    with patch("agent.nodes._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_llm_response
        result = synthesizer_node(state)
    assert result["final_answer"] == "Attention is a mechanism that..."
    assert len(result["trajectory"]) == 1


def test_break_planner_forces_single_subgoal(monkeypatch):
    monkeypatch.setenv("BREAK_PLANNER", "1")
    mock_llm_response = MagicMock()
    mock_llm_response.content = '["What is attention in transformers?"]'
    with patch("agent.nodes._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_llm_response
        result = planner_node({**MINIMAL_STATE, "question": "Compare attention in BERT vs GPT."})
    # With degraded planner, should return exactly 1 sub-goal
    assert len(result["sub_goals"]) == 1
