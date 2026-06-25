import json
import os
import re
import time
from groq import RateLimitError
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState, TrajectoryStep
from config import AGENT_MODEL, AGENT_TEMPERATURE, RETRIEVAL_K

_llm = ChatGroq(model=AGENT_MODEL, temperature=AGENT_TEMPERATURE)

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research assistant planning how to answer a question about ML research. "
        "Break the question into 1-3 specific retrieval sub-goals. "
        "For multi-part questions (comparing two things, explaining differences, covering multiple concepts), "
        "always produce at least 2 sub-goals. "
        "Return ONLY a valid JSON array of strings, e.g. [\"sub-goal 1\", \"sub-goal 2\"]. "
        "No markdown, no explanation, no thinking. Output the JSON array immediately."
    )),
    ("human", "Question: {question}"),
])

# Degraded prompt activated by BREAK_PLANNER=1 — forces one-shot behavior on multi-hop questions
_DEGRADED_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Answer questions directly. Always return the original question as the only sub-goal. "
        "Return ONLY a valid JSON array with one string."
    )),
    ("human", "Question: {question}"),
])

_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a research assistant. Answer the question using the provided contexts. "
        "The contexts may each cover different parts of the answer — synthesize across "
        "them to form one coherent answer. Partial or distributed evidence is acceptable; "
        "combine what is present. "
        "Refuse ONLY if the contexts contain no relevant information at all. "
        "If you refuse, reply exactly: 'I cannot determine this from the available information.'"
    )),
    ("human", "Question: {question}\n\nContexts:\n{contexts}"),
])


def _strip_thinking(text: str) -> str:
    """Qwen3 prepends <think>...</think> blocks before actual output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def planner_node(state: AgentState) -> dict:
    prompt = _DEGRADED_PLANNER_PROMPT if os.getenv("BREAK_PLANNER") else _PLANNER_PROMPT
    response = _llm.invoke(prompt.format_messages(question=state["question"]))
    content = _strip_thinking(response.content)
    try:
        sub_goals = json.loads(content)
        if not isinstance(sub_goals, list):
            sub_goals = [state["question"]]
    except (json.JSONDecodeError, ValueError):
        sub_goals = [state["question"]]

    step = TrajectoryStep(
        thought=f"Decomposed question into {len(sub_goals)} sub-goal(s)",
        tool_called=None,
        tool_args=None,
        tool_result=None,
        observation=f"Sub-goals: {sub_goals}",
    )
    return {
        "sub_goals": sub_goals,
        "current_goal_idx": 0,
        "contexts": [],
        "trajectory": [step],
        "step_count": 0,
    }


def retriever_node(state: AgentState) -> dict:
    from agent.tools import vector_search
    goal = state["sub_goals"][state["current_goal_idx"]]
    results = vector_search.invoke({"query": goal})

    step = TrajectoryStep(
        thought=f"Retrieving for sub-goal: {goal}",
        tool_called="vector_search",
        tool_args={"query": goal, "k": RETRIEVAL_K},
        tool_result=str(results[:2]),  # truncate for trace readability
        observation=f"Retrieved {len(results)} passages",
    )
    return {
        "contexts": state["contexts"] + results,
        "trajectory": state["trajectory"] + [step],
        "current_goal_idx": state["current_goal_idx"] + 1,
        "step_count": state["step_count"] + 1,
    }


def should_continue(state: AgentState) -> str:
    """Router: keep retrieving sub-goals or move to synthesis."""
    if state["current_goal_idx"] >= len(state["sub_goals"]):
        return "synthesize"
    if state["step_count"] >= state["max_steps"]:
        return "synthesize"
    return "retrieve"


def _invoke_with_retry(prompt, max_retries: int = 5, base_wait: float = 2.0):
    for attempt in range(max_retries):
        try:
            return _llm.invoke(prompt)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            time.sleep(wait)


def synthesizer_node(state: AgentState) -> dict:
    contexts_text = "\n\n---\n\n".join(state["contexts"])
    response = _invoke_with_retry(_SYNTHESIZER_PROMPT.format_messages(
        question=state["question"],
        contexts=contexts_text,
    ))
    answer = _strip_thinking(response.content)
    step = TrajectoryStep(
        thought="Synthesizing final answer from all retrieved contexts",
        tool_called=None,
        tool_args=None,
        tool_result=None,
        observation=f"Answer length: {len(answer)} chars",
    )
    return {
        "final_answer": answer,
        "trajectory": state["trajectory"] + [step],
    }
