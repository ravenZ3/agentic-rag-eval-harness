#!/usr/bin/env python3
"""
Intentionally degrades the agent by activating BREAK_PLANNER=1.
Runs multi-hop questions under normal and degraded conditions.
This is Step 4 of the build order — the failure that the eval harness exists to catch.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

MULTI_HOP_QUESTIONS = [
    "How does RLHF differ from supervised fine-tuning in terms of reward modeling?",
    "Compare the retrieval strategies in RAG-Token vs RAG-Sequence models.",
    "What architectural differences between BERT and GPT explain their strengths?",
    "How do LoRA and full fine-tuning differ in their effect on model representations?",
    "What are the tradeoffs between sparse and dense retrieval in RAG systems?",
]

def run_agent(question: str) -> dict:
    # Re-import agent after env change so BREAK_PLANNER is read fresh
    import importlib
    import agent.nodes
    importlib.reload(agent.nodes)
    import agent.graph
    importlib.reload(agent.graph)
    from agent.graph import agent
    from agent.state import AgentState

    initial: AgentState = {
        "question": question,
        "sub_goals": [],
        "current_goal_idx": 0,
        "contexts": [],
        "trajectory": [],
        "final_answer": "",
        "step_count": 0,
        "max_steps": 6,
    }
    return agent.invoke(initial)


def main():
    print("=== NORMAL AGENT ===")
    normal_results = []
    for q in MULTI_HOP_QUESTIONS:
        r = run_agent(q)
        normal_results.append(r)
        print(f"Q: {q[:60]}...")
        print(f"  sub_goals={len(r['sub_goals'])}, steps={r['step_count']}")

    print("\n=== DEGRADED AGENT (BREAK_PLANNER=1) ===")
    os.environ["BREAK_PLANNER"] = "1"
    degraded_results = []
    for q in MULTI_HOP_QUESTIONS:
        r = run_agent(q)
        degraded_results.append(r)
        print(f"Q: {q[:60]}...")
        print(f"  sub_goals={len(r['sub_goals'])}, steps={r['step_count']}")

    print("\n=== COMPARISON (normal_steps vs degraded_steps) ===")
    for q, n, d in zip(MULTI_HOP_QUESTIONS, normal_results, degraded_results):
        delta = n["step_count"] - d["step_count"]
        print(f"  {q[:50]}... | normal={n['step_count']} degraded={d['step_count']} Δ={delta:+d}")

    os.environ.pop("BREAK_PLANNER", None)
    print("\nDone. The degraded agent one-shots multi-hop questions.")
    print("Golden dataset (Task 5) will be built to catch exactly this failure.")


if __name__ == "__main__":
    main()
