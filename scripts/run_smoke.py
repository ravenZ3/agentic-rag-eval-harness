#!/usr/bin/env python3
"""Run agent on 10 questions and print trajectory. Step 3 of build order."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import eval.patches  # noqa: F401 — must be first; applies import-time patches

from dotenv import load_dotenv
load_dotenv()

from agent.graph import agent
from agent.state import initial_state

QUESTIONS = [
    # single-hop
    "What is the transformer attention mechanism?",
    "What is LoRA fine-tuning?",
    "How does RLHF work?",
    "What is retrieval-augmented generation?",
    "What is knowledge distillation?",
    # multi-hop (should require 2 retrieval steps)
    "How does RLHF differ from supervised fine-tuning in terms of reward modeling?",
    "What are the tradeoffs between LoRA and full fine-tuning for large language models?",
    # unanswerable (no paper in corpus covers this)
    "What was the weather in San Francisco on January 1, 2020?",
    # adversarial (designed to bait one-shot multi-hop)
    "Compare the retrieval strategies in RAG-Token vs RAG-Sequence models, and explain when each is preferred.",
    "What architectural differences between BERT and GPT explain their different strengths in generation vs understanding tasks?",
]

def run_question(q: str) -> None:
    result = agent.invoke(initial_state(q))
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"Sub-goals: {result['sub_goals']}")
    print(f"Steps taken: {result['step_count']}")
    print(f"Contexts retrieved: {len(result['contexts'])}")
    print(f"A: {result['final_answer'][:300]}...")
    print(f"Trajectory steps:")
    for i, step in enumerate(result["trajectory"]):
        print(f"  [{i}] tool={step.tool_called} | {step.thought}")

if __name__ == "__main__":
    print("NOTE: Ensure corpus is ingested first (scripts/ingest_corpus.py)")
    for q in QUESTIONS:
        run_question(q)
