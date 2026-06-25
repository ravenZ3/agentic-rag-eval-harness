"""Offline RAGAS scoring from a saved checkpoint — no agent re-run.

Loads the answers/contexts already captured in a checkpoint and runs RAGAS on
them using whatever SCORING_PROVIDER is set in config.py. Use this to score
both baseline and degraded on the SAME answers + SAME judge model, so the
baseline-vs-degraded RAGAS comparison is clean.

Usage:
    python scripts/score_ragas_offline.py data/chkpts/checkpoint_baseline-traj.done.json
    python scripts/score_ragas_offline.py data/stale/checkpoint_degraded.json
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval.patches  # noqa: F401
from dotenv import load_dotenv
load_dotenv()

from eval.contracts import AnswerRecord
from eval.ragas_track import run_ragas
from config import SCORING_PROVIDER, RAGAS_LLM_MODEL


def main():
    if len(sys.argv) < 2:
        print("usage: score_ragas_offline.py <checkpoint.json>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)

    answers = [AnswerRecord(**r) for r in data["answer_records"]]
    print(f"Scoring {len(answers)} answers from {path}")
    print(f"Provider: {SCORING_PROVIDER}  model: {RAGAS_LLM_MODEL}")

    result = run_ragas(answers)

    out = {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
    }
    print("\nRAGAS results:")
    for k, v in out.items():
        print(f"  ragas/{k:<20} {v:.3f}")

    # Persist alongside the checkpoint so it can't be lost.
    out_path = path.replace(".json", f".ragas_{SCORING_PROVIDER}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
