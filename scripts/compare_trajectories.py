"""Offline trajectory comparison: baseline vs degraded.

Computes the structural trajectory metrics (planning_quality,
tool_selection_correctness, retrieval_strategy_quality, step_efficiency)
directly from saved checkpoints — NO API calls. goal_completion is the only
metric that needs the judge, so it is reported separately / skipped here.

This is the regression-story core: it shows the planner-collapse signal that
answer-level RAGAS metrics miss, without spending a cent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.schema import load_golden_set
from eval.checkpoint_io import load_trajectory_records
from eval.trajectory import score_trajectory


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def score_set(path, golden_by_q):
    records = load_trajectory_records(path)
    # goal_completion is judge-injected; pass neutral 0.5 since we only care
    # about the structural metrics here.
    scores = [score_trajectory(r, golden_by_q[r.question], 2.5) for r in records]
    return {
        "planning_quality": mean([s.planning_quality for s in scores]),
        "tool_selection_correctness": mean([s.tool_selection_correctness for s in scores]),
        "retrieval_strategy_quality": mean([s.retrieval_strategy_quality for s in scores]),
        "step_efficiency": mean([s.step_efficiency for s in scores]),
        "n": len(scores),
    }


def main():
    golden = load_golden_set("dataset/golden_v1.yaml")
    golden_by_q = {g.question: g for g in golden}

    baseline_path = "data/chkpts/checkpoint_baseline-traj.done.json"
    # degraded full run (k=2) was superseded by the methodology fix and lives in stale/
    degraded_path = "data/stale/checkpoint_degraded.json"
    if not os.path.exists(degraded_path):
        degraded_path = "data/stale/checkpoint_degraded.done.json"

    base = score_set(baseline_path, golden_by_q)
    deg = score_set(degraded_path, golden_by_q)

    print(f"\n{'Trajectory metric':<32} {'Baseline':>10} {'Degraded':>10} {'Delta':>10}")
    print("-" * 64)
    for key in ["planning_quality", "tool_selection_correctness",
                "retrieval_strategy_quality", "step_efficiency"]:
        b, d = base[key], deg[key]
        if b is None or d is None:
            print(f"{key:<32} {str(b):>10} {str(d):>10} {'n/a':>10}")
        else:
            print(f"{key:<32} {b:>10.3f} {d:>10.3f} {d - b:>+10.3f}")
    print("-" * 64)
    print(f"{'n questions':<32} {base['n']:>10} {deg['n']:>10}")


if __name__ == "__main__":
    main()
