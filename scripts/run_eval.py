"""Run the full eval suite directly and log to W&B. No API server needed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import argparse
from dataset.schema import load_golden_set
from agent.graph import agent
from agent.state import AgentState
from eval.contracts import AnswerRecord, TrajectoryRecord, TrajectoryStep as EvalStep
from eval.ragas_track import run_ragas
from eval.trajectory import score_trajectory
from eval.judge import judge_trajectory
from regression.tracker import log_run, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="baseline")
parser.add_argument("--golden", default="dataset/golden_v1.yaml")
parser.add_argument("--limit", type=int, default=None, help="Only run N questions (faster for testing)")
args = parser.parse_args()

golden_items = load_golden_set(args.golden)
if args.limit:
    golden_items = golden_items[:args.limit]

print(f"Running eval on {len(golden_items)} golden items (run: {args.run_name})")

answer_records, trajectory_records = [], []

for i, item in enumerate(golden_items):
    print(f"  [{i+1}/{len(golden_items)}] {item.question[:60]}...")
    initial: AgentState = {
        "question": item.question,
        "sub_goals": [],
        "current_goal_idx": 0,
        "contexts": [],
        "trajectory": [],
        "final_answer": "",
        "step_count": 0,
        "max_steps": 6,
    }
    result = agent.invoke(initial)
    answer_records.append(AnswerRecord(
        question=item.question,
        answer=result["final_answer"],
        contexts=result["contexts"],
        ground_truth=item.ground_truth,
    ))
    trajectory_records.append(TrajectoryRecord(
        question=item.question,
        steps=[EvalStep(**vars(s)) for s in result["trajectory"]],
        final_answer=result["final_answer"],
    ))

print("Scoring with RAGAS...")
ragas_result = run_ragas(answer_records)

print("Running LLM judge on trajectories...")
judge_results = [judge_trajectory(r, n_samples=3) for r in trajectory_records]

traj_scores = [
    score_trajectory(r, g, float(j.goal_completion))
    for r, g, j in zip(trajectory_records, golden_items, judge_results)
]

print("Logging to W&B...")
wandb_run_id = log_run(
    ragas=ragas_result,
    traj_scores=traj_scores,
    judge_results=judge_results,
    config={"golden_path": args.golden, "run_name": args.run_name},
    golden_path=args.golden,
    run_name=args.run_name,
)

if wandb_run_id:
    print(f"\nDone. W&B run ID: {wandb_run_id}")
else:
    print("\nDone. (W&B logging skipped/failed — metrics below are still valid.)")

print("\nMetrics:")
metrics = compute_metrics(ragas_result, traj_scores, judge_results)
for key in sorted(metrics):
    print(f"  {key:<42} {metrics[key]:.3f}")
