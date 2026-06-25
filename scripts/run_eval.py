"""Run the full eval suite directly and log to W&B. No API server needed."""
import sys
import os
import time
import json
import dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval.patches  # noqa: F401 — must be first; applies import-time patches

from dotenv import load_dotenv
load_dotenv()

import argparse
from dataset.schema import load_golden_set
from agent.graph import agent
from agent.state import initial_state
from eval.contracts import AnswerRecord, TrajectoryRecord, TrajectoryStep as EvalStep, OperationalRecord
from eval.ragas_track import run_ragas
from eval.trajectory import score_trajectory
from eval.judge import judge_trajectory
from regression.tracker import log_run, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="baseline")
parser.add_argument("--golden", default="dataset/golden_v1.yaml")
parser.add_argument("--limit", type=int, default=None, help="Only run N questions (faster for testing)")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if one exists")
args = parser.parse_args()

CHECKPOINT_PATH = f"data/checkpoint_{args.run_name}.json"


def save_checkpoint(answer_records, trajectory_records, ops_records):
    os.makedirs("data", exist_ok=True)
    data = {
        "answer_records": [dataclasses.asdict(r) for r in answer_records],
        "trajectory_records": [
            {
                "question": r.question,
                "final_answer": r.final_answer,
                "steps": [dataclasses.asdict(s) for s in r.steps],
            }
            for r in trajectory_records
        ],
        "ops_records": [dataclasses.asdict(r) for r in ops_records],
    }
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return [], [], []
    with open(CHECKPOINT_PATH) as f:
        data = json.load(f)
    answer_records = [AnswerRecord(**r) for r in data["answer_records"]]
    trajectory_records = [
        TrajectoryRecord(
            question=r["question"],
            final_answer=r["final_answer"],
            steps=[EvalStep(**s) for s in r["steps"]],
        )
        for r in data["trajectory_records"]
    ]
    ops_records = [OperationalRecord(**r) for r in data.get("ops_records", [])]
    return answer_records, trajectory_records, ops_records


golden_items = load_golden_set(args.golden)
if args.limit:
    golden_items = golden_items[:args.limit]

# Resume from checkpoint if requested and available
if args.resume:
    answer_records, trajectory_records, ops_records = load_checkpoint()
    completed = {r.question for r in answer_records}
    golden_items = [g for g in golden_items if g.question not in completed]
    print(f"Resuming: {len(completed)} already done, {len(golden_items)} remaining")
else:
    answer_records, trajectory_records, ops_records = [], [], []

print(f"Running eval on {len(golden_items)} golden items (run: {args.run_name})")

for i, item in enumerate(golden_items):
    print(f"  [{i+1}/{len(golden_items)}] {item.question[:60]}...")
    if i > 0:
        time.sleep(10)  # pace Groq TPM — 6k tokens/min shared across all calls
    t0 = time.time()
    result = agent.invoke(initial_state(item.question))
    latency_ms = (time.time() - t0) * 1000

    contexts = result["contexts"]
    answer = result["final_answer"] or ""
    steps = result["trajectory"]

    answer_records.append(AnswerRecord(
        question=item.question,
        answer=answer,
        contexts=contexts,
        ground_truth=item.ground_truth,
    ))
    trajectory_records.append(TrajectoryRecord(
        question=item.question,
        steps=[EvalStep(**vars(s)) for s in steps],
        final_answer=answer,
    ))
    ops_records.append(OperationalRecord(
        question=item.question,
        latency_ms=latency_ms,
        steps_taken=len(steps),
        contexts_retrieved=len(contexts),
        total_context_chars=sum(len(c) for c in contexts),
        answer_length_chars=len(answer),
    ))
    save_checkpoint(answer_records, trajectory_records, ops_records)

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
    ops_records=ops_records,
    config={"golden_path": args.golden, "run_name": args.run_name},
    golden_path=args.golden,
    run_name=args.run_name,
)

if wandb_run_id:
    print(f"\nDone. W&B run ID: {wandb_run_id}")
else:
    print("\nDone. (W&B logging skipped/failed — metrics below are still valid.)")

print("\nMetrics:")
metrics = compute_metrics(ragas_result, traj_scores, judge_results, ops_records)
for key in sorted(metrics):
    print(f"  {key:<42} {metrics[key]:.3f}")

# Clean up checkpoint on successful completion
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print(f"Checkpoint removed: {CHECKPOINT_PATH}")
