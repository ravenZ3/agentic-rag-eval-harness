"""Run the full eval suite directly and log to W&B. No API server needed."""
import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval.patches  # noqa: F401 — must be first; applies import-time patches

from dotenv import load_dotenv
load_dotenv()

import argparse
from dataset.schema import load_golden_set
from agent.graph import agent
from agent.state import initial_state
from eval.contracts import AnswerRecord, TrajectoryRecord, TrajectoryStep as EvalStep, OperationalRecord
from eval.checkpoint_io import serialize_checkpoint, load_checkpoint as _load_checkpoint
from eval.ragas_track import run_ragas
from eval.trajectory import score_trajectory
from eval.judge import judge_trajectory
from regression.tracker import log_run, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="baseline")
parser.add_argument("--golden", default="dataset/golden_v1.yaml")
parser.add_argument("--limit", type=int, default=None, help="Only run N questions (faster for testing)")
parser.add_argument("--category", default=None,
                    help="Only run questions of this category (e.g. multi_hop). Applied before --limit.")
parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if one exists")
parser.add_argument("--skip-ragas", action="store_true",
                    help="Skip the (slow/expensive) RAGAS scoring. Trajectory + judge still run. "
                         "RAGAS metrics are left empty; use this when you already have them.")
args = parser.parse_args()

CHECKPOINT_DIR = "data/chkpts"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint_{args.run_name}.json"


def save_checkpoint(answer_records, trajectory_records, ops_records, ragas_result=None):
    data = serialize_checkpoint(answer_records, trajectory_records, ops_records, ragas_result)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return [], [], [], None
    return _load_checkpoint(CHECKPOINT_PATH)


golden_items = load_golden_set(args.golden)
if args.category:
    golden_items = [g for g in golden_items if g.category.value == args.category]
if args.limit:
    golden_items = golden_items[:args.limit]

# Full golden set, kept intact for scoring. `pending_items` is the subset
# still needing an agent run; on full resume this is empty but golden_items
# stays complete so trajectory scoring still has its golden references.
pending_items = golden_items

# Resume from checkpoint if requested and available
if args.resume:
    answer_records, trajectory_records, ops_records, ragas_result = load_checkpoint()
    completed = {r.question for r in answer_records}
    pending_items = [g for g in golden_items if g.question not in completed]
    print(f"Resuming: {len(completed)} already done, {len(pending_items)} remaining")
    if ragas_result is not None:
        print("RAGAS results loaded from checkpoint — skipping RAGAS scoring")
else:
    answer_records, trajectory_records, ops_records, ragas_result = [], [], [], None

print(f"Running eval on {len(pending_items)} golden items (run: {args.run_name})")

for i, item in enumerate(pending_items):
    print(f"  [{i+1}/{len(golden_items)}] {item.question[:60]}...")
    if i > 0:
        time.sleep(20)  # pace Groq TPM — 6k tokens/min; synthesizer bursts ~3500 tokens
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

if ragas_result is not None:
    print("Skipping RAGAS (loaded from checkpoint)")
elif args.skip_ragas:
    from eval.contracts import RagasResult
    nan = float("nan")
    ragas_result = RagasResult(nan, nan, nan, nan, per_item=[])
    print("Skipping RAGAS (--skip-ragas); RAGAS metrics will be empty")
else:
    print("Scoring with RAGAS...")
    ragas_result = run_ragas(answer_records)
    save_checkpoint(answer_records, trajectory_records, ops_records, ragas_result)

print("Running LLM judge on trajectories...")
judge_results = [judge_trajectory(r, n_samples=3) for r in trajectory_records]

golden_by_question = {g.question: g for g in golden_items}
traj_scores = [
    score_trajectory(r, golden_by_question[r.question], float(j.goal_completion))
    for r, j in zip(trajectory_records, judge_results)
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

# Preserve checkpoint on success (renamed) so trajectories/answers can be
# re-scored offline without re-running the agent. A fresh (non-resume) run
# overwrites CHECKPOINT_PATH anyway, so this never blocks a clean re-run.
if os.path.exists(CHECKPOINT_PATH):
    done_path = CHECKPOINT_PATH.replace(".json", ".done.json")
    os.replace(CHECKPOINT_PATH, done_path)
    print(f"Checkpoint preserved: {done_path}")
