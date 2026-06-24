import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Eval Harness", version="0.1.0")

_runs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)


class AnswerRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


class TrajectoryRequest(BaseModel):
    question: str
    final_answer: str
    steps: list[dict]


class RunRequest(BaseModel):
    golden_path: str = "dataset/golden_v1.yaml"
    run_name: str | None = None


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate_answer(req: AnswerRequest) -> dict:
    """Synchronous: RAGAS on a single answer record."""
    from eval.contracts import AnswerRecord
    from eval.ragas_track import run_ragas
    record = AnswerRecord(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    result = run_ragas([record])
    return {
        "faithfulness": result.faithfulness,
        "answer_relevancy": result.answer_relevancy,
        "context_precision": result.context_precision,
        "context_recall": result.context_recall,
    }


@app.post("/evaluate-trajectory")
def evaluate_trajectory(req: TrajectoryRequest) -> dict:
    """Synchronous: judge on a single trajectory."""
    from eval.contracts import TrajectoryRecord, TrajectoryStep as EvalStep
    from eval.judge import judge_trajectory
    steps = [EvalStep(**s) for s in req.steps]
    record = TrajectoryRecord(question=req.question, steps=steps, final_answer=req.final_answer)
    judge = judge_trajectory(record, n_samples=1)
    return {
        "safety": judge.safety,
        "tone": judge.tone,
        "hallucination": judge.hallucination,
        "goal_completion": judge.goal_completion,
        "reasoning": judge.reasoning,
        "variance": judge.variance,
    }


@app.post("/run")
def start_run(req: RunRequest, background_tasks: BackgroundTasks) -> dict:
    """Async: full eval suite on golden set. Poll /runs/{id} for results."""
    run_id = str(uuid.uuid4())
    _runs[run_id] = {"status": "running", "result": None}
    background_tasks.add_task(_execute_full_run, run_id, req.golden_path, req.run_name)
    return {"run_id": run_id, "status": "running"}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]


def _execute_full_run(run_id: str, golden_path: str, run_name: str | None) -> None:
    try:
        from dataset.schema import load_golden_set
        from agent.graph import agent
        from agent.state import AgentState
        from eval.contracts import AnswerRecord, TrajectoryRecord, TrajectoryStep as EvalStep
        from eval.ragas_track import run_ragas
        from eval.trajectory import score_trajectory
        from eval.judge import judge_trajectory
        from regression.tracker import log_run, compute_metrics
        from regression.gate import check_regression

        golden_items = load_golden_set(golden_path)
        answer_records, trajectory_records = [], []

        for item in golden_items:
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

        ragas_result = run_ragas(answer_records)
        judge_results = [judge_trajectory(r, n_samples=3) for r in trajectory_records]
        traj_scores = [
            score_trajectory(r, g, float(j.goal_completion))
            for r, g, j in zip(trajectory_records, golden_items, judge_results)
        ]

        wandb_run_id = log_run(
            ragas=ragas_result,
            traj_scores=traj_scores,
            judge_results=judge_results,
            config={"golden_path": golden_path},
            golden_path=golden_path,
            run_name=run_name,
        )

        run_metrics = compute_metrics(ragas_result, traj_scores, judge_results)
        _runs[run_id] = {
            "status": "complete",
            "wandb_run_id": wandb_run_id,
            "result": run_metrics,
        }
    except Exception as e:
        _runs[run_id] = {"status": "error", "error": str(e)}
