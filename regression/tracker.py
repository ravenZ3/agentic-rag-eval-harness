import os
import math
import hashlib
from pathlib import Path

import wandb

from eval.contracts import RagasResult, TrajectoryScore, JudgeResult


def _hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def _clean(value: float | None) -> float | None:
    """Drop NaN/inf — RAGAS returns NaN when a metric can't be computed.
    Logging NaN to W&B pollutes charts and silently breaks the regression gate."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _mean(values: list) -> float | None:
    """Mean over non-None, finite values. Returns None for an empty set
    instead of raising ZeroDivisionError."""
    filtered = [_clean(v) for v in values]
    filtered = [v for v in filtered if v is not None]
    return float(sum(filtered) / len(filtered)) if filtered else None


def _wandb_available() -> bool:
    """True if W&B can run without blocking on an interactive login prompt.
    Offline/disabled modes are always fine; online mode needs an API key."""
    mode = os.getenv("WANDB_MODE", "online").lower()
    if mode in ("offline", "disabled", "dryrun"):
        return True
    return bool(os.getenv("WANDB_API_KEY"))


def compute_metrics(
    ragas: RagasResult,
    traj_scores: list[TrajectoryScore],
    judge_results: list[JudgeResult],
) -> dict[str, float]:
    """Aggregate raw eval outputs into the flat metric dict that gets logged
    and fed to the regression gate. NaN-safe and empty-safe; never raises."""
    metrics = {
        "ragas/faithfulness": _clean(ragas.faithfulness),
        "ragas/answer_relevancy": _clean(ragas.answer_relevancy),
        "ragas/context_precision": _clean(ragas.context_precision),
        "ragas/context_recall": _clean(ragas.context_recall),
        "trajectory/planning_quality": _mean([s.planning_quality for s in traj_scores]),
        "trajectory/tool_selection_correctness": _mean(
            [s.tool_selection_correctness for s in traj_scores]
        ),
        "trajectory/retrieval_strategy_quality": _mean(
            [s.retrieval_strategy_quality for s in traj_scores]
        ),
        "trajectory/step_efficiency": _mean([s.step_efficiency for s in traj_scores]),
        "trajectory/goal_completion": _mean([s.goal_completion for s in traj_scores]),
        "judge/safety": _mean([j.safety for j in judge_results]),
        "judge/tone": _mean([j.tone for j in judge_results]),
        "judge/hallucination": _mean([j.hallucination for j in judge_results]),
        "judge/goal_completion_raw": _mean([j.goal_completion for j in judge_results]),
    }
    # Strip keys whose value couldn't be computed — never log None/NaN.
    return {k: v for k, v in metrics.items() if v is not None}


def _build_per_item_table(
    ragas: RagasResult,
    traj_scores: list[TrajectoryScore],
    judge_results: list[JudgeResult],
):
    """A W&B Table for per-question drill-down in the dashboard."""
    columns = [
        "idx",
        "judge_safety",
        "judge_tone",
        "judge_hallucination",
        "judge_goal_completion",
        "traj_goal_completion",
        "traj_planning_quality",
        "traj_tool_selection",
        "traj_retrieval_quality",
        "failing_claims",
    ]
    rows = []
    n = max(len(traj_scores), len(judge_results))
    for i in range(n):
        t = traj_scores[i] if i < len(traj_scores) else None
        j = judge_results[i] if i < len(judge_results) else None
        rows.append([
            i,
            j.safety if j else None,
            j.tone if j else None,
            j.hallucination if j else None,
            j.goal_completion if j else None,
            t.goal_completion if t else None,
            t.planning_quality if t else None,
            t.tool_selection_correctness if t else None,
            t.retrieval_strategy_quality if t else None,
            "; ".join(j.failing_claims) if j and j.failing_claims else "",
        ])
    return wandb.Table(columns=columns, data=rows)


def log_run(
    ragas: RagasResult,
    traj_scores: list[TrajectoryScore],
    judge_results: list[JudgeResult],
    config: dict,
    golden_path: str | Path = "dataset/golden_v1.yaml",
    run_name: str | None = None,
) -> str | None:
    """Log a full evaluation run to W&B. Returns the W&B run ID, or None if
    logging was skipped/failed. Scoring work is never lost to a W&B outage —
    failures here are caught and surfaced, not propagated."""
    metrics = compute_metrics(ragas, traj_scores, judge_results)

    if not _wandb_available():
        print(
            "[tracker] WANDB_API_KEY not set and WANDB_MODE is online — "
            "skipping W&B logging to avoid an interactive login hang.\n"
            "          Set WANDB_API_KEY, or WANDB_MODE=offline to log locally.\n"
            f"[tracker] Computed metrics: {metrics}"
        )
        return None

    golden_hash = _hash_file(golden_path)
    run = None
    try:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "rag-eval-harness"),
            name=run_name,
            config={**config, "golden_hash": golden_hash},
            reinit="finish_previous",
        )
        run.log(metrics)
        try:
            run.log({"per_item": _build_per_item_table(ragas, traj_scores, judge_results)})
        except Exception as e:  # table is a nice-to-have, never fatal
            print(f"[tracker] per-item table logging skipped: {e}")
        return run.id
    except Exception as e:
        print(
            f"[tracker] W&B logging failed ({type(e).__name__}: {e}). "
            f"Eval results preserved locally: {metrics}"
        )
        return None
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
