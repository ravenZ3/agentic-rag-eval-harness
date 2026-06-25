"""Shared (de)serialization for eval checkpoints.

A checkpoint is a JSON dict written during a run and re-read by the offline
scoring/comparison scripts. Centralizing the record-rebuilding here keeps the
three consumers — run_eval (resume), score_ragas_offline, compare_trajectories —
from each re-implementing it.
"""
import json
import dataclasses

from eval.contracts import (
    AnswerRecord,
    TrajectoryRecord,
    TrajectoryStep,
    OperationalRecord,
    RagasResult,
)


def _answer_records(data: dict) -> list[AnswerRecord]:
    return [AnswerRecord(**r) for r in data.get("answer_records", [])]


def _trajectory_records(data: dict) -> list[TrajectoryRecord]:
    return [
        TrajectoryRecord(
            question=r["question"],
            final_answer=r["final_answer"],
            steps=[TrajectoryStep(**s) for s in r["steps"]],
        )
        for r in data.get("trajectory_records", [])
    ]


def serialize_checkpoint(answer_records, trajectory_records, ops_records, ragas_result=None) -> dict:
    """Build the JSON-serializable checkpoint dict written during a run."""
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
    if ragas_result is not None:
        data["ragas_result"] = dataclasses.asdict(ragas_result)
    return data


def load_answer_records(path: str) -> list[AnswerRecord]:
    with open(path) as f:
        return _answer_records(json.load(f))


def load_trajectory_records(path: str) -> list[TrajectoryRecord]:
    with open(path) as f:
        return _trajectory_records(json.load(f))


def load_checkpoint(path: str):
    """Full load -> (answer_records, trajectory_records, ops_records, ragas_result|None)."""
    with open(path) as f:
        data = json.load(f)
    ops = [OperationalRecord(**r) for r in data.get("ops_records", [])]
    ragas = RagasResult(**data["ragas_result"]) if "ragas_result" in data else None
    return _answer_records(data), _trajectory_records(data), ops, ragas
