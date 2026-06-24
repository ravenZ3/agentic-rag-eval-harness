import math
from unittest.mock import patch, MagicMock

import pytest

from eval.contracts import RagasResult, TrajectoryScore, JudgeResult
from regression.tracker import compute_metrics, log_run, _wandb_available, _mean, _clean


def _ragas(faith=0.8, ar=0.7, cp=0.6, cr=0.9):
    return RagasResult(
        faithfulness=faith,
        answer_relevancy=ar,
        context_precision=cp,
        context_recall=cr,
        per_item=[],
    )


def _traj(goal=0.8, plan=0.7, tool=0.9, retr=0.6, eff=1.0):
    return TrajectoryScore(
        planning_quality=plan,
        tool_selection_correctness=tool,
        retrieval_strategy_quality=retr,
        step_efficiency=eff,
        goal_completion=goal,
        judge_used_for=[],
    )


def _judge(safety=5, tone=4, hall=5, gc=4):
    return JudgeResult(
        safety=safety,
        tone=tone,
        hallucination=hall,
        reasoning="ok",
        failing_claims=[],
        variance={},
        goal_completion=gc,
    )


def test_clean_drops_nan_and_inf():
    assert _clean(float("nan")) is None
    assert _clean(float("inf")) is None
    assert _clean(None) is None
    assert _clean(0.5) == 0.5


def test_mean_ignores_none_and_nan():
    assert _mean([1.0, None, float("nan"), 3.0]) == 2.0


def test_mean_empty_returns_none_not_zerodiv():
    assert _mean([]) is None
    assert _mean([None, float("nan")]) is None


def test_compute_metrics_basic():
    metrics = compute_metrics(_ragas(), [_traj()], [_judge()])
    assert metrics["ragas/faithfulness"] == 0.8
    assert metrics["trajectory/goal_completion"] == 0.8
    assert metrics["judge/safety"] == 5.0


def test_compute_metrics_strips_nan_ragas():
    metrics = compute_metrics(_ragas(faith=float("nan")), [_traj()], [_judge()])
    assert "ragas/faithfulness" not in metrics
    assert "ragas/answer_relevancy" in metrics


def test_compute_metrics_empty_lists_never_raises():
    metrics = compute_metrics(_ragas(), [], [])
    # ragas scalars survive; trajectory/judge aggregates are absent, not crashes
    assert "ragas/faithfulness" in metrics
    assert "trajectory/goal_completion" not in metrics
    assert "judge/safety" not in metrics


def test_wandb_available_offline_mode(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _wandb_available() is True


def test_wandb_available_online_no_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _wandb_available() is False


def test_wandb_available_online_with_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    assert _wandb_available() is True


def test_log_run_skips_without_key(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    run_id = log_run(_ragas(), [_traj()], [_judge()], config={}, golden_path=__file__)
    assert run_id is None
    assert "skipping W&B logging" in capsys.readouterr().out


def test_log_run_survives_wandb_failure(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    monkeypatch.setenv("WANDB_MODE", "online")
    with patch("regression.tracker.wandb.init", side_effect=RuntimeError("network down")):
        run_id = log_run(_ragas(), [_traj()], [_judge()], config={}, golden_path=__file__)
    assert run_id is None  # failure isolated, no exception propagated


def test_log_run_returns_id_on_success(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    monkeypatch.setenv("WANDB_MODE", "online")
    fake_run = MagicMock()
    fake_run.id = "abc123"
    with patch("regression.tracker.wandb.init", return_value=fake_run), \
         patch("regression.tracker.wandb.Table", return_value=MagicMock()):
        run_id = log_run(_ragas(), [_traj()], [_judge()], config={}, golden_path=__file__)
    assert run_id == "abc123"
    fake_run.finish.assert_called_once()
