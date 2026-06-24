import pytest
from regression.gate import check_regression, DEFAULT_THRESHOLDS


def test_no_regression_when_metrics_stable():
    baseline = {"ragas/faithfulness": 0.85, "trajectory/goal_completion": 0.80}
    run = {"ragas/faithfulness": 0.86, "trajectory/goal_completion": 0.81}
    result = check_regression(run, baseline)
    assert not result.is_regression
    assert result.failed_metrics == []


def test_regression_detected_on_faithfulness_drop():
    baseline = {"ragas/faithfulness": 0.90}
    run = {"ragas/faithfulness": 0.85}  # Δ = -0.05, threshold = -0.03
    result = check_regression(run, baseline)
    assert result.is_regression
    assert "ragas/faithfulness" in result.failed_metrics
    assert result.deltas["ragas/faithfulness"] == pytest.approx(-0.05)


def test_regression_not_triggered_within_threshold():
    baseline = {"ragas/faithfulness": 0.90}
    run = {"ragas/faithfulness": 0.88}  # Δ = -0.02, threshold = -0.03 → OK
    result = check_regression(run, baseline)
    assert not result.is_regression


def test_regression_on_multiple_metrics():
    baseline = {
        "ragas/faithfulness": 0.90,
        "trajectory/tool_selection_correctness": 0.91,
    }
    run = {
        "ragas/faithfulness": 0.85,
        "trajectory/tool_selection_correctness": 0.68,
    }
    result = check_regression(run, baseline)
    assert result.is_regression
    assert len(result.failed_metrics) == 2


def test_missing_metric_skipped_gracefully():
    baseline = {"ragas/faithfulness": 0.90, "trajectory/goal_completion": 0.80}
    run = {"ragas/faithfulness": 0.85}  # goal_completion absent from run
    result = check_regression(run, baseline)
    assert "ragas/faithfulness" in result.deltas
    assert "trajectory/goal_completion" not in result.deltas


def test_custom_thresholds():
    baseline = {"ragas/faithfulness": 0.90}
    run = {"ragas/faithfulness": 0.88}
    # Δ = -0.02 which is below custom threshold of -0.01
    result = check_regression(run, baseline, thresholds={"ragas/faithfulness": -0.01})
    assert result.is_regression


def test_improvement_not_flagged():
    baseline = {"ragas/faithfulness": 0.80}
    run = {"ragas/faithfulness": 0.90}  # improvement
    result = check_regression(run, baseline)
    assert not result.is_regression
    assert result.deltas["ragas/faithfulness"] == pytest.approx(0.10)
