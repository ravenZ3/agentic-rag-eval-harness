from dataclasses import dataclass

DEFAULT_THRESHOLDS = {
    "ragas/faithfulness": -0.03,
    "ragas/answer_relevancy": -0.03,
    "trajectory/goal_completion": -0.05,
    "trajectory/tool_selection_correctness": -0.05,
    "trajectory/planning_quality": -0.05,
}


@dataclass
class RegressionResult:
    is_regression: bool
    failed_metrics: list[str]
    deltas: dict[str, float]


def check_regression(
    run_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> RegressionResult:
    """
    Compare run_metrics against baseline_metrics.
    A metric regresses if its delta is below the threshold (negative = drop).
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    deltas: dict[str, float] = {}
    failed: list[str] = []

    for metric, threshold in thresholds.items():
        if metric not in run_metrics or metric not in baseline_metrics:
            continue
        delta = run_metrics[metric] - baseline_metrics[metric]
        deltas[metric] = delta
        if delta < threshold:
            failed.append(metric)

    return RegressionResult(
        is_regression=len(failed) > 0,
        failed_metrics=failed,
        deltas=deltas,
    )
