import pytest
from unittest.mock import patch, MagicMock
from eval.contracts import TrajectoryRecord, TrajectoryStep
from eval.judge import judge_trajectory, compute_kappa, _call


def _make_record() -> TrajectoryRecord:
    return TrajectoryRecord(
        question="What is attention?",
        steps=[TrajectoryStep("retrieve", "vector_search", {}, "passage", "got it")],
        final_answer="Attention computes weighted sums of values.",
    )


VALID_JUDGE_JSON = '{"safety": 5, "tone": 4, "hallucination": 5, "reasoning": "Looks good.", "failing_claims": []}'
VALID_GC_JSON = '{"goal_completion": 4, "reasoning": "Mostly answered."}'


# judge_trajectory calls _call() n_samples times for the rubric, then once for
# goal_completion. We patch _call (the provider seam) so these tests are
# provider-agnostic — they exercise aggregation/parsing/defaulting, not the SDK.

def test_judge_returns_result_with_all_fields():
    with patch("eval.judge._call", side_effect=[
        VALID_JUDGE_JSON, VALID_JUDGE_JSON, VALID_JUDGE_JSON, VALID_GC_JSON,
    ]):
        result = judge_trajectory(_make_record(), n_samples=3)

    assert result.safety == 5
    assert result.tone == 4
    assert result.hallucination == 5
    assert result.goal_completion == 4
    assert isinstance(result.variance, dict)
    assert result.variance["safety"] == 0.0  # identical samples -> std 0


def test_judge_median_over_samples():
    scores = [
        '{"safety": 3, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
        '{"safety": 5, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
        '{"safety": 4, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
    ]
    with patch("eval.judge._call", side_effect=[*scores, VALID_GC_JSON]):
        result = judge_trajectory(_make_record(), n_samples=3)
    assert result.safety == 4  # median of [3, 5, 4]


# --- graceful degradation (current behavior: skip bad samples, default to 3) ---
# The judge used to raise on bad JSON; it was hardened to skip-and-default so a
# single malformed judge response can't abort a whole eval run.

def test_judge_defaults_on_unparseable_json():
    with patch("eval.judge._call", side_effect=["not json at all", VALID_GC_JSON]):
        result = judge_trajectory(_make_record(), n_samples=1)
    assert result.safety == 3
    assert result.tone == 3
    assert result.hallucination == 3
    assert result.reasoning == "judge parse failed"


def test_judge_defaults_on_missing_keys():
    bad = '{"safety": 5}'  # missing tone/hallucination/reasoning/failing_claims
    with patch("eval.judge._call", side_effect=[bad, VALID_GC_JSON]):
        result = judge_trajectory(_make_record(), n_samples=1)
    assert result.safety == 3  # sample skipped -> default


def test_judge_defaults_goal_completion_on_bad_gc():
    bad_gc = '{"reasoning": "ok"}'  # missing goal_completion key
    with patch("eval.judge._call", side_effect=[VALID_JUDGE_JSON, bad_gc]):
        result = judge_trajectory(_make_record(), n_samples=1)
    assert result.safety == 5          # rubric still parsed fine
    assert result.goal_completion == 3  # gc fell back to default


# --- _call markdown-fence stripping (tested against the active provider seam) ---

def test_call_strips_markdown_fences():
    fenced = "```json\n" + VALID_JUDGE_JSON + "\n```"
    mock_client = MagicMock()
    # gemini path (the configured provider): resp.text carries the content
    mock_client.models.generate_content.return_value = MagicMock(text=fenced)
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=fenced))]
    )
    with patch("eval.judge._get_client", return_value=mock_client):
        out = _call("system", "user", max_tokens=256)
    assert out == VALID_JUDGE_JSON  # fences stripped


# --- Cohen's kappa ---

def test_compute_kappa_perfect_agreement():
    assert compute_kappa([5, 4, 3, 5, 4], [5, 4, 3, 5, 4]) == pytest.approx(1.0)


def test_compute_kappa_no_agreement():
    kappa = compute_kappa([1, 1, 1, 1], [5, 5, 5, 5])
    assert kappa <= 0.0


def test_compute_kappa_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_kappa([1, 2, 3], [1, 2])
