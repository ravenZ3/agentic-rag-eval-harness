import pytest
from unittest.mock import patch, MagicMock
from eval.contracts import TrajectoryRecord, TrajectoryStep
from eval.judge import judge_trajectory, compute_kappa


def _make_record() -> TrajectoryRecord:
    return TrajectoryRecord(
        question="What is attention?",
        steps=[TrajectoryStep("retrieve", "vector_search", {}, "passage", "got it")],
        final_answer="Attention computes weighted sums of values.",
    )


def _mock_groq_response(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


VALID_JUDGE_JSON = '{"safety": 5, "tone": 4, "hallucination": 5, "reasoning": "Looks good.", "failing_claims": []}'
VALID_GC_JSON = '{"goal_completion": 4, "reasoning": "Mostly answered."}'


def test_judge_returns_result_with_all_fields():
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_groq_response(VALID_JUDGE_JSON),  # sample 1
            _mock_groq_response(VALID_JUDGE_JSON),  # sample 2
            _mock_groq_response(VALID_JUDGE_JSON),  # sample 3
            _mock_groq_response(VALID_GC_JSON),     # goal completion
        ]
        result = judge_trajectory(_make_record(), n_samples=3)

    assert result.safety == 5
    assert result.tone == 4
    assert result.hallucination == 5
    assert result.goal_completion == 4
    assert isinstance(result.variance, dict)
    assert "safety" in result.variance
    assert result.variance["safety"] == 0.0  # all samples identical → std=0


def test_judge_strips_markdown_fences():
    fenced = "```json\n" + VALID_JUDGE_JSON + "\n```"
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_groq_response(fenced),
            _mock_groq_response(fenced),
            _mock_groq_response(fenced),
            _mock_groq_response(VALID_GC_JSON),
        ]
        result = judge_trajectory(_make_record(), n_samples=3)
    assert result.safety == 5


def test_judge_raises_on_unparseable_json():
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response("not json at all")
        with pytest.raises(ValueError, match="unparseable JSON"):
            judge_trajectory(_make_record(), n_samples=1)


def test_judge_raises_on_missing_keys():
    bad = '{"safety": 5}'  # missing tone, hallucination, reasoning, failing_claims
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_groq_response(bad)
        with pytest.raises(ValueError, match="missing keys"):
            judge_trajectory(_make_record(), n_samples=1)


def test_judge_raises_on_missing_goal_completion_key():
    bad_gc = '{"reasoning": "ok"}'  # missing goal_completion
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_groq_response(VALID_JUDGE_JSON),
            _mock_groq_response(bad_gc),
        ]
        with pytest.raises(ValueError, match="missing key"):
            judge_trajectory(_make_record(), n_samples=1)


def test_judge_median_over_samples():
    scores = [
        '{"safety": 3, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
        '{"safety": 5, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
        '{"safety": 4, "tone": 4, "hallucination": 5, "reasoning": "r", "failing_claims": []}',
    ]
    with patch("eval.judge._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            *[_mock_groq_response(s) for s in scores],
            _mock_groq_response(VALID_GC_JSON),
        ]
        result = judge_trajectory(_make_record(), n_samples=3)
    assert result.safety == 4  # median of [3, 5, 4] = 4


def test_compute_kappa_perfect_agreement():
    assert compute_kappa([5, 4, 3, 5, 4], [5, 4, 3, 5, 4]) == pytest.approx(1.0)


def test_compute_kappa_no_agreement():
    # When raters are perfectly consistent but always disagree, κ = 0.0
    # (sklearn returns 0 rather than negative when one rater has zero variance)
    kappa = compute_kappa([1, 1, 1, 1], [5, 5, 5, 5])
    assert kappa <= 0.0


def test_compute_kappa_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_kappa([1, 2, 3], [1, 2])
