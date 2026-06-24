# tests/eval/test_ragas_track.py
import pytest
from unittest.mock import patch, MagicMock
from eval.contracts import AnswerRecord
from eval.ragas_track import run_ragas


def test_run_ragas_returns_result_shape():
    records = [
        AnswerRecord(
            question="What is attention?",
            answer="A mechanism computing weighted sums of values.",
            contexts=["Attention computes weighted sums. Keys and queries determine weights."],
            ground_truth="Attention computes a weighted sum of values based on query-key similarity.",
        )
    ]
    mock_result = MagicMock()
    mock_result.__getitem__ = lambda self, k: {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.8,
        "context_recall": 0.75,
    }[k]
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"question": "What is attention?", "faithfulness": 0.9}]
    mock_result.to_pandas.return_value = mock_df

    with patch("eval.ragas_track.evaluate", return_value=mock_result):
        with patch("eval.ragas_track.Dataset"):
            result = run_ragas(records)

    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.85
    assert result.context_precision == 0.8
    assert result.context_recall == 0.75
    assert isinstance(result.per_item, list)
