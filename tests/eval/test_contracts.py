# tests/eval/test_contracts.py
from eval.contracts import AnswerRecord, TrajectoryRecord, TrajectoryStep


def test_answer_record_fields():
    rec = AnswerRecord(
        question="What is attention?",
        answer="A weighted sum.",
        contexts=["passage A"],
        ground_truth="Attention computes weighted sums.",
    )
    assert rec.question == "What is attention?"
    assert rec.contexts == ["passage A"]


def test_trajectory_record_with_steps():
    steps = [
        TrajectoryStep("plan", None, None, None, "decomposed"),
        TrajectoryStep("retrieve", "vector_search", {"query": "attention"}, "result", "got 4 passages"),
    ]
    rec = TrajectoryRecord(question="What is attention?", steps=steps, final_answer="A weighted sum.")
    assert len(rec.steps) == 2
    assert rec.steps[1].tool_called == "vector_search"
