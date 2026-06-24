import sys
from unittest.mock import MagicMock
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from eval.contracts import AnswerRecord, RagasResult

# Use Groq Llama for RAGAS LLM scoring — no OpenAI key required.
# LangchainEmbeddingsWrapper exposes embed_query, which RAGAS metrics call;
# the ragas-native HuggingFaceEmbeddings does not (new async-only interface).
_ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))
_ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# faithfulness/context_precision fan out into many sequential 70B calls.
# Groq rate-limits + default 180s timeout = TimeoutError. Raise the timeout
# and cap concurrency so we stay under Groq's per-minute token ceiling.
_RUN_CONFIG = RunConfig(timeout=600, max_workers=2, max_retries=5, max_wait=60)


def run_ragas(records: list[AnswerRecord]) -> RagasResult:
    """Track A: answer-level evaluation via RAGAS."""
    data = {
        "question": [r.question for r in records],
        "answer": [r.answer for r in records],
        "contexts": [r.contexts for r in records],
        "ground_truth": [r.ground_truth for r in records],
    }
    ds = Dataset.from_dict(data)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm,
        embeddings=_ragas_embeddings,
        run_config=_RUN_CONFIG,
    )
    scores_df = result.to_pandas()

    def _agg(metric: str) -> float:
        """Aggregate a metric column to a NaN-safe mean. RAGAS returns
        per-item lists/columns rather than a scalar in current versions."""
        if metric in scores_df.columns:
            val = scores_df[metric].mean(skipna=True)
            return float(val) if val == val else float("nan")  # val!=val ⇒ NaN
        # Fallback to the result mapping if column naming differs.
        raw = result[metric]
        if isinstance(raw, (list, tuple)):
            finite = [v for v in raw if v == v]
            return float(sum(finite) / len(finite)) if finite else float("nan")
        return float(raw)

    return RagasResult(
        faithfulness=_agg("faithfulness"),
        answer_relevancy=_agg("answer_relevancy"),
        context_precision=_agg("context_precision"),
        context_recall=_agg("context_recall"),
        per_item=scores_df.to_dict(orient="records"),
    )
