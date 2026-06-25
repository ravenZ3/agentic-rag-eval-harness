import eval.patches  # noqa: F401 — must be first; applies import-time patches

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
from eval.contracts import AnswerRecord, RagasResult
from config import (
    SCORING_PROVIDER, RAGAS_LLM_MODEL, RAGAS_TEMPERATURE, EMBED_MODEL,
    RAGAS_ANSWER_RELEVANCY_STRICTNESS,
    RAGAS_TIMEOUT, RAGAS_MAX_WORKERS, RAGAS_MAX_RETRIES, RAGAS_MAX_WAIT,
)

# LangchainEmbeddingsWrapper exposes embed_query, which RAGAS metrics call;
# the ragas-native HuggingFaceEmbeddings does not (new async-only interface).
if SCORING_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI
    _ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=RAGAS_LLM_MODEL, temperature=RAGAS_TEMPERATURE))
else:
    from langchain_groq import ChatGroq
    _ragas_llm = LangchainLLMWrapper(ChatGroq(model=RAGAS_LLM_MODEL, temperature=RAGAS_TEMPERATURE))
_ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL)
)

# Groq only allows n=1; answer_relevancy defaults to strictness=3 which sends
# n=3 and gets a 400. Force strictness=1 so it stays within Groq's limits.
_answer_relevancy = AnswerRelevancy(strictness=RAGAS_ANSWER_RELEVANCY_STRICTNESS)

# faithfulness/context_precision fan out into many sequential 70B calls.
# Groq rate-limits + default 180s timeout = TimeoutError. Raise the timeout
# and cap concurrency so we stay under Groq's per-minute token ceiling.
_RUN_CONFIG = RunConfig(timeout=RAGAS_TIMEOUT, max_workers=RAGAS_MAX_WORKERS, max_retries=RAGAS_MAX_RETRIES, max_wait=RAGAS_MAX_WAIT)


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
        metrics=[faithfulness, _answer_relevancy, context_precision, context_recall],
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
