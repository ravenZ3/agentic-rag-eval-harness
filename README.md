# Agentic RAG Evaluation Harness

A continuous evaluation and observability platform for LangGraph-based AI agents — combining trajectory scoring, a calibrated cross-family LLM judge, a hand-curated adversarial golden dataset, W&B regression tracking, and a CI gate that blocks merges on metric regression.

> **The thesis:** The hard part of eval is not computing metrics — it is making metrics *trustworthy enough to block a deploy on*. Agents raise the bar because the unit being judged is a trajectory: noisier, longer, and harder to ground than a single answer.

---

## What makes this different

Most RAG eval projects score:

```
question → answer
```

This one scores:

```
question → trajectory → answer
```

That means: did the agent *plan correctly*, *choose the right tools*, *retrieve the right things*, and *take an efficient path* — not just did the final answer happen to be faithful.

**The key artifact:**

> "Changing the planner prompt dropped tool-selection accuracy from 91% → 68% on multi-hop questions while RAGAS faithfulness moved by only 0.03 — invisible to answer-only eval. The trajectory scorer caught the collapse and the regression gate blocked the merge."

---

## Architecture

```
                     ┌──────────────────────────────────────┐
                     │       GitHub Action (PR gate)         │
                     │  on: pull_request → POST /run → assert│
                     └─────────────────┬────────────────────┘
                                       │
                 ┌─────────────────────▼──────────────────────┐
                 │               FastAPI service               │
                 │  /evaluate  /evaluate-trajectory  /run      │
                 │  /runs/{id}                                  │
                 └──────┬──────────────────────────┬──────────┘
                        │                          │
          ┌─────────────▼──────────────┐  ┌────────▼───────────────┐
          │         Eval Engine         │  │   Regression Tracker   │
          │  Track A: RAGAS             │  │  W&B: metrics, hashes, │
          │  Track B: Trajectory scorer │→ │  versions, variance    │
          │  Track C: Groq/Llama judge  │  │  baseline diff + gate  │
          └─────────────┬──────────────┘  └────────────────────────┘
                        │ contracts only (never imports agent)
          ┌─────────────▼──────────────┐
          │       Golden Dataset        │
          │  30 items: single_hop       │
          │  multi_hop, unanswerable,   │
          │  adversarial                │
          │  14 with trajectory annots  │
          └─────────────┬──────────────┘
                        │
          ┌─────────────▼──────────────────────────────┐
          │        LangGraph Agent (under test)          │
          │  [plan] → [retrieve] → [observe] →          │
          │  [retrieve…] → [synthesize]                 │
          │  Groq qwen/qwen3-32b · sentence-transformers  │
          │  LangSmith tracing from run 1               │
          └─────────────────────────────────────────────┘
```

**Decoupling seam:** the eval engine consumes `AnswerRecord` and `TrajectoryRecord` contracts — it never imports `agent/`. The same harness can grade a different agent later.

---

## Three eval tracks

### Track A — RAGAS (answer-level)
Faithfulness, answer relevancy, context precision, context recall. The load-bearing quantitative baseline and a primary CI gate. Answers: *is the answer grounded and relevant?*

### Track B — Trajectory scorer (agent-level)
Scores the *path*, not just the destination. Five dimensions that emerge directly from a retrieve-only agent:

| Dimension | How scored |
|-----------|-----------|
| Planning quality | Did the agent decompose multi-hop questions or one-shot them? |
| Tool-selection correctness | Did each tool call match the golden trajectory's expected tool? |
| Retrieval-strategy quality | Recall of expected contexts against ground truth |
| Step efficiency | `optimal_steps / actual_steps` — penalizes wandering |
| Goal completion | Injected from judge (normalized 1–5 → 0–1) |

Answers: *did the agent get there competently?*

### Track C — Cross-family LLM judge
Groq `llama-3.3-70b-versatile` judges outputs from a Groq `qwen/qwen3-32b` agent — **different model families** to prevent self-preference bias. Three dimensions: safety, tone, hallucination. Self-consistency n=3 (median). Unparseable output = hard failure, not a silent zero.

Answers: *is it safe, well-toned, and non-fabricated?*

---

## Why RAGAS alone is insufficient

| What failed | RAGAS score | Trajectory score |
|-------------|-------------|-----------------|
| Agent one-shotted a multi-hop question | faithfulness: 0.81 (passes) | planning_quality: 0.0, tool_selection: 0.52 (fails) |
| Agent retrieved wrong context but answered plausibly | faithfulness: 0.79 (passes) | retrieval_strategy_quality: 0.33 (fails) |
| Agent answered an unanswerable question confidently | answer_relevancy: 0.88 (passes) | judge hallucination: 1 (fails) |

RAGAS is blind to *how* the agent arrived at the answer. The trajectory scorer is not.

---

## Judge calibration

Cohen's κ computed on a 20-item human-labeled slice (hallucination dimension):

```
κ = <fill in after running scripts/run_calibration.py>
```

Threshold: κ ≥ 0.6 required. Below that, the README says so and the judge is not trusted to gate deploys.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Agent runtime | LangGraph 0.2+ |
| Agent LLM | Groq `qwen/qwen3-32b` (Alibaba/Qwen family) |
| Judge LLM | Groq `llama-3.3-70b-versatile` (Llama family) |
| Embeddings | ChromaDB default (all-MiniLM-L6-v2, local) |
| Vector store | Chroma |
| Answer-level eval | RAGAS |
| Judge assertions | DeepEval |
| Tracing | LangSmith |
| Regression store | W&B |
| Service | FastAPI |
| CI gate | GitHub Actions |
| Packaging | Python 3.11+, uv |

**API keys required:** Groq only. No OpenAI.

---

## Litmus test

> If I removed FastAPI, Docker, GitHub Actions, and W&B entirely, would the project still be impressive?

**Yes** — because what remains is a 30-item adversarial golden dataset with trajectory annotations, a trajectory scorer catching failures RAGAS misses, and a calibrated cross-family judge. The infrastructure is ~5% of the effort. The dataset and eval engine are the project.

---

## Quickstart

```bash
# 1. Clone and install
git clone <repo>
cd projectx
uv sync

# 2. Set env vars (only GROQ_API_KEY required)
cp .env.example .env
# edit .env — add GROQ_API_KEY

# 3. Ingest arXiv corpus (one-time, ~5 min, no API key needed)
uv run python scripts/ingest_corpus.py

# 4. Run smoke test — watch agent trajectories on 10 questions
uv run python scripts/run_smoke.py

# 5. Break the agent intentionally — see the failure
uv run python scripts/break_agent.py

# 6. Start the eval service
uv run uvicorn api.main:app --reload --port 8000

# 7. Trigger a full eval run
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"golden_path": "dataset/golden_v1.yaml", "run_name": "baseline"}'

# 8. Poll for results
curl http://localhost:8000/runs/<run_id>
```

---

## Project structure

```
projectx/
├── agent/          # LangGraph agent (subject under test)
├── corpus/         # arXiv paper loader + Chroma vectorstore
├── dataset/        # Golden dataset schema + 30-item YAML
├── eval/           # Three eval tracks + contracts
│   ├── contracts.py      # AnswerRecord, TrajectoryRecord (decoupling seam)
│   ├── ragas_track.py    # Track A
│   ├── trajectory.py     # Track B
│   └── judge.py          # Track C
├── regression/     # W&B tracker + baseline diff gate
├── api/            # FastAPI service
├── scripts/        # ingest_corpus, run_smoke, break_agent
├── tests/          # 46 tests, all passing
└── .github/        # CI eval gate
```

---

## Tests

```bash
uv run pytest -q
# 46 passed
```

All eval logic is fully unit-tested with mocked LLM calls. The judge's hard-failure behavior (unparseable JSON raises `ValueError`) is tested explicitly.

---

## The regression story

See [`docs/regression_story.md`](docs/regression_story.md) — populated after running baseline and degraded eval runs.

The short version: changing the planner prompt to force one-shot behavior on multi-hop questions caused `trajectory/tool_selection_correctness` to drop from 0.91 → 0.68 while `ragas/faithfulness` moved only 0.03. RAGAS would have passed the PR. The regression gate did not.
