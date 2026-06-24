# RAG Eval Harness — Build Tracker

Plan: `docs/superpowers/plans/2026-06-24-rag-eval-harness.md`

## Tasks

| # | Task | Status |
|---|------|--------|
| 1 | Project scaffold + arXiv corpus ingestion | ✅ done |
| 2 | Minimal LangGraph agent | ✅ done |
| 3 | LangSmith tracing + smoke test | ✅ done |
| 4 | Break agent intentionally (failure-first origin story) | ✅ done |
| 5 | Golden dataset schema + v1 seed | ✅ done |
| 6 | Eval contracts + RAGAS Track A | ✅ done |
| 7 | Trajectory scorer (Track B) | ✅ done |
| 8 | Cross-family LLM judge + κ calibration (Track C) | ✅ done |
| 9 | W&B regression tracker + baseline diff gate | ✅ done |
| 10 | FastAPI service + Docker | ✅ done |
| 11 | GitHub Actions CI gate | ✅ done |
| 12 | Expand golden set (30→50+) + regression story + README | ✅ done |

## Progress

**44/44 tests passing.**

## Notes

- Agent uses Groq `qwen/qwen3-32b` (Alibaba/Qwen family)
- Judge uses Groq `llama-3.3-70b-versatile` (Meta/Llama family) — cross-family bias control, one API key
- Embeddings use ChromaDB default sentence-transformers (all-MiniLM-L6-v2) — no API key needed
- GROQ_API_KEY and GEMINI_API_KEY added to `.env`
- OpenAI dependency fully removed from agent and corpus
- Task 12 requires manual work: expand golden_v1.yaml to 30+ items after observing real agent failures, then write regression story + README
- RAGAS import deprecation warnings (ragas.metrics → ragas.metrics.collections) — non-breaking, fix when upgrading to ragas v1.0
