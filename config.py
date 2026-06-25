"""Central config for all model, eval, and corpus settings."""

# --- Agent ---
AGENT_MODEL = "llama-3.1-8b-instant"
AGENT_TEMPERATURE = 0
AGENT_MAX_STEPS = 6
RETRIEVAL_K = 2          # chunks per sub-goal; keep low to stay within Groq TPM

# --- Scoring provider: "groq" or "gemini" ---
# Switch here to change both RAGAS and judge without touching eval code.
SCORING_PROVIDER = "gemini"

# --- Judge (Track C) ---
JUDGE_TEMPERATURE = 0
JUDGE_N_SAMPLES = 3          # self-consistency runs; median is taken
JUDGE_MAX_TOKENS = 512
JUDGE_GC_MAX_TOKENS = 256    # goal-completion call is shorter

JUDGE_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}[SCORING_PROVIDER]

# --- RAGAS (Track A) ---
RAGAS_TEMPERATURE = 0
RAGAS_LLM_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
}[SCORING_PROVIDER]
RAGAS_ANSWER_RELEVANCY_STRICTNESS = 1   # Groq only allows n=1; default 3 errors
RAGAS_TIMEOUT = 600
RAGAS_MAX_WORKERS = 2
RAGAS_MAX_RETRIES = 5
RAGAS_MAX_WAIT = 60

# --- Embeddings ---
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Corpus / Chroma ---
CHROMA_PERSIST_DIR = "data/chroma"
CHROMA_COLLECTION = "arxiv_ml"
INGEST_BATCH_SIZE = 100

# --- PDF chunking ---
CHUNK_SIZE = 500        # words per chunk
CHUNK_OVERLAP = 50      # word overlap between consecutive chunks
