import os
import sys
import pytest
from unittest.mock import MagicMock

# Set dummy API keys so clients can be constructed at import time without a
# real key. The agent runs on Groq; the judge/RAGAS tracks run on Groq or
# Gemini depending on SCORING_PROVIDER. None of these are actually called in
# tests — network seams are mocked — but the SDKs validate key presence on init.
os.environ.setdefault("OPENAI_API_KEY", "mock-key-not-used")
os.environ.setdefault("ANTHROPIC_API_KEY", "mock-anthropic-key")
os.environ.setdefault("GROQ_API_KEY", "mock-groq-key")
os.environ.setdefault("GEMINI_API_KEY", "mock-gemini-key")
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
