import os
import sys
import pytest
from unittest.mock import MagicMock

# Set dummy API keys for tests to prevent client validation errors on import
os.environ["OPENAI_API_KEY"] = "mock-key-not-used"  # ragas now uses Groq
os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-key"
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
