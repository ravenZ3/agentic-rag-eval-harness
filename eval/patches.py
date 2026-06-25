"""
One-shot import-time patches for third-party library bugs.
Import this module first in any entrypoint script.
"""
from unittest.mock import MagicMock
import sys

# chromadb 0.5.x calls posthog.capture with 3 positional args but the posthog
# library changed to a keyword-only API — silence the resulting stderr spam.
import chromadb.telemetry.product.posthog as _chroma_ph
_chroma_ph.Posthog.capture = lambda *a, **kw: None

# ragas imports ChatVertexAI from langchain_community, which was removed in
# langchain-community 0.3+. Stub it out so ragas can be imported safely.
sys.modules.setdefault("langchain_community.chat_models.vertexai", MagicMock())
