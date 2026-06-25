from langchain_core.tools import tool
from config import RETRIEVAL_K


@tool
def vector_search(query: str, k: int = RETRIEVAL_K) -> list[str]:
    """Search the arXiv corpus for relevant paper passages."""
    from corpus.vectorstore import similarity_search
    return similarity_search(query, k=k)
