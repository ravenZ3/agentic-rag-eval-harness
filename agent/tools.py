from langchain_core.tools import tool


@tool
def vector_search(query: str, k: int = 4) -> list[str]:
    """Search the arXiv corpus for relevant paper passages."""
    from corpus.vectorstore import similarity_search
    return similarity_search(query, k=k)
