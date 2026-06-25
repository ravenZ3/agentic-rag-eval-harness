from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = Path("data/chroma")
COLLECTION_NAME = "arxiv_ml"

# all-MiniLM-L6-v2 already cached at ~/.cache/huggingface/hub/ — no download.
_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_EMBEDDINGS,
        persist_directory=str(PERSIST_DIR),
    )


def _scalarize(value):
    # Chroma metadata accepts only str/int/float/bool; flatten lists/tuples
    # (e.g. authors) into a comma-joined string so the data is preserved.
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return value


def ingest_papers(papers: list[dict]) -> None:
    vs = get_vectorstore()
    docs = [
        Document(
            page_content=p["text"],
            metadata={k: _scalarize(v) for k, v in p.items() if k != "text"},
        )
        for p in papers
    ]
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        vs.add_documents(docs[i : i + batch_size])
        print(f"  ingested {min(i + batch_size, len(docs))}/{len(docs)}")


def similarity_search(query: str, k: int = 4) -> list[str]:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
