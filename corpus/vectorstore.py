from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, INGEST_BATCH_SIZE, EMBED_MODEL

PERSIST_DIR = Path(CHROMA_PERSIST_DIR)
COLLECTION_NAME = CHROMA_COLLECTION

# Model already cached at ~/.cache/huggingface/hub/ — no download on repeat runs.
_EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


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
    for i in range(0, len(docs), INGEST_BATCH_SIZE):
        vs.add_documents(docs[i : i + INGEST_BATCH_SIZE])
        print(f"  ingested {min(i + INGEST_BATCH_SIZE, len(docs))}/{len(docs)}")


def similarity_search(query: str, k: int = 4) -> list[str]:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
