import io
import time
import logging
from typing import Generator

import arxiv
import requests
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

PDF_TIMEOUT = 30        # seconds per download
PDF_RETRY_DELAY = 2     # seconds between retries on 429/503


def _download_pdf(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=PDF_TIMEOUT)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 503):
                time.sleep(PDF_RETRY_DELAY * (attempt + 1))
                continue
            logger.warning("PDF download failed %s: HTTP %s", url, r.status_code)
            return None
        except requests.RequestException as e:
            logger.warning("PDF download error %s: %s", url, e)
            return None
    return None


def _extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning("PDF text extraction failed: %s", e)
        return ""


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def fetch_arxiv_papers(
    categories: list[str] | None = None,
    max_results: int = 500,
    full_text: bool = True,
) -> Generator[dict, None, None]:
    if categories is None:
        categories = ["cs.LG", "cs.AI", "cs.CL"]

    client = arxiv.Client()
    query = " OR ".join(f"cat:{c}" for c in categories)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for paper in client.results(search):
        arxiv_id = paper.entry_id.split("/")[-1]
        meta = {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "),
            "authors": [str(a) for a in paper.authors[:5]],
            "year": paper.published.year,
        }

        if full_text:
            pdf_bytes = _download_pdf(paper.pdf_url)
            if pdf_bytes:
                body = _extract_text(pdf_bytes)
                chunks = _chunk_text(body) if body else []
            else:
                chunks = []

            if chunks:
                for i, chunk in enumerate(chunks):
                    yield {**meta, "chunk_index": i, "total_chunks": len(chunks), "text": chunk}
                continue

        # Fallback to abstract if PDF unavailable or full_text=False
        yield {**meta, "chunk_index": 0, "total_chunks": 1,
               "text": f"Title: {paper.title}\n\nAbstract: {meta['abstract']}"}
