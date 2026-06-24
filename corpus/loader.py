from typing import Generator
import arxiv


def fetch_arxiv_papers(
    categories: list[str] | None = None,
    max_results: int = 500,
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
        yield {
            "arxiv_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "abstract": paper.summary.replace("\n", " "),
            "authors": [str(a) for a in paper.authors[:5]],
            "year": paper.published.year,
            "text": f"Title: {paper.title}\n\nAbstract: {paper.summary.replace(chr(10), ' ')}",
        }
