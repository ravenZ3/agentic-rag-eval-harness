from unittest.mock import patch, MagicMock
from corpus.vectorstore import ingest_papers, similarity_search


def _mock_vs():
    return MagicMock()


def test_ingest_papers_adds_documents():
    fake_vs = MagicMock()
    papers = [
        {
            "arxiv_id": "2401.00001",
            "title": "Test Paper",
            "abstract": "A test.",
            "text": "Title: Test Paper\n\nAbstract: A test.",
            "authors": ["A"],
            "year": 2024,
        }
    ]
    with patch("corpus.vectorstore.get_vectorstore", return_value=fake_vs):
        ingest_papers(papers)
    fake_vs.add_documents.assert_called_once()
    added = fake_vs.add_documents.call_args[0][0]
    assert len(added) == 1
    assert added[0].page_content == papers[0]["text"]
    assert added[0].metadata["arxiv_id"] == "2401.00001"


def test_ingest_papers_batches_large_input():
    fake_vs = MagicMock()
    papers = [
        {
            "arxiv_id": f"2401.{i:05d}",
            "title": f"Paper {i}",
            "abstract": "x",
            "text": f"Title: Paper {i}",
            "authors": [],
            "year": 2024,
        }
        for i in range(250)
    ]
    with patch("corpus.vectorstore.get_vectorstore", return_value=fake_vs):
        ingest_papers(papers)
    assert fake_vs.add_documents.call_count == 3


def test_similarity_search_returns_documents():
    fake_vs = MagicMock()
    fake_doc = MagicMock()
    fake_doc.page_content = "passage A"
    fake_vs.similarity_search.return_value = [fake_doc]
    with patch("corpus.vectorstore.get_vectorstore", return_value=fake_vs):
        results = similarity_search("attention mechanism", k=1)
    assert results == ["passage A"]
    fake_vs.similarity_search.assert_called_once_with("attention mechanism", k=1)


def test_similarity_search_empty_result():
    fake_vs = MagicMock()
    fake_vs.similarity_search.return_value = []
    with patch("corpus.vectorstore.get_vectorstore", return_value=fake_vs):
        results = similarity_search("anything")
    assert results == []
