#!/usr/bin/env python3
"""Fetch arXiv ML papers and ingest into Chroma. Run once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from corpus.loader import fetch_arxiv_papers
from corpus.vectorstore import ingest_papers

if __name__ == "__main__":
    print("Fetching 500 arXiv ML papers...")
    papers = list(fetch_arxiv_papers(max_results=500))
    print(f"Fetched {len(papers)} papers. Ingesting into Chroma...")
    ingest_papers(papers)
    print("Done.")
