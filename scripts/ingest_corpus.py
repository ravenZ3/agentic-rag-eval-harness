#!/usr/bin/env python3
"""Fetch arXiv ML papers (full PDF text), chunk, and ingest into Chroma."""
import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from corpus.loader import fetch_arxiv_papers
from corpus.vectorstore import get_vectorstore, PERSIST_DIR
from langchain_core.documents import Document
from corpus.vectorstore import _scalarize, INGEST_BATCH_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--max-results", type=int, default=200,
                    help="Number of papers to fetch (each produces multiple chunks)")
parser.add_argument("--clear", action="store_true",
                    help="Wipe existing Chroma collection before ingesting")
parser.add_argument("--abstract-only", action="store_true",
                    help="Skip PDF download; use title+abstract only")
args = parser.parse_args()

if args.clear and PERSIST_DIR.exists():
    print(f"Clearing existing corpus at {PERSIST_DIR}...")
    shutil.rmtree(PERSIST_DIR)

vs = get_vectorstore()
batch: list[Document] = []
total_chunks = 0
total_papers = 0

print(f"Fetching up to {args.max_results} papers ({'abstract only' if args.abstract_only else 'full PDF'})...")

for paper in fetch_arxiv_papers(max_results=args.max_results, full_text=not args.abstract_only):
    if paper.get("chunk_index", 0) == 0:
        total_papers += 1

    doc = Document(
        page_content=paper["text"],
        metadata={k: _scalarize(v) for k, v in paper.items() if k != "text"},
    )
    batch.append(doc)
    total_chunks += 1

    if len(batch) >= INGEST_BATCH_SIZE:
        vs.add_documents(batch)
        print(f"  {total_papers} papers / {total_chunks} chunks ingested...")
        batch = []

if batch:
    vs.add_documents(batch)

print(f"Done. {total_papers} papers → {total_chunks} chunks in Chroma.")
