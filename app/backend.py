from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List
from uuid import uuid4

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models as qm

# Text embedder: extract text from PDF, embed with OpenAI, store/search in Qdrant.

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-large"
TEXT_EMBED_DIM = 3072


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract plain text from a PDF using pdfplumber."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text()
            if t:
                parts.append(t)
            else:
                logger.warning("[PDF] Page %s/%s — no text, skipping", i, len(pdf.pages))

    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(
            f"No text extracted from {pdf_path.name}. "
            "It may be a scanned PDF; use OCR if needed."
        )
    logger.info("[PDF] Extracted %s characters from %s", len(text), pdf_path.name)
    return text


def embed_text(text: str) -> list[float]:
    """Embed string with text-embedding-3-large."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    text = text.strip().replace("\n", " ")
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    embedding = response.data[0].embedding
    logger.info("[TEXT] Embedded %s chars → %s-dim", len(text), len(embedding))
    return embedding


def embed_pdf(pdf_path: str | Path) -> list[float]:
    """Extract PDF text then embed (convenience for scripts/tests)."""
    return embed_text(extract_pdf_text(pdf_path))


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_collections() -> None:
    """Ensure the `resumes` collection exists. Requires Qdrant listening on QDRANT_HOST:QDRANT_PORT."""
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
    except Exception as e:
        raise RuntimeError(
            f"Qdrant is not reachable at {QDRANT_HOST}:{QDRANT_PORT} ({e}). "
            "Start it in another terminal, for example:\n"
            "  docker run -p 6333:6333 qdrant/qdrant"
        ) from e
    if "resumes" not in existing:
        qdrant.create_collection(
            collection_name="resumes",
            vectors_config=qm.VectorParams(size=TEXT_EMBED_DIM, distance=qm.Distance.COSINE),
        )
        logger.info("Created Qdrant collection: resumes")
    else:
        logger.info("Qdrant collection already exists: resumes")


def search_resumes(query: str, top_k: int = 5) -> List[qm.ScoredPoint]:
    query_vector = embed_text(query)
    return qdrant.search(
        collection_name="resumes",
        query_vector=query_vector,
        limit=top_k,
    )


def index_resume_pdf(path: str, original_name: str | None = None) -> str:
    text = extract_pdf_text(path)
    vector = embed_text(text)
    doc_id = str(uuid4())

    qdrant.upsert(
        collection_name="resumes",
        points=[
            qm.PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "type": "resume",
                    "file_name": original_name or os.path.basename(path),
                    "full_text": text,
                },
            )
        ],
    )
    return doc_id


if __name__ == "__main__":
    import sys

    # This file is normally imported by `app.main`. To exercise it directly:
    #   Repo root:  python -m app.backend --embed-smoke
    # Needs OPENAI_API_KEY (app/.env or env) and network for OpenAI.
    if len(sys.argv) > 1 and sys.argv[1] == "--embed-smoke":
        dim = len(embed_text("smoke test"))
        print(f"embed_text OK, dimension={dim}")
    else:
        print(
            "backend.py is a library, not the HTTP server.\n"
            "  Start API (from project root):  uvicorn app.main:app --reload\n"
            "  Quick OpenAI check:             python -m app.backend --embed-smoke\n"
            "  Qdrant must be running for ingest/search (default localhost:6333)."
        )
