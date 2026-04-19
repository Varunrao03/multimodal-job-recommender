from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Literal
from uuid import uuid4

import google.generativeai as genai
import pdfplumber
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models as qm

# PDF text → Gemini embeddings → Qdrant.

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API key
_gemini_key = os.getenv("GEMINI_API_KEY") 
if not _gemini_key:
    logger.warning("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; embed_text will fail until it is.")
else:
    genai.configure(api_key=_gemini_key)

# Default: text-embedding-004 → 768-dim vectors (must match Qdrant collection size).
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
TEXT_EMBED_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "768"))


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


def embed_text(
    text: str,
    *,
    task_type: Literal["retrieval_document", "retrieval_query"] = "retrieval_document",
) -> list[float]:
    """Embed string with the Gemini embedding model."""
    if not _gemini_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in app/.env or the environment.")
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    text = text.strip().replace("\n", " ")

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    embedding = result["embedding"]
    if len(embedding) != TEXT_EMBED_DIM:
        logger.warning(
            "Embedding length %s != GEMINI_EMBEDDING_DIM %s; update GEMINI_EMBEDDING_DIM or recreate the Qdrant collection.",
            len(embedding),
            TEXT_EMBED_DIM,
        )
    logger.info("[TEXT] Embedded %s chars → %s-dim (%s)", len(text), len(embedding), task_type)
    return embedding


def embed_pdf(pdf_path: str | Path) -> list[float]:
    """Extract PDF text then embed (convenience for scripts/tests)."""
    return embed_text(extract_pdf_text(pdf_path), task_type="retrieval_document")

"---------------------------------------------------------------------------------------------------"
# Qdrant vector database client and operations

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
    query_vector = embed_text(query, task_type="retrieval_query")
    return qdrant.search(
        collection_name="resumes",
        query_vector=query_vector,
        limit=top_k,
    )

"---------------------------------------------------------------------------------------------------"
# Ingest PDF resumes: extract text, embed, and upsert to Qdrant with metadata.

def index_resume_pdf(path: str, original_name: str | None = None) -> str:
    text = extract_pdf_text(path)
    vector = embed_text(text, task_type="retrieval_document")
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

    if len(sys.argv) > 1 and sys.argv[1] == "--embed-smoke":
        dim = len(embed_text("smoke test", task_type="retrieval_query"))
        print(f"embed_text OK, dimension={dim}")
    else:
        print(
            "backend.py is a library, not the HTTP server.\n"
            "  Start API (from project root):  uvicorn app.main:app --reload\n"
            "  Quick Gemini check:             python -m app.backend --embed-smoke\n"
            "  Set GEMINI_API_KEY (or GOOGLE_API_KEY) in app/.env\n"
            "  Qdrant must be running for ingest/search (default localhost:6333)."
        )
