import os
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .embeddings import embed_text, CLIP_EMBED_DIM, TEXT_EMBED_DIM


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_collections() -> None:
    # Text collections (same embedding model / dimension)
    qdrant.recreate_collection(
        collection_name="resumes",
        vectors_config=qm.VectorParams(size=TEXT_EMBED_DIM, distance=qm.Distance.COSINE),
    )
    qdrant.recreate_collection(
        collection_name="qa_transcripts",
        vectors_config=qm.VectorParams(size=TEXT_EMBED_DIM, distance=qm.Distance.COSINE),
    )

    # Image collection (CLIP)
    qdrant.recreate_collection(
        collection_name="images",
        vectors_config=qm.VectorParams(size=CLIP_EMBED_DIM, distance=qm.Distance.COSINE),
    )


def search_text_collections(query: str, top_k: int = 5) -> Tuple[List[qm.ScoredPoint], List[qm.ScoredPoint]]:
    query_vector = embed_text(query)

    def _search(name: str) -> List[qm.ScoredPoint]:
        return qdrant.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
        )

    return _search("resumes"), _search("qa_transcripts")


def search_similar_images(image_path: str, top_k: int = 5) -> List[qm.ScoredPoint]:
    from .embeddings import embed_image

    query_vector = embed_image(image_path)
    return qdrant.search(
        collection_name="images",
        query_vector=query_vector,
        limit=top_k,
    )

