import os
import subprocess
from uuid import uuid4

import fitz  # pymupdf
import whisper

from .embeddings import embed_text, embed_image
from .vector_store import qdrant
from qdrant_client.http import models as qm


whisper_model_name = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(whisper_model_name)


def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    texts = [page.get_text("text") for page in doc]
    return "\n".join(texts)


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


def _av_to_wav(input_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _transcribe_audio(path: str) -> str:
    result = whisper_model.transcribe(path, language="en")
    return result["text"]


def index_av_file(path: str, original_name: str | None = None) -> str:
    wav_path = f"{path}.wav"
    _av_to_wav(path, wav_path)
    try:
        transcript = _transcribe_audio(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    vector = embed_text(transcript)
    doc_id = str(uuid4())

    qdrant.upsert(
        collection_name="qa_transcripts",
        points=[
            qm.PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "type": "qa_av",
                    "file_name": original_name or os.path.basename(path),
                    "transcript": transcript,
                },
            )
        ],
    )
    return doc_id


def index_image_file(path: str, original_name: str | None = None) -> str:
    vector = embed_image(path)
    doc_id = str(uuid4())

    qdrant.upsert(
        collection_name="images",
        points=[
            qm.PointStruct(
                id=doc_id,
                vector=vector,
                payload={
                    "type": "image",
                    "file_name": original_name or os.path.basename(path),
                },
            )
        ],
    )
    return doc_id

