"""
embeddings.py
=============
Multimodal Job Recommender — Embedding Pipeline
Handles PDF and video embeddings using OpenAI
text-embedding-3-large as the unified embedding model.

Dependencies:
    pip install openai pdfplumber ffmpeg-python openai-whisper
    brew install ffmpeg   # macOS
    winget install ffmpeg # Windows

Environment Variables:
    OPENAI_API_KEY — your OpenAI API key
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import pdfplumber   # PDF text extraction
import ffmpeg       # Video → audio extraction
import whisper      # Audio → text transcription
from openai import OpenAI

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM   = 3072  # output dimension for text-embedding-3-large


# ─────────────────────────────────────────────
# Core: Text → Embedding
# Shared internally by both PDF and video pipelines.
# All modalities reduce to text first, then call this.
# ─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Convert any text string into a vector embedding
    using text-embedding-3-large.

    This is an internal helper — called by embed_pdf()
    and embed_video() after they extract text from
    their respective sources.

    Args:
        text: Raw text string to embed.

    Returns:
        List of 3072 floats representing the text vector.

    Raises:
        ValueError: If text is empty or whitespace only.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    # Normalize whitespace
    text = text.strip().replace("\n", " ")

    try:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
        )
        embedding = response.data[0].embedding
        logger.info(f"[TEXT] Embedded {len(text)} chars → {len(embedding)}-dim vector")
        return embedding

    except Exception as e:
        logger.error(f"[TEXT] OpenAI embedding API call failed: {e}")
        raise


# ─────────────────────────────────────────────
# 1. PDF Embedding
#    PDF → pdfplumber (extract text) → embed_text()
# ─────────────────────────────────────────────

def embed_pdf(pdf_path: str) -> list[float]:
    """
    Extract text from a resume PDF and embed it.

    Pipeline:
        PDF file
            → pdfplumber (page by page text extraction)
            → raw text string
            → embed_text()
            → 3072-dim vector

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of 3072 floats — the resume vector.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted from the PDF.

    Example:
        vector = embed_pdf("resume.pdf")
        # vector = [0.12, 0.87, 0.34, ...]  (3072 floats)
    """
    pdf_path = Path(pdf_path)

    # Validate file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"[PDF] Extracting text from: {pdf_path.name}")

    extracted_text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()

                if page_text:
                    extracted_text += f"\n{page_text}"
                    logger.info(f"[PDF] Page {page_num}/{total_pages} — extracted {len(page_text)} chars")
                else:
                    # Page might be image-based (scanned PDF) — skip gracefully
                    logger.warning(f"[PDF] Page {page_num}/{total_pages} — no text found, skipping")

    except Exception as e:
        logger.error(f"[PDF] Failed to open or read PDF: {e}")
        raise

    # Guard: ensure we got something
    if not extracted_text.strip():
        raise ValueError(
            f"No text could be extracted from: {pdf_path.name}. "
            f"The PDF may be image-based (scanned). Consider using OCR."
        )

    logger.info(f"[PDF] Total extracted: {len(extracted_text)} characters")

    # Pass extracted text to the core embedding function
    return embed_text(extracted_text)


# ─────────────────────────────────────────────
# 2. Video Embedding
#    Video → FFmpeg (extract audio) → Whisper (transcribe) → embed_text()
# ─────────────────────────────────────────────

def embed_video(video_path: str) -> list[float]:
    """
    Extract speech from a candidate video introduction and embed the transcript.

    Pipeline:
        Video file (.mp4 / .mov / .webm)
            → FFmpeg (strips audio track → .wav file)
            → Whisper (transcribes speech → text)
            → embed_text()
            → 3072-dim vector

    The temporary .wav file is deleted automatically after transcription.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        List of 3072 floats — the video introduction vector.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If Whisper returns an empty transcript.
        ffmpeg.Error: If FFmpeg fails to process the video.

    Example:
        vector = embed_video("intro.mp4")
        # vector = [0.33, 0.91, 0.22, ...]  (3072 floats)
    """
    video_path = Path(video_path)

    # Validate file exists
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info(f"[VIDEO] Processing: {video_path.name}")

    # Create a temp file to store extracted audio
    # delete=False so FFmpeg can write to it after context manager exits
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        # ── Step 1: Extract audio track using FFmpeg ──
        logger.info("[VIDEO] Extracting audio track with FFmpeg...")
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    tmp_audio_path,
                    acodec="pcm_s16le",  # Uncompressed WAV — best for Whisper
                    ac=1,                # Mono channel
                    ar="16000",          # 16kHz sample rate (Whisper requirement)
                    vn=None,             # Drop video stream, keep audio only
                )
                .overwrite_output()
                .run(quiet=True)         # Suppress FFmpeg console output
            )
            logger.info(f"[VIDEO] Audio extracted → temp file: {tmp_audio_path}")

        except ffmpeg.Error as e:
            logger.error(f"[VIDEO] FFmpeg failed to extract audio: {e}")
            raise

        # ── Step 2: Transcribe audio using Whisper ──
        transcript = _transcribe_audio(tmp_audio_path)

    finally:
        # Always clean up the temp audio file
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
            logger.info("[VIDEO] Temp audio file cleaned up")

    # ── Step 3: Embed the transcript ──
    return embed_text(transcript)


# ─────────────────────────────────────────────
# Internal Helper: Whisper Transcription
# Used only by embed_video() — not exposed publicly
# ─────────────────────────────────────────────

def _transcribe_audio(audio_path: str) -> str:
    """
    Transcribe a .wav audio file to text using OpenAI Whisper.

    Internal helper called by embed_video() only.

    Whisper model sizes (speed vs accuracy tradeoff):
        tiny   → fastest, least accurate
        base   → good balance          ← we use this
        small  → better accuracy, slower
        medium → high accuracy, much slower
        large  → best accuracy, needs GPU

    Args:
        audio_path: Path to a .wav audio file (16kHz mono).

    Returns:
        Transcribed text string.

    Raises:
        ValueError: If Whisper returns an empty transcript.
    """
    logger.info("[WHISPER] Loading model (base)...")
    model = whisper.load_model("base")

    logger.info(f"[WHISPER] Transcribing: {audio_path}")

    try:
        result = model.transcribe(audio_path, language="en")
        transcript = result["text"].strip()

    except Exception as e:
        logger.error(f"[WHISPER] Transcription failed: {e}")
        raise

    if not transcript:
        raise ValueError(
            "Whisper returned an empty transcript. "
            "Check that the video contains clear speech."
        )

    logger.info(f"[WHISPER] Transcript ({len(transcript)} chars): {transcript[:120]}...")
    return transcript


# ─────────────────────────────────────────────
# Unified Entry Point
# ─────────────────────────────────────────────

def embed_candidate_profile(
    pdf_path:   Optional[str] = None,
    video_path: Optional[str] = None,
) -> dict[str, list[float]]:
    """
    Generate embeddings for a candidate's PDF resume and/or video introduction.
    Returns a dict of named vectors ready to upsert into Qdrant.

    At least one of pdf_path or video_path must be provided.

    Args:
        pdf_path:   Path to resume PDF file (optional).
        video_path: Path to video introduction file (optional).

    Returns:
        Dict with keys "resume" and/or "video".
        Each value is a list of 3072 floats.

    Raises:
        ValueError: If neither pdf_path nor video_path is provided.

    Example:
        vectors = embed_candidate_profile(
            pdf_path="resume.pdf",
            video_path="intro.mp4",
        )
        # vectors = {
        #     "resume": [0.12, 0.87, ...],   ← 3072-dim
        #     "video":  [0.33, 0.91, ...],   ← 3072-dim
        # }

        # Store in Qdrant:
        # qdrant.upsert(collection="candidates", points=[
        #     PointStruct(id=candidate_id, vector=vectors, payload={...})
        # ])
    """
    if not pdf_path and not video_path:
        raise ValueError("At least one of pdf_path or video_path must be provided.")

    vectors = {}

    if pdf_path:
        logger.info("── Embedding Resume PDF ──")
        vectors["resume"] = embed_pdf(pdf_path)

    if video_path:
        logger.info("── Embedding Video Introduction ──")
        vectors["video"] = embed_video(video_path)

    logger.info(f"[DONE] Generated embeddings for: {list(vectors.keys())}")
    return vectors


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test core text embedding
    sample_text = "Experienced ML Engineer with 3 years in NLP and PyTorch"
    vector = embed_text(sample_text)
    print(f"✅ Text embedding dim: {len(vector)}")  # Should print 3072

    # Uncomment to test with real files:
    # vector = embed_pdf("resume.pdf")
    # print(f"✅ PDF embedding dim: {len(vector)}")

    # vector = embed_video("intro.mp4")
    # print(f"✅ Video embedding dim: {len(vector)}")