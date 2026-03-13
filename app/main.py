from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import tempfile
import shutil
import os

from .vector_store import init_collections, search_text_collections, search_similar_images
from .ingestion import index_resume_pdf, index_av_file, index_image_file


app = FastAPI(title="RAG Multimodal Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_collections()


@app.post("/ingest/resume")
async def ingest_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for resumes.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = index_resume_pdf(tmp_path, original_name=file.filename)
    finally:
        os.remove(tmp_path)

    return {"status": "ok", "id": doc_id}


@app.post("/ingest/qa-av")
async def ingest_qa_av(file: UploadFile = File(...)):
    # Accept common audio/video containers; ffmpeg will handle specifics.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = index_av_file(tmp_path, original_name=file.filename)
    finally:
        os.remove(tmp_path)

    return {"status": "ok", "id": doc_id}


@app.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = index_image_file(tmp_path, original_name=file.filename)
    finally:
        os.remove(tmp_path)

    return {"status": "ok", "id": doc_id}


@app.post("/query/text")
async def query_text(query: str, top_k: int = 5):
    resume_hits, qa_hits = search_text_collections(query, top_k=top_k)
    return JSONResponse(
        {
            "resumes": [hit.dict() for hit in resume_hits],
            "qa_transcripts": [hit.dict() for hit in qa_hits],
        }
    )


@app.post("/query/image")
async def query_image(file: UploadFile = File(...), top_k: int = 5):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        hits = search_similar_images(tmp_path, top_k=top_k)
    finally:
        os.remove(tmp_path)

    return JSONResponse({"images": [hit.dict() for hit in hits]})

