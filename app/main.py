import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .backend import index_resume_pdf, init_collections, search_resumes


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

@app.post("/query/text")
async def query_text(query: str, top_k: int = 5):
    hits = search_resumes(query, top_k=top_k)
    return JSONResponse({"resumes": [hit.dict() for hit in hits]})
