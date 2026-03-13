This is the RAG multi-modal platform for job recommendation.

## Setup

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Run Qdrant** (example with Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

- **Environment variables**:

Set your OpenAI API key (for embeddings) before running the app:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Run the API

From the project root:

```bash
uvicorn app.main:app --reload
```

Then open the interactive docs at:

- Swagger UI: `http://localhost:8000/docs`

## Frontend (React + Vite)

The `frontend` folder contains a React UI that talks to the FastAPI backend.

From the `frontend` folder:

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173` and is configured (via CORS) to call the backend at `http://localhost:8000`.

## Endpoints

- `POST /ingest/resume`: Upload a PDF resume; it will be parsed, embedded with `text-embedding-3-large`, and stored in Qdrant.
- `POST /ingest/qa-av`: Upload an audio/video QA file; it will be converted with `ffmpeg`, transcribed with Whisper, embedded, and stored in Qdrant.
- `POST /ingest/image`: Upload an image; it will be converted to CLIP embeddings and stored in Qdrant.
- `POST /query/text`: Text query over resumes and QA transcripts (RAG-ready retrieval).
- `POST /query/image`: Image similarity search over stored images.
