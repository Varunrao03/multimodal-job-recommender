import React, { useState } from "react";

const API_BASE = "http://localhost:8000";

async function uploadFile(endpoint, file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Upload failed: ${res.statusText}`);
  }
  return res.json();
}

async function textQuery(query) {
  const params = new URLSearchParams({ query });
  const res = await fetch(`${API_BASE}/query/text?${params.toString()}`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(`Query failed: ${res.statusText}`);
  }
  return res.json();
}

function SectionCard({ title, description, children }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <p className="card-description">{description}</p>
      {children}
    </div>
  );
}

export default function App() {
  const [resumeStatus, setResumeStatus] = useState("");
  const [qaStatus, setQaStatus] = useState("");
  const [imageStatus, setImageStatus] = useState("");
  const [query, setQuery] = useState("");
  const [queryResult, setQueryResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleResumeUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setResumeStatus("Uploading...");
    try {
      const res = await uploadFile("/ingest/resume", file);
      setResumeStatus(`Indexed resume with id: ${res.id}`);
    } catch (err) {
      setResumeStatus(err.message);
    }
  };

  const handleQaUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setQaStatus("Uploading & transcribing...");
    try {
      const res = await uploadFile("/ingest/qa-av", file);
      setQaStatus(`Indexed QA AV with id: ${res.id}`);
    } catch (err) {
      setQaStatus(err.message);
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageStatus("Uploading & embedding...");
    try {
      const res = await uploadFile("/ingest/image", file);
      setImageStatus(`Indexed image with id: ${res.id}`);
    } catch (err) {
      setImageStatus(err.message);
    }
  };

  const handleTextQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setQueryResult(null);
    try {
      const res = await textQuery(query.trim());
      setQueryResult(res);
    } catch (err) {
      setQueryResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>RAG Multimodal Platform</h1>
        <p>Ingest and search resumes, QA sessions, and images.</p>
      </header>

      <main className="grid">
        <SectionCard
          title="Resume (PDF)"
          description="Upload resumes as PDFs. They will be parsed, embedded, and stored in Qdrant."
        >
          <input type="file" accept="application/pdf" onChange={handleResumeUpload} />
          {resumeStatus && <p className="status">{resumeStatus}</p>}
        </SectionCard>

        <SectionCard
          title="QA Audio / Video"
          description="Upload recordings of interviews or QA sessions. Audio is transcribed and indexed."
        >
          <input type="file" accept="audio/*,video/*" onChange={handleQaUpload} />
          {qaStatus && <p className="status">{qaStatus}</p>}
        </SectionCard>

        <SectionCard
          title="Images (CLIP)"
          description="Upload images to store CLIP embeddings for similarity search."
        >
          <input type="file" accept="image/*" onChange={handleImageUpload} />
          {imageStatus && <p className="status">{imageStatus}</p>}
        </SectionCard>
      </main>

      <section className="query-section">
        <h2>Text Query over Resumes & QA</h2>
        <form onSubmit={handleTextQuery} className="query-form">
          <input
            type="text"
            placeholder="Ask something like: 'Find candidates with Python backend experience'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={loading}>
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {queryResult && (
          <div className="results">
            {queryResult.error && <p className="status error">{queryResult.error}</p>}

            {queryResult.resumes && (
              <div className="result-block">
                <h3>Resumes</h3>
                {queryResult.resumes.length === 0 && <p>No matches.</p>}
                {queryResult.resumes.map((hit) => (
                  <div key={hit.id} className="result-item">
                    <p>
                      <strong>ID:</strong> {hit.id}
                    </p>
                    <p>
                      <strong>Score:</strong> {hit.score?.toFixed(4)}
                    </p>
                    <p>
                      <strong>File:</strong> {hit.payload?.file_name}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {queryResult.qa_transcripts && (
              <div className="result-block">
                <h3>QA Transcripts</h3>
                {queryResult.qa_transcripts.length === 0 && <p>No matches.</p>}
                {queryResult.qa_transcripts.map((hit) => (
                  <div key={hit.id} className="result-item">
                    <p>
                      <strong>ID:</strong> {hit.id}
                    </p>
                    <p>
                      <strong>Score:</strong> {hit.score?.toFixed(4)}
                    </p>
                    <p>
                      <strong>File:</strong> {hit.payload?.file_name}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

