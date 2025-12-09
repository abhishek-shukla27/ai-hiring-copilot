from fastapi import FastAPI
from app.routes import ingest, scoring

app = FastAPI(
    title="AI Hiring Copilot",
    description="AI system to rank resumes against job descriptions",
    version="1.0.0"
)

app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(scoring.router, prefix="/score", tags=["Scoring"])

@app.get("/")
def root():
    return {"status": "AI Hiring Copilot is running ✅"}
