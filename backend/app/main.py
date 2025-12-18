from fastapi import FastAPI,Request
from app.routes import ingest, scoring
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

app.include_router(ingest.router,prefix="/ingest",tags=["Ingestion"])

app=FastAPI()
app.mount("/static",StaticFiles(directory="static"),name="static")
templates=Jinja2Templates(directory="templates")
