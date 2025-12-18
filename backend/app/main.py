from fastapi import FastAPI,Request
from app.routes import ingest, scoring
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
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




BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),"..",".."))
app.mount("/static",StaticFiles(directory=os.path.join(BASE_DIR,"static")),name="static")
templates=Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.get("/upload")
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html",{"request":request})
@app.get("/rank-ui")
def rank_page(request: Request):
    return templates.TemplateResponse("rank.html",{"request":request})