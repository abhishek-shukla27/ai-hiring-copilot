from fastapi import APIRouter,HTTPException,UploadFile,File,Form
from pydantic import BaseModel
from app.services.matching.skill_matcher import match_skills
from app.services.vector.vector_store import VectorStore
from app.services.embeddings.embedding_service import embed_text
from app.services.scoring.scorer import explain_candidate
from typing import List,Dict,Any,Optional
import re
import io
import os
from pypdf import PdfReader
import docx
import numpy as np
from pydantic import BaseModel

router = APIRouter()
vector_store=VectorStore()

COMMON_SKILLS=[
    "python","java","c++","c#","javascript","node.js","react","angular",
    "fastapi","django","flask","sql","postgresql","mysql","mongodb",
    "docker","kubernetes","aws","gcp","azure","ml","machine learning",
    "deep learning", "pytorch", "tensorflow", "nlp", "spark", "redis", "git",
    "rest", "graphql", "html", "css", "typescript", "linux"

]

def extract_text_from_pdf(file_bytes: bytes)->str:
    reader=PdfReader(io.BytesIO(file_bytes))
    pages=[]
    for page in reader.pages:
        text=page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)



def clean_text(text: str) -> str:
    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove phones (simple patterns)
    text = re.sub(r"\+?\d[\d \-\(\)]{7,}\d", " ", text)
    # Remove bullet characters and weird unicode bullets
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219•\-•]", " ", text)
    # Replace multiple newlines and whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Strip
    return text.strip()

def extract_email(text: str) -> Optional[str]:
    m = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    return m.group(1).lower() if m else None

def extract_skills(text: str, skills_list: List[str] = COMMON_SKILLS) -> List[str]:
    text_lower = text.lower()
    found = []
    for skill in skills_list:
        # match whole word or with dots (node.js)
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(list(set(found)), key=lambda s: skills_list.index(s) if s in skills_list else 9999)


def extract_text_from_docx(file_bytes: bytes) -> str:
    # python-docx expects a file-like object that has a name attribute OR a path.
    # But it works with BytesIO as well.
    bio = io.BytesIO(file_bytes)
    doc = docx.Document(bio)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def clean_text(text: str) -> str:
    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove phones (simple patterns)
    text = re.sub(r"\+?\d[\d \-\(\)]{7,}\d", " ", text)
    # Remove bullet characters and weird unicode bullets
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219•\-•]", " ", text)
    # Replace multiple newlines and whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Strip
    return text.strip()

def extract_phone(text: str) -> Optional[str]:
    m = re.search(r"(\+?\d[\d\-\s\(\)]{7,}\d)", text)
    return m.group(1) if m else None

class UploadResponse(BaseModel):
    ok:bool
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    detected_skills: List[str]
    stored: bool
    message: Optional[str] = None

class ScoreRequest(BaseModel):
    resume_skills: list
    jd_skills: list

class CompareRequest(BaseModel):
    resume_skills: list
    jd_skills: list

class SemanticMatchRequest(BaseModel):
    resume: str
    jd: str

class AddResumeRequest(BaseModel):
    candidate_id: str
    name: str
    email: str
    resume_text: str
    skills: List[str]=[]

class RankRequest(BaseModel):
    job_id: str=None
    jd_text: str
    jd_skills: List[str]=[]
    top_k: int=10

@router.post("/score")
async def score(req: ScoreRequest):
    result = match_skills(req.resume_skills, req.jd_skills)
    return result

@router.post("/semantic_search")
async def semantic_match(req: SemanticMatchRequest):
    
    resume_vec = embed_text(req.resume)
    jd_vec = embed_text(req.jd)

    resume_vec=np.array(resume_vec)
    jd_vec=np.array(jd_vec)

    score=float(np.dot(resume_vec, jd_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(jd_vec)))

    return {"semantic_score":score}
@router.post("/compare")
async def compare(req:CompareRequest):
    result = match_skills(req.resume_skills, req.jd_skills)
    return result

@router.post("/add_resume")
async def add_resume(req: AddResumeRequest):
    vec=embed_text(req.resume_text)
    if not isinstance(vec,(list,tuple)):
        raise HTTPException(status_code=500, detail="embed_text must return a list/array of floats")
    
    meta = {
        "candidate_id": req.candidate_id,
        "name": req.name,
        "email": req.email,
        "resume_text": req.resume_text,
        "skills": req.skills
    }

    try:
        vector_store.add_vector(vec, meta)
        vector_store.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "message": "Resume added"}

@router.post("/rank")
async def rank_candidates(req: RankRequest):
    # embed JD
    jd_vec = embed_text(req.jd_text)
    if not isinstance(jd_vec, (list, tuple)):
        raise HTTPException(status_code=500, detail="embed_text must return a list/array of floats")

    # search vector store
    results = vector_store.search(jd_vec, top_k=req.top_k)

    # build explainability for each result
    ranked = []
    for r in results:
        meta = r["metadata"]
        resume_text = meta.get("resume_text", "")
        resume_skills = meta.get("skills", [])
        explanation = explain_candidate(
            jd_text=req.jd_text,
            resume_text=resume_text,
            jd_vec=jd_vec,
            resume_vec=embed_text(resume_text),  # re-embed (cheap) — or store embedding in metadata to avoid re-embedding
            resume_skills=resume_skills,
            jd_skills=req.jd_skills
        )
        # include faiss score too
        explanation["faiss_score"] = r["score"]
        explanation["candidate"] = {
            "candidate_id": meta.get("candidate_id"),
            "name": meta.get("name"),
            "email": meta.get("email")
        }
        ranked.append(explanation)

    # sort by final_score descending (faiss returns in order but final score may reorder)
    ranked = sorted(ranked, key=lambda x: x["final_score"], reverse=True)
    return {"results": ranked}

@router.post("/upload_resume",response_model=UploadResponse)
async def upload_resume(
    candidate_id: str=Form(...),
    name: Optional[str]=Form(None),
    file: UploadFile=File(...),
    extra_skills: Optional[str]=Form(None)
):
    """
    Upload resume file (pdf/docx/txt). Required form fields:
    - candidate_id (string)
    - file (UploadFile)
    Optional:
    - name
    - extra_skills (comma-separated)
    """

    filename=file.filename or "resume"
    content_type=file.content_type or ""
    allowed=("pdf","docx","txt")
    ext=(os.path.splitext(filename)[1] or "").lower().strip(".")
    if ext not in allowed and "pdf" not in content_type and "word" not in content_type and "text" not in content_type:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload pdf/docx/txt")

    file_bytes=await file.read()

    parsed_text=""
    try:
        if ext=="pdf":
            parsed_text=extract_text_from_pdf(file_bytes)
        elif ext=="docx":
            parsed_text=extract_text_from_docx(file_bytes)
        
        elif ext=="txt":
            parsed_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PDF, DOCX, or TXT."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse uploaded file: {e}")

    if not parsed_text or len(parsed_text.strip())==0:
        raise HTTPException(status_code=400,detail="No text could be extracted from the uploaded file")
    
    cleaned=clean_text(parsed_text)

    email = extract_email(parsed_text)
    phone = extract_phone(parsed_text)
    detected_skills=extract_skills(parsed_text)

    if extra_skills:
        user_skills = [s.strip() for s in extra_skills.split(",") if s.strip()]
        for s in user_skills:
            if s.lower() not in [x.lower() for x in detected_skills]:
                detected_skills.append(s)

    try:
        embedding = embed_text(cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    if not isinstance(embedding, (list, tuple)):
        # some models return numpy arrays; convert to list
        try:
            embedding = list(embedding)
        except Exception:
            raise HTTPException(status_code=500, detail="embed_text must return a list/array of floats")

    meta = {
        "candidate_id": candidate_id,
        "name": name,
        "email": email,
        "phone": phone,
        "filename": filename,
        "raw_text": parsed_text,
        "cleaned_text": cleaned,
        "skills": detected_skills
    }

    try:
        vector_store.add_vector(embedding, meta)
        # The vector_store.save() persists index+metadata to disk
        vector_store.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store vector: {e}")

    return {
        "ok": True,
        "candidate_id": candidate_id,
        "name": name,
        "email": email,
        "phone": phone,
        "detected_skills": detected_skills,
        "stored": True,
        "message": "Resume parsed, embedded and stored in vector store"
    }