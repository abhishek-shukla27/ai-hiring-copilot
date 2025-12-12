from fastapi import APIRouter,HTTPException,UploadFile,File,Form
from pydantic import BaseModel
from app.services.matching.skill_matcher import match_skills
from app.services.vector.vector_store import VectorStore
from app.services.embeddings.embedding_service import embed_text
from app.services.scoring.scorer import explain_candidate
from typing import List,Dict,Any
import re
import io
import os
from pypdf import PdfReader
import docx
import numpy as np
from pydantic import BaseModel

router = APIRouter()
vector_store=VectorStore()

COMMON_SKILLS=[]

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