from fastapi import APIRouter
from pydantic import BaseModel
from app.services.matching.skill_matcher import match_skills
from app.services.vector.vector_store import semantic_search
from app.services.embeddings.embedding_service import embed_text
import numpy as np
from pydantic import BaseModel
router = APIRouter()

class ScoreRequest(BaseModel):
    resume_skills: list
    jd_skills: list

class CompareRequest(BaseModel):
    resume_skills: list
    jd_skills: list

class SemanticMatchRequest(BaseModel):
    resume: str
    jd: str
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
