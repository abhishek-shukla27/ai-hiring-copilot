from fastapi import APIRouter
from pydantic import BaseModel
from app.services.matching.skill_matcher import match_skills
from app.services.vector.vector_store import semantic_search
from app.services.embeddings.embedding_service import embed_text
import numpy as np
router = APIRouter()

class ScoreRequest(BaseModel):
    resume_skills: list
    jd_skills: list

class CompareRequest(BaseModel):
    resume_skills: list
    jd_skills: list
@router.post("/score")
async def score(req: ScoreRequest):
    result = match_skills(req.resume_skills, req.jd_skills)
    return result

@router.post("/semantic-match")
async def semantic_match(data: dict):
    resume_text=data["resume"]
    jd_text=data["jd"]

    resume_vec = embed_text(resume_text)
    jd_vec = embed_text(jd_text)

    score=float(np.dot(resume_vec, jd_vec) / (np.linalg.norm(resume_vec) * np.linalg.norm(jd_vec)))

    return {"semantic_score":score}
@router.post("/compare")
async def compare(resume_skills: list, jd_skills: list):
    result = match_skills(resume_skills, jd_skills)
    return result
