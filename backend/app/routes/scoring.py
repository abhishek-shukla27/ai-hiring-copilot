from fastapi import APIRouter
from pydantic import BaseModel
from app.services.matching.skill_matcher import match_skills

router = APIRouter()

class ScoreRequest(BaseModel):
    resume_skills: list
    jd_skills: list

@router.post("/score")
async def score(req: ScoreRequest):
    result = match_skills(req.resume_skills, req.jd_skills)
    return result
