import numpy as np
from typing import List,Dict,Tuple
from app.services.matching.skill_matcher import match_skills

DEFAULT_WEIGHTS={
    "semantic":  0.6,
    "keyword": 0.3,
    "skill": 0.1
}

def cosine_similarity(a: List[float],b: List[float])-> float:
    a=np.array(a,dtype=float)
    b=np.array(b,dtype=float)

    if np.linalg.norm(a)==0 or np.linalg.norm(b) ==0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b)))

def keyword_score(jd_text: str, resume_text: str) -> Tuple[float, List[str]]:
    # Very simple keyword extraction by splitting and lowercasing unique words
    # For production, replace with proper NLP keyword extraction (RAKE / YAKE / spaCy)
    jd_tokens = set([t.strip().lower() for t in jd_text.split() if len(t) > 2])
    resume_tokens = set([t.strip().lower() for t in resume_text.split() if len(t) > 2])

    common = jd_tokens.intersection(resume_tokens)
    if len(jd_tokens) == 0:
        score = 0.0
    else:
        score = len(common) / len(jd_tokens)
    return score, list(common)[:20]

def weighted_score(semantic_score: float, keyword_s: float, skill_s: float, weights=DEFAULT_WEIGHTS):
    w = weights
    final = w["semantic"] * semantic_score + w["keyword"] * keyword_s + w["skill"] * skill_s
    return float(final)

def explain_candidate(jd_text: str, resume_text: str, jd_vec: List[float], resume_vec: List[float], resume_skills: List[str], jd_skills: List[str]):
    sem = cosine_similarity(jd_vec, resume_vec)
    kw_score, common_keywords = keyword_score(jd_text, resume_text)
    skill_result = match_skills(resume_skills, jd_skills)  # returns dict with matched / missing etc.

    # skill_score: proportion of JD skills present
    num_jd_skills = max(1, len(jd_skills))
    skill_score = len(skill_result.get("matched", [])) / num_jd_skills

    final = weighted_score(sem, kw_score, skill_score)

    explanation = {
        "final_score": final,
        "semantic_score": sem,
        "keyword_score": kw_score,
        "keyword_matches": common_keywords,
        "skill_score": skill_score,
        "skill_matches": skill_result.get("matched", []),
        "missing_skills": skill_result.get("missing", []),
        "raw": {
            "jd_skills": jd_skills,
            "resume_skills": resume_skills
        }
    }
    return explanation