from typing import List,Dict

def normalize_skill(s:str)->str:
    return s.strip().lower()

def match_skills(resume_skills: List[str],jd_skills: List[str])->Dict:
    rs = {normalize_skill(x) for x in resume_skills if x}
    js = [normalize_skill(x) for x in jd_skills if x]

    matched = [s for s in js if s in rs]
    missing = [s for s in js if s not in rs]

    return {
        "matched": matched,
        "missing": missing,
        "match_count": len(matched),
        "jd_skill_count": len(js)
    }
