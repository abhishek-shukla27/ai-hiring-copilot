
def match_skills(resume_skills: list, jd_skills: list):
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = list(resume_set & jd_set)
    missing = list(jd_set - resume_set)

    score = 0
    if jd_skills:
        score = round((len(matched) / len(jd_skills)) * 100, 2)

    return {
        "match_score": score,
        "matched_skills": matched,
        "missing_skills": missing
    }
