from app.services.matching.skill_matcher import match_skills

resume = ["python", "sql", "aws", "fastapi"]
jd = ["python", "sql", "machine learning"]

print(match_skills(resume, jd))
