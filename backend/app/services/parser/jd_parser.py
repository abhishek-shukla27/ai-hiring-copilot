import re

COMMON_SKILLS = [
    "python", "java", "sql", "machine learning", "deep learning",
    "nlp", "fastapi", "django", "flask",
    "aws", "docker", "kubernetes",
    "react", "typescript"
]

async def parse_jd(file):
    text = (await file.read()).decode("utf-8", errors="ignore").lower()

    found_skills = []
    for skill in COMMON_SKILLS:
        if re.search(rf"\b{skill}\b", text):
            found_skills.append(skill)

    return {
        "raw_text": text.strip(),
        "skills": found_skills
    }
