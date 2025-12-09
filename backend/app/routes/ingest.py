from fastapi import APIRouter, UploadFile, File
from app.services.parser.resume_parser import parse_resume
from app.services.parser.jd_parser import parse_jd

router = APIRouter()

# ✅ Resume Upload
@router.post("/resume")
async def upload_resume(file: UploadFile = File(...)):
    parsed_resume = await parse_resume(file)

    return {
        "filename": file.filename,
        "message": "Resume uploaded successfully ✅",
        "resume_length": parsed_resume["length"],
        "extracted_text_preview": parsed_resume["raw_text"][:500]
    }


# ✅ Job Description Upload
@router.post("/jd")
async def upload_jd(file: UploadFile = File(...)):
    parsed_jd = await parse_jd(file)

    return {
        "filename": file.filename,
        "message": "JD uploaded successfully ✅",
        "skills_extracted": parsed_jd["skills"],
        "jd_preview": parsed_jd["raw_text"][:500]
    }
