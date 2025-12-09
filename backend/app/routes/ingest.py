from fastapi import APIRouter,UploadFile,File
from app.services.parser.resume_parser import parse_resume
from app.services.parser.jd_parser import parse_jd

router=APIRouter()

@router.post("/resume")
async def upload_resume(file: UploadFile=File(...)):
    text=await parse_resume(file)
    return{
        "filename":file.filename,
        "extracted_text_preview":text[:500]
    }

@router.post("/jd")
async def upload_jd(file: UploadFile = File(...)):
    text, skills = await parse_jd(file)
    return {
        "filename": file.filename,
        "skills_extracted": skills,
        "jd_preview": text[:500]
    }

