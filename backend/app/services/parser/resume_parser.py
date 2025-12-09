from pdfminer.high_level import extract_text
from docx import Document
import tempfile
import os
from fastapi import UploadFile


async def parse_resume(file: UploadFile) -> dict:
    suffix = file.filename.split(".")[-1].lower()

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == "pdf":
            text = extract_text(tmp_path)

        elif suffix in ["docx", "doc"]:
            doc = Document(tmp_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        else:
            text = "Unsupported file format"

    finally:
        os.remove(tmp_path)

    return {
        "raw_text": text.strip(),
        "length": len(text.strip())
    }
