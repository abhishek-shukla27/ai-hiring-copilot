from fastapi import APIRouter

router=APIRouter()

@router.get("/")
def scoring_health():
    return {"message":"Scoring service ready"}