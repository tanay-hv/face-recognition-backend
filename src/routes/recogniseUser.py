from fastapi import APIRouter, UploadFile

router = APIRouter()

@router.post("/recogniseUser")
async def recogniseUser(image : UploadFile):
    return {"success" : image.filename}