from fastapi import APIRouter, UploadFile, Depends, File
from services.faceDetectionService import FaceDetectionService

router = APIRouter()

@router.post("/recogniseUser")
async def recogniseUser(
    image : UploadFile = File(...),
    faceDetectionService : FaceDetectionService = Depends()
):
    imageContent = await image.read()

    facePixels, confidence = await faceDetectionService.detectFace(imageContent)

    return {"success" : confidence}