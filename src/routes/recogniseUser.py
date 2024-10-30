from fastapi import APIRouter, UploadFile, Depends, File
from services.faceDetectionService import FaceDetectionService
from services.featureExtractionService import FeatureExtractionService

router = APIRouter()

@router.post("/recogniseUser")
async def recogniseUser(
    image : UploadFile = File(...),
    faceDetectionService : FaceDetectionService = Depends(),
    featureExtractionService : FeatureExtractionService = Depends()
):
    imageContent = await image.read()

    facePixels, confidence = await faceDetectionService.detectFace(imageContent)

    vectors = await featureExtractionService.extractFeatures(facePixels=facePixels)

    return  {
                "confidence" : confidence,
                "vectors" : vectors
            }