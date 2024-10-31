from fastapi import APIRouter, UploadFile, Depends, File
from services.faceDetectionService import FaceDetectionService
from services.featureExtractionService import FeatureExtractionService
from services.similaritySearchService import SimilaritySearchService

router = APIRouter()

@router.post("/recogniseUser")
async def recogniseUser(
    image : UploadFile = File(...),
    faceDetectionService : FaceDetectionService = Depends(),
    featureExtractionService : FeatureExtractionService = Depends(),
    similaritySearchService : SimilaritySearchService = Depends(),
):
    imageContent = await image.read()

    facePixels, confidence = await faceDetectionService.detectFace(imageContent)

    vectors = await featureExtractionService.extractFeatures(facePixels=facePixels)

    matchResult, cacheKey = await similaritySearchService.findMatch(vectors=vectors)

    if matchResult:
        return {
            "status": "success",
            "message": "User recognised",
            "user": matchResult
        }
    else:
        return {
            "status": "no_match",
            "message": "No matching user found.",
            "reqId": cacheKey
        }