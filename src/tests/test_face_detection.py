from services.faceDetectionService import FaceDetectionService
from PIL import Image 
import io
import pytest
from facenet_pytorch import MTCNN
from unittest.mock import AsyncMock, patch
import torch
from exception.exceptions import FaceNotDetected

@pytest.fixture
def faceDetectionService():
    return FaceDetectionService()

@pytest.fixture
def generateMockImage():
    img = Image.new("RGB", (200, 200), color="blue")
    imgBytes = io.BytesIO()
    img.save(imgBytes, format="JPEG")
    return imgBytes.getvalue()

@pytest.fixture
def generateLargeImage():
    img = Image.new("RGB", (5000, 5000), color="blue")
    imgBytes = io.BytesIO()
    img.save(imgBytes, format="JPEG")
    return imgBytes.getvalue()

def test_optimiseImageSync_resize(faceDetectionService, generateLargeImage):
    image  = faceDetectionService.optimiseImageSync(generateLargeImage)
    assert max(image.size) == faceDetectionService.maxImageSize

def test_optimiseImageSync_rgbConversion(faceDetectionService, generateMockImage):
    img = Image.open(io.BytesIO(generateMockImage)).convert("L")
    imgBytes = io.BytesIO()
    img.save(imgBytes, format="JPEG")
    convertedImg = faceDetectionService.optimiseImageSync(imgBytes.getvalue())
    assert convertedImg.mode == "RGB"

@pytest.mark.asyncio
async def test_optimiseImage(faceDetectionService, generateLargeImage):
    image  = await faceDetectionService.optimiseImage(generateLargeImage)
    assert max(image.size) == faceDetectionService.maxImageSize

@pytest.mark.asyncio
@patch.object(MTCNN, '__call__', new_callable=AsyncMock)
async def test_detectFace_success(mockMtcnn, faceDetectionService, generateMockImage):
    print(f"{mockMtcnn}ffffffffffffffffffffffffffffff")
    mockMtcnn.return_value = (torch.rand(1,2,160,160), 0.99)
    faceTensor, prob = await faceDetectionService.detectFace(generateMockImage)
    assert faceTensor is not None
    assert prob >= faceDetectionService.threshold

@pytest.mark.asyncio
@patch.object(MTCNN, '__call__', new_callable=AsyncMock)
async def test_detectFace_noFace(mockMtcnn, faceDetectionService, generateMockImage):
    mockMtcnn.return_value = (None, 0)
    with pytest.raises(FaceNotDetected):
        await faceDetectionService.detectFace(generateMockImage)
