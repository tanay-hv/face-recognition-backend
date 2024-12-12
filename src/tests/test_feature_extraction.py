import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from services.featureExtractionService import FeatureExtractionService
from facenet_pytorch import InceptionResnetV1
from exception.exceptions import BadRequest

@pytest.fixture
def featureExtractionService():
    return FeatureExtractionService()

@pytest.fixture
def generateMockFaceTensors():
    return torch.rand(1, 3, 160, 160)

@pytest.mark.asyncio
@patch.object(InceptionResnetV1, 'forward', new_callable=MagicMock)
async def test_featureExtraction_success(mockModel, generateMockFaceTensors, featureExtractionService):
    mockModel.return_value = torch.rand(1, 512)
    features = await featureExtractionService.extractFeatures(generateMockFaceTensors)
    assert isinstance(features, np.ndarray)
    assert features.shape == (512,)

@pytest.mark.asyncio
async def test_featureExtraction_invalidTensors(featureExtractionService):
    with pytest.raises(BadRequest) as exc_info:
        await featureExtractionService.extractFeatures(None)
    assert exc_info.value.detail["message"] == "invalid face tensors for feature extraction"