import pytest
from services.similaritySearchService import SimilaritySearchService
from unittest.mock import MagicMock, AsyncMock
import numpy as np

@pytest.fixture
def similaritySearchService():
    service = SimilaritySearchService()
    service.db = AsyncMock()
    service.cache = AsyncMock()
    return service

@pytest.mark.asyncio
async def test_findMatch_success(similaritySearchService):
    mockVectors = np.random.rand(512)
    mockResult = {
        "id" : "12345",
        "name" : "tanay"
    }
    print(f"{mockResult}fffffffffffffffff")
    similaritySearchService.db.findSimilarFaces.return_value = mockResult
    print(f"{mockResult}fffffffffffffffff")
    match, cacheKey = await similaritySearchService.findMatch(mockVectors)
    assert match == mockResult
    assert cacheKey is None

@pytest.mark.asyncio
async def test_findMatch_noMatch(similaritySearchService):
    mockVectors = np.random.rand(512)
    similaritySearchService.db.findSimilarFaces.return_value = None
    cacheKey = "keyyyyy"
    similaritySearchService.cache.cacheVectors.return_value = None
    match, cacheKey = await similaritySearchService.findMatch(mockVectors)
    assert match is None
    assert cacheKey is not None