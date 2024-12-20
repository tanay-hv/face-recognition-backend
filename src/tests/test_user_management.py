import pytest
from services.userManagementService import UserManagementService
from unittest.mock import AsyncMock
from models.user import UserReq, UserRes
import numpy as np

@pytest.fixture
def userManagementService():
    service = UserManagementService()
    service.db = AsyncMock()
    service.cache = AsyncMock()
    return service

@pytest.mark.asyncio
async def test_addUser_success(userManagementService):
    req = UserReq(
        name="Tanay",
        birthdate="2002-02-20",
        reqId="12345"
    )
    mockVectors = np.random.rand(512)
    mockRes = {
        "id" : "12345678",
        "name" : "Tanay",
        "birthdate" : "2002-02-20"
    }
    userManagementService.cache.getCachedVectors.return_value = mockVectors
    userManagementService.db.addUser.return_value = mockRes
    userManagementService.cache.deleteCachedVectors.return_value = None
    result = await userManagementService.addUser(req)
    assert isinstance(result, UserRes)
    assert result.message == "Tanay was added"
    assert isinstance(result.userId, str)
    assert len(result.userId) > 0