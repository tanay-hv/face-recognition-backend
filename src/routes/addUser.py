from fastapi import APIRouter, Depends
from models.user import UserReq
from services.userManagementService import UserManagementService

router = APIRouter()

@router.post("/addUser")
async def addUser(
        user : UserReq,
        userManagementService : UserManagementService = Depends()
    ):
    result = await userManagementService.addUser(user)
    return result