from fastapi import APIRouter
from models.user import UserReq, UserRes
import uuid

router = APIRouter()

@router.post("/addUser")
async def addUser(user : UserReq) -> UserRes:
    userId = str(uuid.uuid4())
    message = f"{user.name} was added successfully."
    return UserRes(message=message, userId=userId)