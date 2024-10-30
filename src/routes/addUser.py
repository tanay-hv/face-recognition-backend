from fastapi import APIRouter
from pydantic import BaseModel
import datetime

class User(BaseModel):
    name : str
    age : int

router = APIRouter()

@router.post("/addUser")
async def addUser(user : User):
    return {"success": user}