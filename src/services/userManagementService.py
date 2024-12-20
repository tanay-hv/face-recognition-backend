from database.db import Database
from models.user import UserReq, UserRes
import uuid
import config
from exception.exceptions import BadRequest, InternalServer

class UserManagementService:
    def __init__(self):
        self.db = Database(connectionString=config.DB_CONNECTION_STRING)

    async def addUser(self, req: UserReq) -> UserRes:

        vectors = await self.db.getCachedVectors(key=req.reqId)
        
        try:
            userId = str(uuid.uuid4())

            await self.db.addUser(
                userId=userId,
                name=req.name,
                birthdate=req.birthdate,
                faceVectors=vectors
            )
            
            await self.db.deleteCachedVectors(key=req.reqId)
            
            return UserRes(
                message=f"{req.name} was added",
                userId=userId
            )
        
        except Exception as e:
            raise InternalServer(message=f"Failed to add user") from e