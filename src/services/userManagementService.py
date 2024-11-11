from database.db import Database
from database.cache import Cache
from models.user import UserReq, UserRes
import uuid
import config
from exception.exceptions import BadRequest, InternalServer

class UserManagementService:
    def __init__(self):
        self.db = Database(connectionString=config.DB_CONNECTION_STRING)
        self.cache = Cache(host=config.REDIS_HOST, port=config.REDIS_PORT, password=config.REDIS_PASSWORD)

    async def addUser(self, req: UserReq) -> UserRes:

        vectors = await self.cache.getCachedVectors(req.reqId)
        
        try:
            userId = str(uuid.uuid4())

            await self.db.addUser(
                userId=userId,
                name=req.name,
                birthdate=req.birthdate,
                faceVectors=vectors
            )
            
            await self.cache.deleteCachedVectors(req.reqId)
            
            return UserRes(
                message=f"{req.name} was added",
                userId=userId
            )
        
        except Exception as e:
            raise InternalServer(message="Failed to add user") from e