from database.db import Database
from database.cache import Cache
from models.user import UserReq, UserRes
import uuid
from database.userSchema import User
import config

class UserManagementService:
    def __init__(self):
        self.db = Database(connectionString = config.DB_CONNECTION_STRING)
        self.cache = Cache(host = config.REDIS_HOST, port = config.REDIS_PORT, password=config.REDIS_PASSWORD)

    async def addUser(self, req : UserReq) -> UserRes:
        vectors = await self.cache.getCachedVectors(req.reqId)

        newUser = User(
            id=str(uuid.uuid4()),
            name=req.name,
            birthdate=req.birthdate,
            faceVectors=vectors
        )

        with self.db.getSession() as session:
            session.add(newUser)
            session.commit()
            session.refresh(newUser)

        return UserRes(
            message=f"{req.name} was added",
            userId=newUser.id
        )