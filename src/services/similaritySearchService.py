from database.db import Database
from database.cache import Cache
from database.userSchema import Base
from sqlalchemy.orm import Session
import numpy as np
from typing import Tuple, Optional, Dict, Any
import uuid
import config

class SimilaritySearchService :
    def __init__(self):
        self.db = Database(connectionString = config.DB_CONNECTION_STRING)
        self.cache = Cache(host = config.REDIS_HOST, port = config.REDIS_PORT, password=config.REDIS_PASSWORD)
        self.similarityThreshold = 0.85

    async def findMatch(self, vectors : np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        match = await self.db.findSimilarFaces(
            vectors=vectors,
            threshold=self.similarityThreshold
        )

        print(f"{match.id} hhhhhhhhhhhhhhhhhhhhhhhh")

        if match :
            print(f"{match.id} hhhhhhhhhhhhhhhhhhhhhhhh")
            return {
                "userId": match.id,
                "name": match.name,
            }, None

        cacheKey = str(uuid.uuid4())
        await self.cache.cacheVectors(reqId=cacheKey, vectors=vectors)
        return None, cacheKey