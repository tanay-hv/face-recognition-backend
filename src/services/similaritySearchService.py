from database.db import Database
from database.cache import Cache
import numpy as np
from typing import Tuple, Optional, Dict, Any
import uuid
import config
from exception.exceptions import ServiceUnavailable

class SimilaritySearchService:
    def __init__(self):
        self.db = Database(connectionString=config.DB_CONNECTION_STRING)
        self.cache = Cache(host=config.REDIS_HOST, port=config.REDIS_PORT)
        self.similarityThreshold = 0.70

    async def findMatch(self, vectors: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:

        try:
            match = await self.db.findSimilarFaces(
                vectors=vectors,
                threshold=self.similarityThreshold
            )

        except Exception as e:
            raise ServiceUnavailable(message="Database unavailable") from e
        
        if match:
            return match, None

        cacheKey = str(uuid.uuid4())
        await self.cache.cacheVectors(reqId=cacheKey, vectors=vectors)
        return None, cacheKey