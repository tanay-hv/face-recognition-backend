import numpy as np
from typing import Optional
from redis import Redis
import json

class Cache:
    def __init__(self, host, port, password):
        self.redis = Redis(host=host, port=port, password=password)
        self.expiry = 900

    async def cacheVectors(self, reqId : str, vectors : np.ndarray):
        vectorsJson = json.dumps(vectors.tolist())
        self.redis.setex(
            f"vectors : {reqId}",
            self.expiry,
            vectorsJson
        )

    async def getCachedVectors(self, reqId : str) -> Optional[np.ndarray]:
        cache = self.redis.get(f"vectors : {reqId}")
        if cache:
            vectorsList = json.loads(cache)
            return np.array(vectorsList)
        return None
    
    async def deleteCachedVectors(self, reqId: str) -> None:
        self.redis.delete(f"ectors:{reqId}")