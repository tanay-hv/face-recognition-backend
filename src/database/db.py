from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from database.userSchema import Base, User
import numpy as np
from typing import List, Optional

class Database:
    def __init__(self, connectionString):
        self.engine = create_engine(connectionString, echo=True)
        Base.metadata.create_all(self.engine)

    def getSession(self) -> Session:
        return Session(self.engine)

    async def addUser(self, user : User):
        with self.getSession() as session:
            session.add(user)
            session.commit()

    async def findSimilarFaces(self, vectors: np.ndarray, threshold: float) -> Optional[User]:
        with self.getSession() as session:
            mostSimilar = session.query(User).order_by(
                User.faceVectors.cosine_distance(vectors)
            ).first()

            if mostSimilar:
                dist_of_ms = session.query(
                    User.faceVectors.cosine_distance(vectors)
                ).filter(User.id == mostSimilar.id).scalar()
                
                similarity = 1 - dist_of_ms
                if similarity >= threshold:
                    return mostSimilar, similarity
            return None