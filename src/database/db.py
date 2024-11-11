import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import Optional, Dict, Any
from contextlib import contextmanager
from exception.exceptions import ServiceUnavailable

class Database:
    def __init__(self, connectionString: str):
        self.connectionString = connectionString
        self.initDb()

    @contextmanager
    def getConnection(self):
        try:
            conn = psycopg2.connect(self.connectionString)
            register_vector(conn) 
            yield conn
        except Exception as e:
            raise ServiceUnavailable(message="Database connection failed") from e
        finally:
            if 'conn' in locals():
                conn.close()

    def initDb(self):
        from database.queries import (
            ENABLE_VECTOR_EXTENSION,
            CREATE_USER_TABLE,
            CREATE_VECTOR_INDEX
        )
        
        with self.getConnection() as conn:
            with conn.cursor() as cur:
                cur.execute(ENABLE_VECTOR_EXTENSION)
                cur.execute(CREATE_USER_TABLE)
                cur.execute(CREATE_VECTOR_INDEX)
            conn.commit()

    async def addUser(self, userId: str, name: str, birthdate: str, faceVectors: np.ndarray) -> Dict[str, Any]:
        from database.queries import INSERT_USER
        
        try:
            with self.getConnection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(INSERT_USER, (
                        userId,
                        name,
                        birthdate,
                        faceVectors 
                    ))
                    result = cur.fetchone()
                conn.commit()
                return dict(result)
        except Exception as e:
            raise ServiceUnavailable(message="Failed to add user") from e

    async def findSimilarFaces(self, vectors: np.ndarray, threshold: float) -> Optional[Dict[str, Any]]:
        from database.queries import FIND_SIMILAR_FACE
        
        try:
            with self.getConnection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(FIND_SIMILAR_FACE, (vectors, vectors))
                    result = cur.fetchone()
                    
                    if result and float(result['similarity']) >= threshold:
                        return {
                            'id': result['id'],
                            'name': result['name']
                        }
                    return None
        except Exception as e:
            raise ServiceUnavailable(message="Failed to search for similar faces") from e