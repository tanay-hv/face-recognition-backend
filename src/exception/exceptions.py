from fastapi import HTTPException

class BaseException(HTTPException):
    def __init__(self, statusCode : int, errorCode : int, message : str):
        super().__init__(status_code=statusCode, detail={
            "status" : "error",
            "errorCode" : errorCode,
            "message" : message,
        })

class BadRequest(BaseException):
    def __init__(self, message: str = "Bad Request"):
        super().__init__(statusCode=400, errorCode=400, message=message) 

class NotFound(BaseException):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(statusCode=404, errorCode=404, message=message) 

class DuplicateEntry(BaseException):
    def __init__(self, message: str = "Duplicate Entry"):
        super().__init__(statusCode=409, errorCode=409, message=message) 

class InternalServer(BaseException):
    def __init__(self, message: str = "Internal server error"):
        super().__init__(statusCode=500, errorCode=500, message=message) 

class ServiceUnavailable(BaseException):
    def __init__(self, message: str = "Service unavailable"):
        super().__init__(statusCode=503, errorCode=503, message=message) 

class ServiceUnavailable(BaseException):
    def __init__(self, message: str = "Service unavailable"):
        super().__init__(statusCode=503, errorCode=503, message=message) 

class FaceNotDetected(BaseException):
    def __init__(self, message: str = "No face was detected"):
        super().__init__(statusCode=400, errorCode=601, message=message) 

class LowSimilarityScore(BaseException):
    def __init__(self, message: str = "Similarity score is below threshold"):
        super().__init__(statusCode=400, errorCode=602, message=message) 