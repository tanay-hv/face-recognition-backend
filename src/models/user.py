import datetime
from pydantic import BaseModel

class UserReq(BaseModel):
    name : str
    birthdate : datetime.date
    reqId : str

class UserRes(BaseModel):
    message : str
    userId : str