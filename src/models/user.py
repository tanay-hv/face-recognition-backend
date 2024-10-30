from pydantic import BaseModel
import datetime

class UserReq(BaseModel):
    name : str
    birthdate : datetime.date
    reqId : str

class UserRes(BaseModel):
    message : str
    userId : str