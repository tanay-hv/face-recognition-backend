from fastapi import FastAPI, Request
from routes.recogniseUser import router as recogniseUser
from routes.addUser import router as addUser
from exception.exceptions import BaseException
from fastapi.responses import JSONResponse

app = FastAPI(title="Facial Recognition System")

@app.exception_handler(BaseException)
async def baseHandler(req : Request, err : BaseException):
    return(
        JSONResponse(
            status_code = err.status_code,
            content = err.detail
        )
    )

@app.exception_handler(Exception)
async def generalHandler(req : Request, err : Exception):
    return(
        JSONResponse(
            status_code=500,
            content={
                "status" : "error",
                "errorCode" : 500,
                "message" : str(err)
            }
        )
    )

app.include_router(recogniseUser, prefix="/api")
app.include_router(addUser, prefix="/api")

if __name__ ==  "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)