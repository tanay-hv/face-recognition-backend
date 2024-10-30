from fastapi import FastAPI
from routes.recogniseUser import router as recogniseUser
from routes.addUser import router as addUser

app = FastAPI(title="Facial Recognition System")

app.include_router(recogniseUser, prefix="/api")
app.include_router(addUser, prefix="/api")

if __name__ ==  "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)