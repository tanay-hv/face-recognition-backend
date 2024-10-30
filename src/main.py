from fastapi import FastAPI

app = FastAPI(title="Facial Recognition System")

@app.get("/")
async def root():
    return {"message": "Face Recognition System is running"}

if __name__ ==  "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)