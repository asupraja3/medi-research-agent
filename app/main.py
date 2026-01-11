from fastapi import FastAPI
from app.api import routes

app = FastAPI(title="MedSync Agent - Production RAG")

app.include_router(routes.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "MedSync Agent is running. Use /api/v1/ingest to load data."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)