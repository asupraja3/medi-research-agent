from fastapi import Header, HTTPException
from app.core.config import settings

async def get_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key