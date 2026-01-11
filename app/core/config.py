import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedSync Agent"
    OPENAI_API_KEY: str
    API_SECRET_KEY: str
    DATA_PATH: str
    VECTOR_DB_PATH: str

    class Config:
        env_file = ".env"

settings = Settings()
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY