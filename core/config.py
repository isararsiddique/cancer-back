from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    database_url: str = Field(..., alias="DATABASE_URL")
    jwt_secret: str = Field(default="change-me", alias="JWT_SECRET")
    jwt_issuer: str = Field(default="registry.api", alias="JWT_ISS")
    jwt_audience: str = Field(default="registry.clients", alias="JWT_AUD")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    db_enc_key: Optional[str] = Field(default=None, alias="DB_ENC_KEY")
    # WHO ICD-11 API credentials (never exposed to frontend)
    who_client_id: Optional[str] = Field(default=None, alias="WHO_CLIENT_ID")
    who_client_secret: Optional[str] = Field(default=None, alias="WHO_CLIENT_SECRET")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()
