import os
from typing import List, Union

ENV_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import field_validator

    class Settings(BaseSettings):
        PROJECT_NAME: str = "Customer Churn Intelligence Platform"
        API_PREFIX: str = "/api"
        
        SECRET_KEY: str = "c8f49e0b6d2a7153b940e8a71fc936e25d8a9f01bc427389ef12480ad9146ec7"
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  

        DATABASE_URL: str = "sqlite:///./churn_database.db"
        
        ALLOWED_ORIGINS: Union[List[str], str] = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "https://customerchurnprediction-id-l.vercel.app"
        ]
        
        MODEL_PATH: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "ml_pipeline", "artifacts", "churn_model.pkl")
        )

        GROQ_API_KEY: str = ""
        GROQ_MODEL: str = "openai/gpt-oss-120b"
        REDIS_URL: str = ""

        RESEND_API_KEY: str = ""
        BREVO_API_KEY: str = ""
        SMTP_HOST: str = "smtp.gmail.com"
        SMTP_PORT: int = 587
        SMTP_USER: str = ""
        SMTP_PASSWORD: str = ""
        SMTP_FROM_NAME: str = "Customer Churn Intelligence"

        @field_validator("ALLOWED_ORIGINS", mode="before")
        @classmethod
        def parse_cors_origins(cls, v):
            if isinstance(v, str):
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    import json
                    try:
                        return json.loads(v)
                    except Exception:
                        pass
                return [origin.strip() for origin in v.split(",") if origin.strip()]
            return v

        @field_validator("DATABASE_URL", mode="before")
        @classmethod
        def fix_postgres_url(cls, v: str) -> str:
            if v and v.startswith("postgres://"):
                return v.replace("postgres://", "postgresql://", 1)
            return v

        model_config = SettingsConfigDict(
            env_file=ENV_FILE_PATH,
            env_file_encoding="utf-8",
            extra="allow"
        )

except ImportError:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE_PATH)
    except ImportError:
        pass

    class Settings:
        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Customer Churn Intelligence Platform")
        API_PREFIX: str = os.getenv("API_PREFIX", "/api")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "c8f49e0b6d2a7153b940e8a71fc936e25d8a9f01bc427389ef12480ad9146ec7")
        ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

        raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./churn_database.db")
        if raw_db_url.startswith("postgres://"):
            raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
        DATABASE_URL: str = raw_db_url

        _raw_origins = os.getenv("ALLOWED_ORIGINS", "")
        if _raw_origins:
            ALLOWED_ORIGINS: Union[List[str], str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]
        else:
            ALLOWED_ORIGINS: Union[List[str], str] = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
                "https://customerchurnprediction-id-l.vercel.app",
            ]

        MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "ml_pipeline", "artifacts", "churn_model.pkl")
        ))

        GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        REDIS_URL: str = os.getenv("REDIS_URL", "")

        RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
        SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
        SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
        SMTP_USER: str = os.getenv("SMTP_USER", "")
        SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
        SMTP_FROM_NAME: str = os.getenv("SMTP_FROM_NAME", "Customer Churn Intelligence")

settings = Settings()
