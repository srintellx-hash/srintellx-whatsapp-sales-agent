"""Central configuration.

All secrets and tunables are loaded from environment variables (.env locally,
Railway variables in production). Nothing sensitive is hard-coded.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- App -----
    app_name: str = "SrintellX WhatsApp Sales Agent"
    environment: str = Field(default="development")  # development | production
    log_level: str = Field(default="INFO")
    timezone: str = Field(default="Asia/Kolkata")

    # ----- Database -----
    # Railway provides DATABASE_URL. We normalise it to the async driver.
    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/srintellx")

    @field_validator("database_url")
    @classmethod
    def _force_async_driver(cls, v: str) -> str:
        # Railway/Heroku style URLs use the sync driver name.
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://") and "+asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @property
    def sync_database_url(self) -> str:
        """Alembic runs migrations synchronously."""
        return self.database_url.replace("+asyncpg", "+psycopg2").replace(
            "postgresql+psycopg2", "postgresql"
        ).replace("postgresql", "postgresql+psycopg2", 1)

    # ----- LLM (Groq — OpenAI-compatible) -----
    llm_api_key: str = Field(default="")
    llm_base_url: str = Field(default="https://api.groq.com/openai/v1")
    llm_model: str = Field(default="llama-3.3-70b-versatile")
    llm_max_tokens: int = Field(default=600)
    llm_temperature: float = Field(default=0.3)

    # ----- WhatsApp / Meta -----
    whatsapp_token: str = Field(default="")               # permanent system-user token
    whatsapp_phone_number_id: str = Field(default="")     # from WhatsApp > API setup
    whatsapp_verify_token: str = Field(default="")        # arbitrary string you choose
    whatsapp_app_secret: str = Field(default="")          # app secret for signature checks
    whatsapp_api_version: str = Field(default="v21.0")
    whatsapp_api_base: str = Field(default="https://graph.facebook.com")

    # ----- Google Calendar -----
    google_calendar_id: str = Field(default="primary")
    # Path OR raw JSON of a service-account key (Railway: paste JSON into the var).
    google_service_account_json: str = Field(default="")
    google_service_account_file: str = Field(default="")
    demo_organizer_email: str = Field(default="")  # who hosts the demo (Rajesh)

    # ----- Sales / business rules -----
    escalation_contact_name: str = Field(default="Rajesh")
    company_city: str = Field(default="Bangalore")
    demo_duration_minutes: int = Field(default=30)

    # ----- Security -----
    rate_limit_per_minute: int = Field(default=20)   # inbound msgs per sender / minute
    allowed_origins: str = Field(default="*")  # comma-separated or just "*"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
