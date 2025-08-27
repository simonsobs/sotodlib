from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):  # pragma: no cover
    database_name: str = "mapcat.db"
    database_type: Literal["sqlite", "postgresql"] = "sqlite"

    model_config: SettingsConfigDict = {
        "env_prefix": "mapcat_model_",
    }

    @property
    def async_database_url(self) -> str:
        if self.database_type == "sqlite":
            return f"sqlite+aiosqlite:///{self.database_name}"
        if self.database_type == "postgresql":
            return f"postgresql+asyncpg://{self.database_name}"

    @property
    def sync_database_url(self) -> str:
        if self.database_type == "sqlite":
            return f"sqlite:///{self.database_name}"
        if self.database_type == "postgresql":
            return f"postgresql://{self.database_name}"


settings = Settings()
