"""
Helper for mapcat database.
"""

from typing import Literal

from pydantic import PrivateAttr
from sqlmodel import create_engine
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Engine
from pathlib import Path


class Settings(BaseSettings):  # pragma: no cover
    database_name: str = "mapcat.db"
    database_type: Literal["sqlite", "postgresql"] = "sqlite"

    depth_one_parent: Path = "/"
    atomic_parent: Path = "/"

    echo: bool = False

    model_config: SettingsConfigDict = {
        "env_prefix": "mapcat_",
    }

    _engine: Engine | None = PrivateAttr(default=None)
    _sessionmaker: sessionmaker | None = PrivateAttr(default=None)

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
    
    @property
    def engine(self) -> Engine:
        """Create a synchronous engine."""
        if self._engine is not None:
            return self._engine
        self._engine = create_engine(self.sync_database_url, echo=self.echo)
        return self._engine
    
    @property
    def session(self) -> sessionmaker:
        """Create a synchronous session maker."""
        if self._sessionmaker is not None:
            return self._sessionmaker

        self._sessionmaker = sessionmaker(self.engine, expire_on_commit=False)
        return self._sessionmaker


settings = Settings()
