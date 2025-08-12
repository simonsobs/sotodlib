"""
Core database tables storing information about sources.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import JSON, Column, Field, SQLModel

from .settings import settings
