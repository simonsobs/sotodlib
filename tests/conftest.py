import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def run_migration(database_path: str):
    """
    Run the migration on the database.
    """
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    database_url = f"sqlite:///{database_path}"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "heads")

    return


@pytest_asyncio.fixture(scope="session", autouse=True)
async def database_async_sesionmaker(tmp_path_factory):
    """
    Create a temporary SQLite database for testing. This is a
    somewhat tricky scenario as we must create the database using
    the synchronous engine, but access it using the asynchronous
    engine.
    """

    tmp_path = tmp_path_factory.mktemp("mapcat")
    # Create a temporary SQLite database for testing.
    database_path = tmp_path / "test.db"

    # Run the migration on the database. This is blocking.
    run_migration(database_path)

    database_url = f"sqlite+aiosqlite:///{database_path}"

    async_engine = create_async_engine(database_url, echo=True, future=True)

    yield async_sessionmaker(bind=async_engine, expire_on_commit=False)

    # Clean up the database (don't do this in case we want to inspect)
    # database_path.unlink()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """
    Create a test client for the FastAPI app.
    """
    tmp_path = tmp_path_factory.mktemp("mapcat")
    # Create a temporary SQLite database for testing.
    database_path = tmp_path / "test.db"

    run_migration(database_path)

    os.environ["mapcat_model_database_name"] = str(database_path)

    from mapcat.api import app

    yield TestClient(app)


@pytest.fixture(scope="session")
def mock_client_depth1(tmp_path_factory):
    """
    Create a test mock database client for mock test.
    """
    from mapcat.client import mock

    yield mock.DepthOneClient()


@pytest.fixture(scope="session")
def mock_client_processing(tmp_path_factory):
    """
    Create a test mock database client for mock test.
    """
    from mapcat.client import mock

    yield mock.ProcessingClient()


@pytest.fixture(scope="session")
def mock_client_pointing(tmp_path_factory):
    """
    Create a test mock pointing data base
    """
    from mapcat.client import mock

    yield mock.ProcessingClient()
