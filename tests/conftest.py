import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def run_migration(database_path: str):
    """
    Run the migration on the database.
    """
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("./sotodlib/mapcat/alembic.ini")
    database_url = f"sqlite:///{database_path}"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.set_main_option("script_location", "./sotodlib/mapcat/alembic/")
    command.upgrade(alembic_cfg, "heads")

    return


@pytest.fixture(scope="session", autouse=True)
def database_sesionmaker(tmp_path_factory):
    """
    Create a temporary SQLite database for testing.
    """

    tmp_path = tmp_path_factory.mktemp("mapcat")
    # Create a temporary SQLite database for testing.
    database_path = tmp_path / "test.db"

    # Run the migration on the database. This is blocking.
    run_migration(database_path)

    database_url = f"sqlite:///{database_path}"

    engine = create_engine(database_url, echo=True, future=True)

    yield sessionmaker(bind=engine, expire_on_commit=False)

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

    from sotodlib.mapcat.mapcat.api import app

    yield TestClient(app)


@pytest.fixture(scope="session")
def mock_client_depth1(tmp_path_factory):
    """
    Create a test mock database client for mock test.
    """
    from sotodlib.mapcat.mapcat.client import mock

    yield mock.DepthOneClient()


@pytest.fixture(scope="session")
def mock_client_processing(tmp_path_factory):
    """
    Create a test mock database client for mock test.
    """
    from sotodlib.mapcat.mapcat.client import mock

    yield mock.ProcessingClient()


@pytest.fixture(scope="session")
def mock_client_pointing(tmp_path_factory):
    """
    Create a test mock pointing data base
    """
    from sotodlib.mapcat.mapcat.client import mock

    yield mock.ProcessingClient()
