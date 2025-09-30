"""
Test the core functions
"""

import pytest

from sotodlib.io.mapcat.database import (
    DepthOneMapTable,
    ProcessingStatusTable,
    TODDepthOneTable,
    PointingResidualTable,
    PipelineInformationTable,
    SkyCoverageTable,
)


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def run_migration(database_path: str):
    """
    Run the migration on the database.
    """
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("./sotodlib/io/mapcat/alembic.ini")
    database_url = f"sqlite:///{database_path}"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    alembic_cfg.set_main_option("script_location", "./sotodlib/io/mapcat/alembic/")
    command.upgrade(alembic_cfg, "heads")

    return


@pytest.fixture(scope="session", autouse=True)
def database_sessionmaker(tmp_path_factory):
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
    database_path.unlink()



def test_database_exists(database_sessionmaker):
    return


def test_create_depth_one(database_sessionmaker):
    # Create a depth one map
    with database_sessionmaker() as session:
        data = DepthOneMapTable(
            map_name="myDepthOne",
            map_path="/PATH/TO/DEPTH/ONE",
            tube_slot="OTi1",
            frequency="f090",
            ctime=1755787524.0,
            start_time=1755687524.0,
            stop_time=1755887524.0,
        )

        session.add(data)
        session.commit()
        session.refresh(data)

        map_id = data.map_id

    # Get depth one map back
    with database_sessionmaker() as session:
        dmap = session.get(DepthOneMapTable, map_id)

    assert dmap.map_id == map_id
    assert dmap.map_name == "myDepthOne"
    assert dmap.map_path == "/PATH/TO/DEPTH/ONE"
    assert dmap.tube_slot == "OTi1"
    assert dmap.frequency == "f090"
    assert dmap.ctime == 1755787524.0

    # Make child tables
    with database_sessionmaker() as session:
        processing_status = ProcessingStatusTable(
            processing_start=1756787524.0,
            processing_end=1756797524.0,
            processing_status="done",
            map=dmap
        )

        pointing_residual = PointingResidualTable(
            ra_offset=1.2,
            dec_offset=-0.8,
            map=dmap
        )

        tod = TODDepthOneTable(
            obs_id="obs_1753486724_lati6_111",
            pwv=0.7,
            ctime=1755787524.0,
            start_time=1755687524.0,
            stop_time=1755887524.0,
            nsamples=28562,
            telescope="lat",
            telescope_flavor="lat",
            tube_slot="i6",
            tube_flavor="mf",
            frequency="150",
            scan_type="ops",
            subtype="cmb",
            wafer_count=3,
            duration=100000,
            az_center=180.0,
            az_throw=90.0,
            el_center=50.0,
            el_throw=0.0,
            roll_center=0.0,
            roll_throw=0.0,
            wafer_slots_list="ws0,ws1,ws2",
            stream_ids_list="ufm_mv25,ufm_mv26,ufm_mv11",
            maps=[dmap]
        )

        pipeline_info = PipelineInformationTable(
            sotodlib_version="1.2.3",
            map_maker="minkasi",
            preprocess_info={"config": "test"},
            map=dmap
        )

        sky_coverage = SkyCoverageTable(
            map=dmap,
            x=5,
            y=2,
        )

        session.add_all(
            [processing_status, pointing_residual, tod, pipeline_info, sky_coverage]
        )
        session.commit()

        proc_id = processing_status.processing_status_id
        point_id = pointing_residual.pointing_residual_id
        tod_id = tod.tod_id
        pipe_id = pipeline_info.pipeline_information_id
        sky_id = sky_coverage.patch_id

    # Get child tables back
    with database_sessionmaker() as session:
        proc = session.get(ProcessingStatusTable, proc_id)
        point = session.get(PointingResidualTable, point_id)
        tod = session.get(TODDepthOneTable, tod_id)
        pipe = session.get(PipelineInformationTable, pipe_id)
        sky = session.get(SkyCoverageTable, sky_id)

    assert proc.processing_status_id == proc_id
    assert proc.map_id == map_id
    assert proc.processing_start == 1756787524.0
    assert proc.processing_end == 1756797524.0
    assert proc.processing_status == "done"

    assert point.pointing_residual_id == point_id
    assert point.map_id == map_id
    assert point.ra_offset == 1.2
    assert point.dec_offset == -0.8

    assert tod.tod_id == tod_id
    assert tod.pwv == 0.7
    assert tod.obs_id == "obs_1753486724_lati6_111"
    assert tod.ctime == 1755787524.0
    assert tod.start_time == 1755687524.0
    assert tod.stop_time == 1755887524.0
    assert tod.nsamples == 28562
    assert tod.telescope == "lat"
    assert tod.telescope_flavor == "lat"
    assert tod.tube_slot == "i6"
    assert tod.tube_flavor == "mf"
    assert tod.frequency == "150"
    assert tod.scan_type == "ops"
    assert tod.subtype == "cmb"
    assert tod.wafer_count == 3
    assert tod.duration == 100000
    assert tod.az_center == 180.0
    assert tod.az_throw == 90.0
    assert tod.el_center == 50.0
    assert tod.el_throw == 0.0
    assert tod.roll_center == 0.0
    assert tod.roll_throw == 0.0
    assert tod.wafer_slots_list == "ws0,ws1,ws2"
    assert tod.stream_ids_list == "ufm_mv25,ufm_mv26,ufm_mv11"

    assert pipe.pipeline_information_id == pipe_id
    assert pipe.map_id == map_id
    assert pipe.sotodlib_version == "1.2.3"
    assert pipe.map_maker == "minkasi"
    assert pipe.preprocess_info == {"config": "test"}

    assert sky.patch_id == sky_id
    assert sky.x == '5'
    assert sky.y == '2'

    # Check bad map ID raises ValueError
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            result = session.get(DepthOneMapTable, 999999)
            if result is None:
                raise ValueError("Map ID does not exist")


def test_add_remove_child_tables(database_sessionmaker):
    # Create a depth one map
    with database_sessionmaker() as session:
        dmap = DepthOneMapTable(
            map_name="myDepthOne2",
            map_path="/PATH/TO/DEPTH/ONE",
            tube_slot="OTi1",
            frequency="f090",
            ctime=1755787524.0,
            start_time=1755687524.0,
            stop_time=1755887524.0,
        )

        processing_status = ProcessingStatusTable(
            processing_start=1756787524.0,
            processing_end=1756797524.0,
            processing_status="done",
            map=dmap
        )

        pointing_residual = PointingResidualTable(
            ra_offset=1.2,
            dec_offset=-0.8,
            map=dmap
        )

        tod = TODDepthOneTable(
        obs_id="obs_1753486724_lati6_111",
            pwv=0.7,
            ctime=1755787524.0,
            start_time=1755687524.0,
            stop_time=1755887524.0,
            nsamples=28562,
            telescope="lat",
            telescope_flavor="lat",
            tube_slot="i6",
            tube_flavor="mf",
            frequency="150",
            scan_type="ops",
            subtype="cmb",
            wafer_count=3,
            duration=100000,
            az_center=180.0,
            az_throw=90.0,
            el_center=50.0,
            el_throw=0.0,
            roll_center=0.0,
            roll_throw=0.0,
            wafer_slots_list="ws0,ws1,ws2",
            stream_ids_list="ufm_mv25,ufm_mv26,ufm_mv11",
            maps=[dmap]
        )

        pipeline_info = PipelineInformationTable(
            sotodlib_version="1.2.3",
            map_maker="minkasi",
            preprocess_info={"config": "test"},
            map=dmap
        )

        sky_coverage = SkyCoverageTable(
            map=dmap,
            x=5,
            y=2,
        )

        session.add_all(
            [dmap, processing_status, pointing_residual, tod, pipeline_info, sky_coverage]
        )
        session.commit()
        
        dmap_id = dmap.map_id
        proc_id = processing_status.processing_status_id
        point_id = pointing_residual.pointing_residual_id
        tod_id = tod.tod_id
        pipe_id = pipeline_info.pipeline_information_id
        sky_id = sky_coverage.patch_id

    # Check the cascades work
    with database_sessionmaker() as session:
        x = session.get(DepthOneMapTable, dmap_id)
        session.delete(x)
        session.commit()

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            x = session.get(ProcessingStatusTable, proc_id)
            if x is None:
                raise ValueError("Not found")

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            x = session.get(PointingResidualTable, point_id)
            if x is None:
                raise ValueError("Not found")

    with database_sessionmaker() as session:
        tod = session.get(TODDepthOneTable, tod_id)
        assert tod is not None
    
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            x = session.get(PipelineInformationTable, pipe_id)
            if x is None:
                raise ValueError("Not found")

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            x = session.get(SkyCoverageTable, sky_id)
            if x is None:
                raise ValueError("Not found")
