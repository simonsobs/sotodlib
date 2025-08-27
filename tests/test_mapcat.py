"""
Test the core functions
"""

import pytest

from sotodlib.mapcat.mapcat import core

from httpx import HTTPStatusError


def test_database_exists(database_sessionmaker):
    return


def test_create_depth_one(database_sessionmaker):
    # Create a depth one map
    with database_sessionmaker() as session:
        map_id = (
            core.create_depth_one(
                map_name="myDepthOne",
                map_path="/PATH/TO/DEPTH/ONE",
                tube_slot="OTi1",
                wafers="ws0,ws1,ws2",
                frequency="f090",
                ctime=1755787524.0,
                session=session,
            )
        ).id

    # Get depth one map back
    with database_sessionmaker() as session:
        dmap = core.get_depth_one(map_id, session=session)

    assert dmap.id == map_id
    assert dmap.map_name == "myDepthOne"
    assert dmap.map_path == "/PATH/TO/DEPTH/ONE"
    assert dmap.tube_slot == "OTi1"
    assert dmap.wafers == "ws0,ws1,ws2"
    assert dmap.frequency == "f090"
    assert dmap.ctime == 1755787524.0

    # Make child tables
    with database_sessionmaker() as session:
        proc_id = (
            core.create_processing_status(
                map_name="myDepthOne",
                processing_start=1756787524.0,
                processing_end=1756797524.0,
                processing_status="done",
                session=session,
            )
        ).id

        point_id = (
            core.create_pointing_residual(
                map_name="myDepthOne",
                ra_offset=1.2,
                dec_offset=-0.8,
                session=session,
            )
        ).id
        tod_id = (
            core.create_tod(
                map_name="myDepthOne",
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
                session=session,
            )
        ).id

    # Get child tables back
    with database_sessionmaker() as session:
        proc = core.get_processing_status(proc_id, session=session)
        point = core.get_pointing_residual(point_id, session=session)
        tod = core.get_tod(tod_id, session=session)

    assert proc.id == proc_id
    assert proc.map_name == "myDepthOne"
    assert proc.processing_start == 1756787524.0
    assert proc.processing_end == 1756797524.0
    assert proc.processing_status == "done"

    assert point.id == point_id
    assert point.map_name == "myDepthOne"
    assert point.ra_offset == 1.2
    assert point.dec_offset == -0.8

    assert tod.id == tod_id
    assert tod.map_name == "myDepthOne"
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

    # Update depth one map
    with database_sessionmaker() as session:
        dmap = core.update_depth_one(
            map_id=map_id,
            map_name="newDepthOne",
            map_path="/NEW/PATH/TO/DEPTH/ONE",
            tube_slot="OTi2",
            wafers="ws0,ws1",
            frequency="f150",
            ctime=1755787525.0,
            session=session,
        )

    assert dmap.map_name == "newDepthOne"
    assert dmap.map_path == "/NEW/PATH/TO/DEPTH/ONE"
    assert dmap.tube_slot == "OTi2"
    assert dmap.wafers == "ws0,ws1"
    assert dmap.frequency == "f150"
    assert dmap.ctime == 1755787525.0

    # Check updating depth one automatically updates child tables
    with database_sessionmaker() as session:
        proc = core.get_pointing_residual(proc_id, session=session)
        point = core.get_pointing_residual(point_id, session=session)
        tod = core.get_tod(tod_id, session=session)

    assert proc.map_name == "newDepthOne"
    assert point.map_name == "newDepthOne"
    assert tod.map_name == "newDepthOne"

    # Update proc status and point resid
    with database_sessionmaker() as session:
        proc = core.update_processing_status(
            proc_id=proc_id,
            map_name=None,
            processing_start=1756887524.0,
            processing_end=1756799524.0,
            processing_status="in progress",
            session=session,
        )

        point = core.update_pointing_residual(
            point_id=point_id,
            map_name=None,
            ra_offset=1.4,
            dec_offset=-0.75,
            session=session,
        )
        tod = core.update_tod(
            tod_id=tod_id,
            map_name=None,
            obs_id="obs_1753486324_lati6_110",
            pwv=0.4,
            ctime=1753486324.0,
            start_time=1753536324.0,
            stop_time=1753581324.0,
            nsamples=28542,
            telescope="sat",
            telescope_flavor="sat",
            tube_slot=None,
            tube_flavor="lf",
            frequency="030",
            scan_type="cal",
            subtype="mars",
            wafer_count=2,
            duration=200000,
            az_center=90.0,
            az_throw=120.0,
            el_center=60.0,
            el_throw=5.0,
            roll_center=25.0,
            roll_throw=5.0,
            wafer_slots_list="ws0,ws1",
            stream_ids_list="ufm_mv25,ufm_mv26",
            session=session,
        )

    assert proc.id == proc_id
    assert proc.map_name == "newDepthOne"
    assert proc.processing_start == 1756887524.0
    assert proc.processing_end == 1756799524.0
    assert proc.processing_status == "in progress"

    assert point.id == point_id
    assert point.map_name == "newDepthOne"
    assert point.ra_offset == 1.4
    assert point.dec_offset == -0.75

    assert tod.map_name == "newDepthOne"
    assert tod.pwv == 0.4
    assert tod.obs_id == "obs_1753486324_lati6_110"
    assert tod.ctime == 1753486324.0
    assert tod.start_time == 1753536324.0
    assert tod.stop_time == 1753581324.0
    assert tod.nsamples == 28542
    assert tod.telescope == "sat"
    assert tod.telescope_flavor == "sat"
    assert tod.tube_slot == "i6"
    assert tod.tube_flavor == "lf"
    assert tod.frequency == "030"
    assert tod.scan_type == "cal"
    assert tod.subtype == "mars"
    assert tod.wafer_count == 2
    assert tod.duration == 200000
    assert tod.az_center == 90.0
    assert tod.az_throw == 120.0
    assert tod.el_center == 60.0
    assert tod.el_throw == 5.0
    assert tod.roll_center == 25.0
    assert tod.roll_throw == 5.0
    assert tod.wafer_slots_list == "ws0,ws1"
    assert tod.stream_ids_list == "ufm_mv25,ufm_mv26"

    # Check bad map ID raises ValueError
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.get_depth_one(999999, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.update_depth_one(
                999999,
                map_name="IGNORE",
                map_path="IGNORE",
                tube_slot="IGNORE",
                wafers="IGNORE",
                frequency="IGNORE",
                ctime=0,
                session=session,
            )

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_depth_one(999999, session=session)

    # Check bad proc ID raises ValueError
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.get_processing_status(999999, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.update_processing_status(
                999999,
                map_name="IGNORE",
                processing_start=0,
                processing_end=0,
                processing_status="IGNORE",
                session=session,
            )

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_processing_status(999999, session=session)

    # Check bad point ID raises ValueError
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.get_pointing_residual(999999, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.update_pointing_residual(
                999999,
                map_name="IGNORE",
                ra_offset=0.0,
                dec_offset=0.0,
                session=session,
            )

    # Check bad tod ID raises ValueError
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.get_tod(999999, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.update_tod(
                999999,
                map_name="INGORE",
                obs_id="INGORE",
                pwv=0.0,
                ctime=0.0,
                start_time=0.0,
                stop_time=0.0,
                nsamples=0.0,
                telescope="INGORE",
                telescope_flavor="INGORE",
                tube_slot="INGORE",
                tube_flavor="INGORE",
                frequency="INGORE",
                scan_type="INGORE",
                subtype="INGORE",
                wafer_count=0.0,
                duration=0.0,
                az_center=0.0,
                az_throw=0.0,
                el_center=0.0,
                el_throw=0.0,
                roll_center=0.0,
                roll_throw=0.0,
                wafer_slots_list="INGORE",
                stream_ids_list="INGORE",
                session=session,
            )

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_tod(999999, session=session)

    # Delete depth 1
    with database_sessionmaker() as session:
        core.delete_depth_one(map_id, session=session)

    # Check that proc stat and point resid were cascade deleted
    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_processing_status(proc_id, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_pointing_residual(point_id, session=session)

    with pytest.raises(ValueError):
        with database_sessionmaker() as session:
            core.delete_tod(tod_id, session=session)


def test_add_remove_child_tables(database_sessionmaker):
    # Create a depth one map
    with database_sessionmaker() as session:
        map_id = (
            core.create_depth_one(
                map_name="myDepthOne",
                map_path="/PATH/TO/DEPTH/ONE",
                tube_slot="OTi1",
                wafers="ws0,ws1,ws2",
                frequency="f090",
                ctime=1755787524.0,
                session=session,
            )
        ).id

    # Make proc stat and point resid tables, child to our depth one map
    with database_sessionmaker() as session:
        proc_id = (
            core.create_processing_status(
                map_name="myDepthOne",
                processing_start=1756787524.0,
                processing_end=1756797524.0,
                processing_status="done",
                session=session,
            )
        ).id

        point_id = (
            core.create_pointing_residual(
                map_name="myDepthOne",
                ra_offset=1.2,
                dec_offset=-0.8,
                session=session,
            )
        ).id

        tod_id = (
            core.create_tod(
                map_name="myDepthOne",
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
                session=session,
            )
        ).id
    with database_sessionmaker() as session:
        core.delete_processing_status(proc_id, session=session)
        core.delete_pointing_residual(point_id, session=session)
        core.delete_tod(tod_id, session=session)

        core.delete_depth_one(map_id, session=session)


"""
def test_depth_one_map(client):
    # Add depth 1 map
    response = client.put(
        "api/v1/depthone/new",
        json={
            "map_name": "myDepthOne",
            "map_path": "/PATH/TO/DEPTH/ONE",
            "tube_slot": "OTi1",
            "wafers": "ws0,ws1,ws2",
            "frequency": "f090",
            "ctime": 1755787524.0,
        },
    )

    map_id = response.json()["id"]
    response.raise_for_status()

    # Get depth 1 map
    response = client.get("api/v1/depthone/{}".format(map_id))

    response.raise_for_status()
    assert response.json()["map_name"] == "myDepthOne"
    assert response.json()["map_path"] == "/PATH/TO/DEPTH/ONE"
    assert response.json()["tube_slot"] == "OTi1"
    assert response.json()["wafers"] == "ws0,ws1,ws2"
    assert response.json()["frequency"] == "f090"
    assert response.json()["ctime"] == 1755787524.0

    # Update depth 1 map
    response = client.post(
        "api/v1/depthone/{}".format(map_id),
        json={
            "map_name": "newDepthOne",
            "map_path": "/NEW/PATH/TO/DEPTH/ONE",
            "tube_slot": "OTi2",
            "wafers": "ws0,ws1",
            "frequency": "f150",
            "ctime": 1755787525.0,
        },
    )

    response.raise_for_status()
    assert response.json()["map_name"] == "newDepthOne"
    assert response.json()["map_path"] == "/NEW/PATH/TO/DEPTH/ONE"
    assert response.json()["tube_slot"] == "OTi2"
    assert response.json()["wafers"] == "ws0,ws1"
    assert response.json()["frequency"] == "f150"
    assert response.json()["ctime"] == 1755787525.0

    # Add proccessing status
    response = client.put(
        "api/v1/procstat/new",
        json={
            "map_name": "newDepthOne",
            "processing_start": 1755787854.0,
            "processing_end": 1755787954.0,
            "processing_status": "done",
        },
    )

    proc_id = response.json()["id"]
    response.raise_for_status()

    # Get processing status
    response = client.get("api/v1/procstat/{}".format(proc_id))

    response.raise_for_status()
    assert response.json()["map_name"] == "newDepthOne"
    assert response.json()["processing_start"] == 1755787854.0
    assert response.json()["processing_end"] == 1755787954.0
    assert response.json()["processing_status"] == "done"

    # Update processing_status
    response = client.post(
        "api/v1/procstat/{}".format(proc_id),
        json={
            "map_name": "newDepthOne",
            "processing_start": 175573454.0,
            "processing_end": 1755997954.0,
            "processing_status": "failed",
        },
    )

    response.raise_for_status()
    assert response.json()["map_name"] == "newDepthOne"
    assert response.json()["processing_start"] == 175573454.0
    assert response.json()["processing_end"] == 1755997954.0
    assert response.json()["processing_status"] == "failed"

    # Delete processing status
    response = client.delete("api/v1/procstat/{}".format(proc_id))

    response.raise_for_status()

    # Add pointing residual
    response = client.put(
        "api/v1/pointresid/new",
        json={
            "map_name": "newDepthOne",
            "ra_offset": 1.2,
            "dec_offset": -0.8,
        },
    )

    point_id = response.json()["id"]
    response.raise_for_status()

    # Get processing status
    response = client.get("api/v1/pointresid/{}".format(point_id))

    response.raise_for_status()
    assert response.json()["map_name"] == "newDepthOne"
    assert response.json()["ra_offset"] == 1.2
    assert response.json()["dec_offset"] == -0.8

    # Update processing_status
    response = client.post(
        "api/v1/pointresid/{}".format(point_id),
        json={
            "map_name": "newDepthOne",
            "ra_offset": 0.1,
            "dec_offset": 1.8,
        },
    )

    response.raise_for_status()
    assert response.json()["map_name"] == "newDepthOne"
    assert response.json()["ra_offset"] == 0.1
    assert response.json()["dec_offset"] == 1.8

    # Delete processing status
    response = client.delete("api/v1/pointresid/{}".format(point_id))

    response.raise_for_status()

    # Delete depth 1 map
    response = client.delete("api/v1/depthone/{}".format(map_id))

    response.raise_for_status()

    response = client.get("api/v1/depthone/{}".format(map_id))

    assert (
        response.status_code == 404
    )  # ID should be deleted, make sure we don't find it again
"""
