"""
Test the core functions
"""

import pytest

from sotodlib.mapcat.mapcat import core

from httpx import HTTPStatusError


def test_database_exists(database_sesionmaker):
    return


def test_create_depth_one(database_sesionmaker):
    with database_sesionmaker() as session:
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

    with database_sesionmaker() as session:
        dmap = core.get_depth_one(map_id, session=session)

    assert dmap.id == map_id
    assert dmap.map_name == "myDepthOne"
    assert dmap.map_path == "/PATH/TO/DEPTH/ONE"
    assert dmap.tube_slot == "OTi1"
    assert dmap.wafers == "ws0,ws1,ws2"
    assert dmap.frequency == "f090"
    assert dmap.ctime == 1755787524.0

    with database_sesionmaker() as session:
        proc_id = (
            core.create_proccessing_status(
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

    with database_sesionmaker() as session:
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

    # Check updating depth one automatically updates chile tables
    with database_sesionmaker() as session:
        proc = core.get_pointing_residual(proc_id, session=session)
        point = core.get_pointing_residual(point_id, session=session)

    assert proc.map_name == "newDepthOne"
    assert point.map_name == "newDepthOne"

    # Check getting an uregistered ID raises ValueError
    with pytest.raises(ValueError):
        with database_sesionmaker() as session:
            core.get_depth_one(999999, session=session)

    with pytest.raises(ValueError):
        with database_sesionmaker() as session:
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

    with database_sesionmaker() as session:
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
