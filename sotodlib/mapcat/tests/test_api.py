import pytest
from httpx import HTTPStatusError


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
