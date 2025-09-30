import random
from datetime import datetime, timedelta

from sotodlib.io.mapcat.database import (
    DepthOneMapTable,
    TODDepthOneTable,
    DepthOneCoaddTable
)
from sotodlib.io.mapcat.helper import settings
from sqlmodel import Session, SQLModel


def main():
    engine = settings.engine
    # Create tables if they don't exist
    SQLModel.metadata.create_all(engine)

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    maps: list[DepthOneMapTable] = []

    # Start TODs around midnight yesterday UTC
    start_time = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0)

    for map_idx in range(2):
        # Map metadata
        map_name = f"depth1_map_{map_idx+1}"
        map_ctime = start_time.timestamp() + (map_idx * 6 * 3600)
        map_start = map_ctime
        map_stop = map_ctime + 3 * 3600  # ~ span of TODs

        dmap = DepthOneMapTable(
            map_id=map_idx + 1,
            map_name=map_name,
            map_path=f"/maps/{map_name}.fits",
            tube_slot=f"T{map_idx+1}",
            frequency="150GHz",
            ctime=map_ctime,
            start_time=map_start,
            stop_time=map_stop,
        )

        # TODs for this map
        tod_start = start_time + timedelta(hours=map_idx * 6)
        for tod_idx in range(3):
            duration_hours = random.randint(1, 2)
            tod_start_ts = tod_start.timestamp()
            tod_stop_ts = (tod_start + timedelta(hours=duration_hours)).timestamp()

            tod = TODDepthOneTable(
                tod_id=(map_idx * 3) + tod_idx + 1,
                obs_id=f"OBS_{map_idx+1}_{tod_idx+1}",
                pwv=0.5 + 0.1 * tod_idx,
                ctime=(tod_start_ts + tod_stop_ts) / 2,
                start_time=tod_start_ts,
                stop_time=tod_stop_ts,
                nsamples=duration_hours * 3600 * 100,  # pretend 100Hz sampling
                telescope="SO_LAT",
                telescope_flavor="LF",
                tube_slot=f"T{map_idx+1}",
                tube_flavor="LF",
                frequency="150GHz",
                scan_type="science",
                subtype="deep",
                wafer_count=10,
                duration=tod_stop_ts - tod_start_ts,
                az_center=180.0,
                az_throw=5.0,
                el_center=50.0,
                el_throw=2.0,
                roll_center=0.0,
                roll_throw=0.0,
                wafer_slots_list="w1,w2,w3",
                stream_ids_list="s1,s2",
            )

            # Just link by appending
            dmap.tods.append(tod)

            tod_start += timedelta(hours=duration_hours)

        maps.append(dmap)

    # Create a coadd
    coadd = DepthOneCoaddTable(
        coadd_name="deep_coadd_1",
        coadd_type="two",
        coadd_path="/coadds/deep_coadd_1.fits",
        frequency="150GHz",
        ctime=sum(x.ctime for x in maps) * 0.5,
        start_time=min(x.start_time for x in maps),
        stop_time=max(x.stop_time for x in maps),
        maps=maps
    )

    # Commit all to DB
    with Session(engine) as session:
        session.add(coadd)
        session.commit()


if __name__ == "__main__":
    main()
