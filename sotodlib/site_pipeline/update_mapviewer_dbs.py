"""update_mapviewer_dbs.py

This module maintains databases for mapviewer instances that show atomic/depth-1 maps
per instrument. 

"""

import os
import subprocess
import sqlite3
import json
import datetime
import tempfile
from pathlib import Path
import argparse
from sotodlib.site_pipeline.utils.logging import init_logger
from sotodlib.site_pipeline.utils.pipeline import main_launcher

logger = init_logger('update_mapviewer_dbs', 'update_mapviewer_dbs: ')

# ------------------------------------------------------------------------------
# BEGIN: Logic used to create the config files for tilemaker-db to consume
# ------------------------------------------------------------------------------

def create_map_group(data):
  """
  Create a map group, the highest level of hierarchy for the mapviewer's layers structure.
  """
  map_group = {}
  map_group["name"] = data["obs_id"]
  map_group["description"] = "Maps from " + data["telescope"] + " at " + str(datetime.datetime.fromtimestamp(int(data["ctime"])))
  map_group["maps"] = []
  return map_group

def create_map(data):
  """
  Create a map to be associated with a map group.
  """
  map = {}
  map["map_id"] = data["obs_id"] + "-" + data["wafer"]
  map["name"] = data["wafer"]
  map["description"] = "Maps from wafer " + data["wafer"]
  map["bands"] = []
  return map

def create_band(data):
  """
  Create a band to be associated with a map.
  """
  band = {}
  band["band_id"] = "band-" + data["obs_id"] + "-" + data["wafer"] + "-" + data["freq_channel"]
  band["name"] = data["freq_channel"]
  band["description"] = "Frequency band " + data["freq_channel"]
  band["layers"] = create_layers(data)
  return band

def create_layer(
    layer_id,
    name,
    quantity,
    units,
    provider,
    description="",
    cmap="RdBu_r",
  ):
  """
  Create a layer to be associated with a band.
  """
  layer = {}
  layer["layer_id"] = layer_id
  layer["name"] = name
  layer["description"] = description
  layer["provider"] = provider
  layer["quantity"] = quantity
  layer["units"] = units
  layer["vmin"] = "auto"
  layer["vmax"] = "auto"
  layer["cmap"] = cmap
  return layer

def create_layers(data):
  """
  Create layers from the _hits.fits, _wmap.fits, _weights.fits files associated with a particular
  database row's prefix_path, setting reasonable default attributes depending on the type of
  fits file.
  """
  layers = []
  # db seems to have an extraneous forward slash, so let's just clean it up
  normalized_prefix_path = os.path.normpath(data["prefix_path"])

  for file_ext in ["_hits.fits", "_wmap.fits", "_weights.fits"]:
    if (file_ext == "_hits.fits"):
      hits_layer = create_layer(
        layer_id="layer-" + data["obs_id"] + "-" + data["wafer"] + "-" + data["freq_channel"] + "-hits",
        name="hits",
        quantity="hits",
        units="hit",
        cmap="viridis",
        provider={
          "provider_type": "fits",
          "filename": f"{normalized_prefix_path}{file_ext}",
        }
      )
      layers.append(hits_layer)
    elif (file_ext == "_wmap.fits"):
      wmap_types = ["I", "Q", "U"]
      for idx in range(len(wmap_types)):
        coadd_layer = create_layer(
          layer_id=f"layer-{data['obs_id']}-{data['wafer']}-{data['freq_channel']}-coadd-{idx}",
          name=wmap_types[idx],
          quantity=f"T ({wmap_types[idx]})",
          units="uK",
          provider={
            "provider_type": "fits_combination",
            "providers": [
              {
                "provider_type": "fits",
                "filename": f"{normalized_prefix_path}{file_ext}",
                "hdu": 0,
                "index": idx,
              },
              {
                "provider_type": "fits",
                "filename": f"{normalized_prefix_path}_weights.fits",
                "hdu": 0,
                "index": idx,
              },
            ],
            "function": "/"
          }
        )
        layers.append(coadd_layer)
    else:
      wmap_types = ["II", "QQ", "UU"]
      for idx in range(len(wmap_types)):
        wmap_layer = create_layer(
          layer_id="layer-" + data["obs_id"] + "-" + data["wafer"] + "-" + data["freq_channel"] + "-" + wmap_types[idx],
          name=wmap_types[idx],
          quantity=f"T ({wmap_types[idx]})",
          units="uK",
          provider={
            "provider_type": "fits",
            "filename": f"{normalized_prefix_path}{file_ext}",
            "hdu": 0,
            "index": idx,
          }
        )
        layers.append(wmap_layer)
  return layers

class AtomicDBQueryError(RuntimeError):
    pass

def query_database(db_path: Path, cutoff_days: int):
  """
    Get the rows of data in an instrument's database whose 'ctime' attribute
    is >= (now - cutoff_days)
  """
  cutoff_seconds = cutoff_days * 86400

  sql = """
      SELECT *
      FROM atomic
      WHERE ctime >= (strftime('%s','now') - ?)
        AND valid = 1
      ORDER BY ctime DESC;
  """

  try:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql, (cutoff_seconds,))
    rows = cursor.fetchall()
    return [dict(row) for row in rows]
  
  except sqlite3.Error as exc:
    raise AtomicDBQueryError(
        f"Query to {db_path} failed: {exc}"
    ) from exc
  
  finally:
    if conn is not None:
      conn.close()


def generate_sat_map_groups(instrument_name: str, instrument_db_path: Path, cutoff_days: int):
  """
  Generate map groups specific to SAT databases
  """
  try:
    query_results = query_database(instrument_db_path, cutoff_days)
  except AtomicDBQueryError as exc:
    logger.error(str(exc))

  # If query_results is empty, give some useful feedback and early return
  if not query_results:
    logger.info("No output from query to " + instrument_name + ". You may wish to increase the cutoff_days parameter.")
    return None

  map_groups = {}
  rows = []
  for row_data in query_results:
    rows.append(row_data)
    map_group = map_groups.get(row_data["ctime"], None)

    if (map_group is None):
      map_groups[row_data["ctime"]] = create_map_group(row_data)
    
    existing_map = None
    for map_obj in map_groups[row_data["ctime"]]["maps"]:
      if (map_obj["name"] == row_data["wafer"]):
        existing_map = map_obj
        break

    if (existing_map is None):
      map_groups[row_data["ctime"]]["maps"].append(create_map(row_data))

    for map_obj in map_groups[row_data["ctime"]]["maps"]:
      if (map_obj["name"] == row_data["wafer"]):
        map_obj["bands"].append(create_band(row_data))
        break

  return {"map_groups": list(map_groups.values())}

# ------------------------------------------------------------------------------
# END: Logic used to create the config files for tilemaker-db to consume
# ------------------------------------------------------------------------------


def get_instrument_name(instruments_db_path: Path):
    """
    Extract instrument name from path like:
    /so/site-pipeline/<instrument>/...
    """
    parts = instruments_db_path.parts
    try:
       idx = parts.index("site-pipeline")
       return parts[idx + 1]
    except:
       return None


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Generate and populate tilemaker databases for SAT instruments"
        )
    parser.add_argument(
        "--mapviewer-dbs-root-path",
        type=str,
        default="/so/services/mapviewer",
        help="Path to root directory where mapviewer-compatible DBs are stored",
    )
    parser.add_argument(
        "--instrument-db-paths",
        nargs="+",
        type=str,
        help="Paths to an instrument's atomic/depth-1 SQLite DB file",
    )
    parser.add_argument(
        "--cutoff-days",
        type=int,
        default=1,
        help="Delete map groups older than this many days and repopulate with maps whose 'ctime' is greater than or equal to (now - cutoff_days)",
    )
    return parser


def main(mapviewer_dbs_root_path: str, instrument_db_paths: list[str], cutoff_days: int):
    """
    Create or update databases for mapviewer instances of each instrument, as available

    Arguments
    ----------
    mapviewer_dbs_root_path : str
        Path to root directory where mapviewer-compatible DBs are stored
    instrument_db_paths : list[str]
        Paths to an instrument's atomic/depth-1 SQLite DB file
    cutoff_days: int
        Delete map groups older than this many days and repopulate with maps whose 'ctime'
        is greater than or equal to (now - cutoff_days)
    """
    if instrument_db_paths is None:
       logger.info("Missing instrument-db-paths required to update mapviewer dbs. Exiting update.")
       return
    
    logger.info("Updating mapviewer dbs")

    # Convert string paths to Path objects
    root = Path(mapviewer_dbs_root_path)
    instrument_db_path_objects = [Path(p) for p in instrument_db_paths]

    # ------------------------------------------------------------------------------
    # BEGIN: Update databases for instruments
    # ------------------------------------------------------------------------------
    for instrument_db_path in instrument_db_path_objects:
        # Try to determine the instrument name and, if not recognized, skip
        instrument_name = get_instrument_name(instrument_db_path)
        
        if (instrument_name is None):
           logger.warning("Instrument name not recognized from the instrument_db_path '" + instrument_db_path + "'. Exiting update for this path.")
           continue

        logger.info("Updating mapviewer db for " + instrument_name)
        instrument_mapviewer_dir = root / instrument_name
        instrument_mapviewer_dir.mkdir(parents=True, exist_ok=True)

        instrument_mapviewer_db_path = instrument_mapviewer_dir / f"{instrument_name}.db"

        # ------------------------------------------------------------------------------
        # a. If db exists, delete any map groups that are older than a day
        # ------------------------------------------------------------------------------
        if instrument_mapviewer_db_path.exists():
            conn = sqlite3.connect(instrument_mapviewer_db_path)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("""
                    DELETE FROM map_groups
                    WHERE id IN (
                        SELECT id
                        FROM map_groups
                        WHERE name LIKE 'obs_%'
                        AND CAST(substr(name, instr(name, 'obs_') + 4, 10) AS INTEGER)
                            < strftime('%s', 'now', '-1 days')
                    );
                """)
                conn.commit()
            finally:
                conn.close()

        # ------------------------------------------------------------------------------
        # b. Use tilemaker-db to generate db entries using the generated map groups
        # ------------------------------------------------------------------------------

        # Tell tilemaker to use a database for its configurations
        os.environ.update({
            "TILEMAKER_CONFIG_PATH": f"sqlite:///{instrument_mapviewer_db_path}",
        })

        map_groups = generate_sat_map_groups(instrument_name, instrument_db_path, cutoff_days)

        # Skip tilemaker-db call if map_groups is null
        if map_groups is None:
           continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(
                map_groups,
                f,
                indent=2
            )
            f.flush()
            subprocess.run(["tilemaker-db", "populate", f.name])


if __name__ == "__main__":
    main_launcher(main, get_parser)
