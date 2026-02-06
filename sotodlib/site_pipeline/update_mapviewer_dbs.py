import os
import subprocess
import sqlite3
import csv
import json
import datetime
import tempfile
from pathlib import Path
import argparse
from sotodlib.site_pipeline.utils.logging import init_logger

logger = init_logger('update_mapviewer_dbs', 'update_mapviewer_dbs: ')

# ------------------------------------------------------------------------------
# BEGIN: Logic used to create the config files for tilemaker-db to consume
# ------------------------------------------------------------------------------

def create_map_group(data):
  map_group = {}
  map_group["name"] = data["obs_id"]
  map_group["description"] = "Maps from " + data["telescope"] + " at " + str(datetime.datetime.fromtimestamp(int(data["ctime"])))
  map_group["maps"] = []
  return map_group

def create_map(data):
  map = {}
  map["map_id"] = data["obs_id"] + "-" + data["wafer"]
  map["name"] = data["wafer"]
  map["description"] = "Maps from wafer " + data["wafer"]
  map["bands"] = []
  return map

def create_band(data):
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
          layer_id=f"layer-{data["obs_id"]}-{data["wafer"]}-{data["freq_channel"]}-coadd-{idx}",
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

def generate_sat_map_groups(sat_instrument, cutoff_days):
  REMOTE_DB_PATH = f"/so/site-pipeline/{sat_instrument}/maps/{sat_instrument}_atomic_250423m/atomic_db.sqlite"

  # Define the SQL query with parameterized NUMBER_OF_DAYS
  sql = (
    "SELECT * "
    "FROM atomic "
    f"WHERE ctime >= (strftime('%s','now') - ({cutoff_days}*86400)) AND valid=1 "
    "ORDER BY ctime DESC;"
  )

  # Build SQL command
  sqlite_cmd = [
    "sqlite3",
    "-header",
    "-csv",
    REMOTE_DB_PATH,
    sql
  ]

  try:
    result = subprocess.run(
      sqlite_cmd,
      text=True,
      capture_output=True,
      timeout=30  # Add a timeout in case the query hangs
    )
  except subprocess.TimeoutExpired:
    logger.error("Query to " + sat_instrument + " db timed out. Exiting.")
    exit(1)

  if result.returncode != 0:
    logger.error("Query to " + sat_instrument + " db failed with exit code {result.returncode}")

  # If stdout is empty, give some useful feedback
  if not result.stdout.strip():
    logger.info("No output from query to " + sat_instrument + ". You may wish to increase the cutoff_days parameter.")

  # Parse CSV output
  query_results = csv.DictReader(result.stdout.splitlines())
  map_groups = {}
  rows = []
  for row_data in query_results:
    rows.append(row_data)
    map_group = map_groups.get(row_data["ctime"], None)

    if (map_group is None):
      map_groups[row_data["ctime"]] = create_map_group(row_data)
    
    maps = map_groups[row_data["ctime"]]["maps"]

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


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Generate and populate tilemaker databases for SAT instruments"
        )
    parser.add_argument(
        "--root",
        default="/so/services/mapviewer",
        help="Root directory where instrument DBs are stored",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["satp1", "satp3"],
        help="SAT instruments to process",
    )
    parser.add_argument(
        "--cutoff-days",
        type=int,
        default=1,
        help="Delete map groups older than this many days",
    )
    return parser


def main(root, instruments, cutoff_days):
    logger.info("Updating mapviewer dbs")
    ROOT = Path(root)
    # ------------------------------------------------------------------------------
    # BEGIN: Update databases for SAT instruments
    # ------------------------------------------------------------------------------
    for instrument in instruments:
        logger.info("Updating mapviewer db for " + instrument)
        instrument_dir = ROOT / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        DB_PATH = instrument_dir / f"{instrument}.db"

        # ------------------------------------------------------------------------------
        # a. If db exists, delete any map groups that are older than a day
        # ------------------------------------------------------------------------------
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
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
            "TILEMAKER_CONFIG_PATH": f"sqlite:///{DB_PATH}",
        })

        map_groups = generate_sat_map_groups(instrument, cutoff_days)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(
                map_groups,
                f,
                indent=2
            )
            f.flush()
            subprocess.run(["tilemaker-db", "populate", f.name])


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
