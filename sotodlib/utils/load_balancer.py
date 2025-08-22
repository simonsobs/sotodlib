import sqlite3
from typing import TYPE_CHECKING, List, Optional, Tuple

import astropy.units as astro_units
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from sklearn.metrics import pairwise_distances

if TYPE_CHECKING:
    import mpi4py  # Only imported during type checking


def az_el_to_radec(ctime: int, az: float, el: float, lat: float, lon: float, alt: float = 0) -> Tuple[float, float]:
    """Convert azimuth and elevation angles to right ascension and declination.

    Parameters
    ----------
    ctime : int
        Unix timestamp of the observation.
    az : float
        Azimuth angle in degrees.
    el : float
        Elevation angle in degrees.
    lat : float
        Latitude of the observer in degrees.
    lon : float
        Longitude of the observer in degrees.
    alt : float, optional
        Altitude of the observer in meters, by default 0

    Returns
    -------
    Tuple[float, float]
        Right ascension and declination in degrees.
    """

    # Convert Unix timestamp to astropy Time object
    time = Time(ctime, format="unix")

    # Define observer's location
    location = EarthLocation(
        lat=lat * astro_units.deg, lon=lon * astro_units.deg, height=alt * astro_units.m
    )

    # Create AltAz frame
    altaz = AltAz(obstime=time, location=location)

    # Define target coordinates in AltAz
    altaz_coord = SkyCoord(az=az * astro_units.deg, alt=el * astro_units.deg, frame=altaz)

    # Convert to equatorial (RA, Dec)
    radec = altaz_coord.transform_to("icrs")

    return radec.ra.deg, radec.dec.deg  # Return RA, Dec in degrees


def balanced_kmeans_soft_minmax(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    min_cluster_tolerance: Optional[float] = None,
    max_cluster_tolerance: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """This algorithms performs balanced K-means clustering with soft min-max constraints.
    It initially clusters the observations based on the distance of their centers. It
    then checks the cluster sizes and tries to balance them by moving observations
    between clusters. The algorithm using an ideal size metric that is equal to the number of observations
    divided by the number of clusters. Any selected tolerance is a factor of this ideal size.

    Parameters
    ----------
    X : np.ndarray
        The input data to cluster.
    k : int
        The number of clusters.
    max_iter : int, optional
        The maximum number of iterations to run the algorithm, by default 100.
    min_cluster_tolerance : Optional[float], optional
        The minimum tolerance for cluster size as a percentage of the ideal size, by default None.
    max_cluster_tolerance : Optional[float], optional
        The maximum tolerance for cluster size as a percentage of the ideal size, by default None.
    random_state : Optional[int], optional
        The random seed for initialization, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The cluster assignments and the final cluster centers.
    """

    np.random.seed(random_state)
    n = len(X)
    ideal_size = n // k
    min_cluster_size = (
        ideal_size - 1
        if min_cluster_tolerance is None
        else int(min_cluster_tolerance * ideal_size)
    )
    max_cluster_size = (
        ideal_size + 1
        if max_cluster_tolerance is None
        else int(max_cluster_tolerance * ideal_size)
    )

    # Initialize centroids randomly
    centroids = X[np.random.choice(n, k, replace=False)]

    for it in range(max_iter):
        # Compute distances (n_samples x k)
        distances = pairwise_distances(X, centroids)
        assignments = -np.ones(n, dtype=int)
        cluster_counts = np.zeros(k, dtype=int)

        # Step 1: Greedy assignment under max_cluster_size
        for i in np.argsort(np.min(distances, axis=1)):
            nearest = np.argsort(distances[i])
            for cluster_id in nearest:
                if cluster_counts[cluster_id] < max_cluster_size:
                    assignments[i] = cluster_id
                    cluster_counts[cluster_id] += 1
                    break

        # Step 2: Fix clusters below min_cluster_size
        for cluster_id in np.where(cluster_counts < min_cluster_size)[0]:
            needed = min_cluster_size - cluster_counts[cluster_id]
            # Find candidate points from overfilled clusters
            donor_ids = np.where(cluster_counts > min_cluster_size)[0]
            donor_points = np.where(np.isin(assignments, donor_ids))[0]
            # Get distances of donor points to the underfilled cluster centroid
            donor_distances = distances[donor_points, cluster_id]
            # Sort donor points by closeness to the underfilled centroid
            move_indices = donor_points[np.argsort(donor_distances)][:needed]
            # Move the points
            for idx in move_indices:
                old_cluster = assignments[idx]
                assignments[idx] = cluster_id
                cluster_counts[old_cluster] -= 1
                cluster_counts[cluster_id] += 1

        # Step 3: Update centroids
        new_centroids = np.array(
            [
                X[assignments == i].mean(axis=0)
                if np.any(assignments == i)
                else centroids[i]
                for i in range(k)
            ]
        )

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return assignments, centroids


def balance(obs_ids: List[str], comm: "mpi4py.MPI.Comm", comm_size: int, rank: int) -> List[str]:
    """Balance the observation IDs across the available MPI ranks based on where they are
    located on the sky.

    Parameters
    ----------
    obs_ids : List[str]
        The list of observation IDs to balance.
    comm : mpi4py.MPI.Comm
        The MPI communicator.
    comm_size : int
        The total number of MPI ranks.
    rank : int
        The rank of the current process.

    Returns
    -------
    List[str]
        The balanced list of observation IDs for the current rank.
    """

    if rank == 0:
        # TODO: Generalize to use any context. Currently it is only ACT.
        conn = sqlite3.connect(
            "/scratch/gpfs/SIMONSOBS/users/ip8725/act_test/acts19_det_set.sqlite"
        )
        # Enable dictionary-like row access
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        obs_json = {}
        for obs_id in obs_ids:
            cursor.execute(
                "select ctimemax, azmax, ctimemin, azmin, el from det_set where obs_id = ?",
                (obs_id.split("_")[0],),
            )
            d = cursor.fetchone()
            d = dict(d)
            ra_max, dec_max = az_el_to_radec(
                ctime=d["ctimemax"],
                az=np.degrees(d["azmax"]),
                el=np.degrees(d["el"]),
                lat=-22.958611,
                lon=-67.787500,
                alt=5190,
            )
            ra_min, dec_min = az_el_to_radec(
                ctime=d["ctimemin"],
                az=np.degrees(d["azmin"]),
                el=np.degrees(d["el"]),
                lat=-22.958611,
                lon=-67.787500,
                alt=5190,
            )

            obs_json[obs_id] = {
                "center": [(ra_max + ra_min) / 2, (dec_max + dec_min) / 2]
            }

        Xp = np.array([[o["center"][0], o["center"][1]] for o in obs_json.values()])
        assignments2, _ = balanced_kmeans_soft_minmax(
            Xp,
            comm.size,
            max_iter=100,
            random_state=0,
        )
        plan = {}
        for obs, c in zip(obs_json.items(), assignments2):
            t = plan.get(c, [])
            t.append(obs[0])
            plan[c] = t
        # print(obs_json, Xp, p_kmeans)
        # print(plan)
        plan_list = [plan.get(r, []) for r in range(comm_size)]
        print(f"PLAN: {plan_list}", flush=True)
    else:
        plan_list = None

    rplan = comm.scatter(plan_list, root=0)
    return rplan
