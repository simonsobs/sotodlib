import argparse as ap
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sotodlib.core import Context
from functools import lru_cache

# Should I implement my own version?
from pycpd import AffineRegistration


def rescale(xy):
    """
    Rescale pointing or template to [0, 1]

    Arguments:

        xy: Pointing or template, should have two columns.

    Returns:

        xy_rs: Rescaled array.
    """
    xy_rs = xy.copy()
    xy_rs[:, 0] /= xy[:, 0].max() - xy[:, 0].min()
    xy_rs[:, 0] -= xy_rs[:, 0].min()
    xy_rs[:, 1] /= xy[:, 1].max() - xy[:, 1].min()
    xy_rs[:, 1] -= xy_rs[:, 1].min()
    return xy_rs


@lru_cache(maxsize=128)
def lev_dist(str1, str2, i1=0, i2=0):
    """
    Calculate levenshtien distance between two strings.
    This is used too see how different two detector ids.

    Arguments:

        str1: First string to compare

        str2: Second string to compare

        i1: Indexes changes in first string.
            This is used by the function when it calls itself recursively and should be left at 0 initially.

        i2: Indexes changes in second string.
            This is used by the function when it calls itself recursively and should be left at 0 initially.

    Returns:

        dist: The levenshtien distance between str1 and str2
    """
    # If we have checked every change in one string
    if i1 == len(str1) or i2 == len(str2):
        return len(str1) - i1 + len(str2) - i2

    # If no change is required here
    if str1[i1] == str2[i2]:
        return lev_dist(str1, str2, i1 + 1, i2 + 1)

    # Otherwise try changes
    return 1 + min(
        lev_dist(str1, str2, i1 + 1, i2),
        lev_dist(str1, str2, i1, i2 + 1),
        lev_dist(str1, str2, i1 + 1, i2 + 1),
    )


def match_template(focal_plane, template):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Fit focal_plane. Should have columns:
                     x, y.

        template: Table containing template of focal plane.
                  Should have columns:
                  x, y.

    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 template[i] = focal_plane[mapping[i]]
    """
    reg = AffineRegistration(**{"X": template, "Y": focal_plane})

    # This should get the maximum probability without collisions
    inv = np.linalg.inv(reg.P)
    mapping = np.argmax(inv, axis=0)

    return mapping


def transform_focal_plane(focal_plane, theta, reflect=False):
    """
    Apply rotation and/or reflection to focal plane.

    Arguments:

        focal_plane: Table containing the following columns:
                     x, y.
                     Note that it is assumed that x and y and rescaled to the region [0, 1].

        theta: Angle to rotate focal plane by.

        reflect: Whether or not to perform reflection about origin.

    Returns:

        focal_plane: Focal plane with transformed coordinates.
                     Note that coordinates are shifted to be in the region [0, 1].
    """
    coords = focal_plane.copy()

    if theta % (2 * np.pi) != 0:
        coords -= 0.5
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        coords = coords @ rotation_matrix
        coords += 0.5

    if reflect:
        coords *= -1
        coords += 1

    return coords


def _focal_plane_min_func(
    theta, readout_id_pointing, pointing_rs, readout_id_smurf, smurf_rs
):
    """
    Function to minimize when trying to fit for focal plane.

    Arguments:

        theta: Angle to rotate focal plane by

        readout_id_pointing: Readout IDs ordered the same as pointing_rs.

        pointing_rs: Table containing pointing with the following columns:
                     xi_0, eta_0.
                     Note that it is assumed that this has been rescaled to [0, 1].

        readout_id_smurf: Readout IDs ordered the same as smurf_rs.

        smurf_rs: Table containing channel map and template with the following columns:
                  x, y.
                  Note that it is assumed that this has been rescaled to [0, 1].

    Returns:

        norm: Distance between rotated pointing and template.
              The metric currently used to get the "distance" between readout_ids is
              the levenshtien distance (see lev_dist function).
    """
    focal_plane = transform_focal_plane(pointing_rs, theta)

    # NOTE: Match template here?
    fp_i = np.lexsort((focal_plane[:, 1], focal_plane[:, 0]))
    focal_plane = focal_plane[fp_i]
    readout_id_pointing = readout_id_pointing[fp_i]
    smurf_i = np.lexsort((smurf_rs[:, 1], smurf_rs[:, 0]))
    smurf = smurf_rs[smurf_i]
    readout_id_smurf = readout_id_smurf[smurf_i]

    dx = focal_plane[:, 0] - smurf[:, 0]
    dy = focal_plane[:, 1] - smurf[:, 1]
    # NOTE: Is there a simpler metric?
    did = np.vectorize(lev_dist, excluded=("i1", "i2"))(
        readout_id_pointing, readout_id_smurf
    ) / len(
        readout_id_pointing[0]
    )  # Assuming all readout_ids are same length

    return np.linalg.norm((dx, dy, did))


def fit_focal_plane(pointing, smurf):
    """
    Fit pointing on sky to focal plane using the channel map and template.

    Arguments:

        pointing: Pointing data. Should be a tuple with elements:
                  readout_id, [xi_0, eta_0].
                  Where readout_id is an (n,) array and [xi_0, eta_0] is (n, 2).

        smurf: Tuple containing channel map and template with elements:
               readout_id, det_id, x, y.
               Where readout_id and det_id are (n,) arrays and [x, y] is (n, 2).

    Returns:

        focal_plane: Table with columns:
                     x, y.
                     Note that this is in the same order as pointing and is rescaled to [0, 1].
    """
    # Rescale pointing and template to be [0, 1]
    pointing_rs = rescale(pointing[1])
    smurf_rs = rescale(smurf[2])

    # Fit for transform from sky to focal plane
    res = opt.minimize(
        _focal_plane_min_func, 0.0, args=(pointing[0], pointing_rs, smurf[0], smurf_rs)
    )
    res_reflect = opt.minimize(
        _focal_plane_min_func,
        0.0,
        args=(
            pointing[0],
            transform_focal_plane(pointing_rs, 0, True),
            smurf[0],
            smurf_rs,
        ),
    )
    if res.fun < res_reflect.fun:
        focal_plane = transform_focal_plane(pointing_rs, res.x, False)
    else:
        focal_plane = transform_focal_plane(pointing_rs, res_reflect.x, True)

    return focal_plane


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    # NOTE: Eventually all of this should just be metadata I can load from a single context?
    parser.add_argument(
        "pointing_data",
        help="Location of HDF5 file containing pointing for each readout channel",
    )
    parser.add_argument(
        "context",
        help="Location of context file which contains the chan map and template",
    )
    # NOTE: Why can't det polangs live in the same table as pointing?
    parser.add_argument(
        "det_polangs",
        help="Location of HDF5 file containing det polang for each readout channel",
    )
    parser.add_argument(
        "bias_groups", help="Location of file mapping bias groups to bands"
    )
    parser.add_argument(
        "-i", "--instrument", help="Instrument mapping from sky to focal plane"
    )
    parser.add_argument(
        "-o", "--observation", help="Which observation to load from context", default=0
    )
    args = parser.parse_args()

    # Load data
    pointing = pd.read_hdf(args.pointing_data)
    context = Context(args.context)
    obs_list = context.obsdb.get()
    meta = context.get_obs(obs_id=obs_list[args.observation]["obs_id"])
    # Load polarization data
    # Load instrument correction if availible

    # Apply instrument to pointing if availible
    # Otherwise try to match from channel map

    # Apply template
    # Get pixel for each channel

    # Set band based on chan map
    # Set pol from wiregrid

    # Add det_id to pointing table
    # Just an f string?
    # Add band and pol as well?
    # Save table


if __name__ == "__main__":
    main()
