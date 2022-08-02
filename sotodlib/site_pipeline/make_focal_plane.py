import argparse as ap
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sotodlib.core import Context
from functools import lru_cache

# Should I implement my own version?
# NOTE: Need this fork for priors https://github.com/agporto/pycpd
# Should be merged soon https://github.com/siavashk/pycpd/pull/57
from pycpd import AffineRegistration, ConstrainedDeformableRegistration


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


def match_template(focal_plane, template, out_thresh=0):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Focal plane after either having optics model applied or being fit with channel map.
                     Should have columns: x, y or x, y, pol.

        template: Table containing template of focal plane.
                  Should have columns: x, y os x, y, pol.

        out_thresh: Threshold at which points will be considered outliers.
                    Should be in range [0, 1) and is checked against the
                    probability that a point matches its mapped point in the template.

    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 focal_plane[i] = template[mapping[i]]

        outliers: Indices of points that are outliers.
                  Note that this is in the basis of mapping and focal_plane, not template.
    """
    reg = AffineRegistration(**{"X": focal_plane, "Y": template})

    # This should get the maximum probability without collisions
    inv = np.linalg.inv(reg.P)
    mapping = np.argmax(inv, axis=0)

    outliers = np.array([])
    if out_thresh > 0:
        outliers = np.where(reg.P[range(reg.P.shape[0]), mapping] < out_thresh)[0]

    return mapping, outliers


def match_chan_map(pointing, template, out_thresh=0, weight=2):
    """
    Fit pointing on sky to focal plane using the channel map and template.

    Arguments:

        pointing: Pointing data. Should be a tuple with elements:
                  readout_id, [xi_0, eta_0] or readout_id, [xi_0, eta_0, pol].
                  Where readout_id is an (n,) array and [xi_0, eta_0(, pol)] is (n, 2(3)).

        template: Tuple containing channel map and template with elements:
                  readout_id, [x, y] or readout_id, [x, y, pol]
                  Where readout_id is an (n,) array and [x, y(, pol)] is (n, 2(3)).

        out_thresh: Threshold at which points will be considered outliers.
                    Should be in range [0, 1) and is checked against the
                    probability that a point matches its mapped point in the template.

        weight: How strongly to weigh the channel map.
                Should be greater than 0, with a higher number indicating more confidence.

    Returns:

        mapping: Mapping between elements in template and pointing.
                 pointing[i] = template[mapping[i]]

        outliers: Indices of points that are outliers.
                  Note that this is in the basis of mapping and pointing, not template.
    """
    # Rescale pointing and template to be [0, 1]
    pointing_rs = rescale(pointing[1])
    template_rs = rescale(template[1])

    # Sort to make priors easier
    p_i = np.argsort(pointing[0])
    t_i = np.argsort(template[0])

    target_id = np.arange(len(pointing_rs))
    source_id = np.arange(len(template_rs))

    reg = ConstrainedDeformableRegistration(
        **{"X": pointing_rs[p_i], "Y": template_rs[t_i]},
        e_alpha=1.0 / weight,
        source_id=source_id,
        target_id=target_id
    )

    # Undo sorting
    p_ii = np.argsort(p_i)
    t_ii = np.argsost(t_i)
    P = reg.P[np.ix_(t_ii, p_ii)]

    # This should get the maximum probability without collisions
    inv = np.linalg.inv(P)
    mapping = np.argmax(inv, axis=0)

    outliers = np.array([])
    if out_thresh > 0:
        outliers = np.where(P[range(P.shape[0]), mapping] < out_thresh)[0]

    return mapping, outliers


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
