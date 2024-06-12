import os
import argparse
import numpy as np

from sotodlib import core
from sotodlib.site_pipeline import util

logger = util.init_logger(__name__)


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('telescope', choices=['satp1', 'satp2', 'satp3'])
    parser.add_argument('-o', dest='output_dir', default='./')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def make_model(telescope):
    """
    Function to make hwp_angle_model. This will be manually maintained.

    Return:
        hwp_angle_model AxisManager

    Note:
        sign_matrix
            relative sign of sign estimators

        mechanical_offset (rad)
             * encoder offset: -1.66 deg
             * encoder assembly offset: 90 deg or -90 deg
             * offset between the encoder origin and the optical axis of
               the achromatic hwp at 90 GHz

        time_offset (rad)
            angle offset that behaves like a time domain offset
            encoder origin offset (center of reference slot to next edge)
            June 2024; value confirmed by WG measurememnts CW vs. CCW.

        hwp_angle_model correction will be
        tod.hwp_angle = sign * (hwp_solution.hwp_angle + time_offset)
                      + mechanical_offset
    """
    out = core.AxisManager()
    if telescope == 'satp1':
        sign = core.AxisManager()
        sign.wrap('pid', 1)
        sign.wrap('offcenter', -1)
        out.wrap('mechanical_offset_1', np.deg2rad(-1.66 - 90 + 49.1))
        out.wrap('mechanical_offset_2', np.deg2rad(-1.66 + 90 + 49.1))

    elif telescope == 'satp3':
        sign = core.AxisManager()
        sign.wrap('pid', -1)
        sign.wrap('offcenter', 1)
        out.wrap('mechanical_offset_1', np.deg2rad(-1.66 + 90 - 2.29))
        out.wrap('mechanical_offset_2', np.deg2rad(-1.66 - 90 - 2.29))

    else:
        raise ValueError('Not supported yet')

    out.wrap('sign_matrix', sign)
    out.wrap('time_offset', np.deg2rad(-1. * 360 / 1140 * 3 / 2))
    return out


def main(output_dir, overwrite, telescope):

    db_file = os.path.join(output_dir, 'hwp_angle_model.sqlite')
    h5_rel = 'hwp_angle_model.h5'
    h5_file = os.path.join(output_dir, h5_rel)

    if os.path.exists(db_file):
        logger.info(f"{db_file} exists, looking for updates")
        db = core.metadata.ManifestDb(db_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Creating {db_file}.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(db_file, scheme=scheme)

    logger.info(f'Making {telescope} model.')
    meta = make_model(telescope)
    logger.info(f'Saving. {h5_file} and {db_file}')
    meta.save(h5_file, 'hwp_angle_model', overwrite=overwrite)
    db.add_entry(
        {'dataset': 'hwp_angle_model'},
        filename=h5_rel,
        replace=overwrite
    )


if __name__ == '__main__':
    util.main_launcher(main, get_parser)
