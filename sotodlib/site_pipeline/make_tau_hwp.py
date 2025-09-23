import numpy as np
import os
import yaml
import argparse
from typing import Optional, List, Callable

from sotodlib import core, preprocess
from sotodlib.io.metadata import write_dataset
from sotodlib.site_pipeline import util
from sotodlib.utils.procs_pool import get_exec_env

logger = util.init_logger('make_tau_hwp', 'make_tau_hwp')


dtype = [
    ('readout_id', '<U40'),
    ('tau_hwp', 'f4'),
    ('tau_hwp_error', 'f4'),
    ('AQ', 'f4'),
    ('AQ_error', 'f4'),
    ('AU', 'f4'),
    ('AU_error', 'f4'),
    ('redchi2s', 'f4'),
]


def run(
    context_path,
    process_pipe,
    obs_id,
    ver_override=None,
    enc_override=None,
    n_split=1,
):
    """
    Function to run process pipeline for tau_hwp.
    This split detectors into `n_split` chunks and process serially.
    Returns obs_id and ResultSet.
    """
    res_arr = np.array([], dtype=dtype)
    rset = core.metadata.ResultSet.from_friend(res_arr)

    ctx = core.Context(context_path)
    meta = ctx.get_meta(obs_id)
    nper = int(np.ceil(meta.dets.count/n_split))
    logger.info(f'process {obs_id}, dets {meta.dets.count}')
    for i in range(n_split):
        dets = meta.dets.vals[i*nper:(i+1)*nper]
        meta_short = meta.restrict('dets', dets, in_place=False)
        tod = ctx.get_obs(meta_short)
        if ver_override or enc_override:
            ver = ver_override if ver_override else meta.hwp_solution.version
            enc = enc_override if enc_override \
                else meta.hwp_solution.primary_encoder
            tod.hwp_angle = tod.hwp_solution.get(f'hwp_angle_ver{ver}_{enc}')
        pipe = preprocess.Pipeline(process_pipe)
        proc = core.AxisManager(tod.dets, tod.samps)
        for process in pipe:
            logger.info(f'{process.name}, dets {tod.dets.count}')
            process.process(tod, proc)
            process.calc_and_save(tod, proc)
        for i in range(tod.dets.count):
            dic = {'readout_id': tod.dets.vals[i]}
            dic.update({v[0]: tod.tau_hwp[v[0]][i] for v in dtype[1:]})
            rset.append(dic)
    assert len(rset) == meta.dets.count
    return obs_id, rset


def _main(
    executor,
    as_completed_callable: Callable,
    context_path: str,
    process_pipe: dict,
    output_dir: str,
    verbose: Optional[int] = 2,
    overwrite: Optional[bool] = False,
    tags: Optional[List[str]] = None,
    min_ctime: Optional[float] = None,
    max_ctime: Optional[float] = None,
    obs_id: Optional[List[str]] = None,
    hwp_angle_version_override: Optional[int] = None,
    hwp_angle_encoder_override: Optional[int] = None,
    n_split: Optional[int] = 10,
    nprocs: Optional[int] = 1,
):
    """
    Main function for making tau_hwp metadata

    Arguments
    ---------
    context_path: str
        Path to context file.
    process_pipe: str
        Dictionary of process pipeline config
    output_dir: str
        Path to the output directory
    verbose: int
        0: Error, 1: Warning, 2: Info, 3: Debug
    overwrite: bool
        If true, overwrites existing entries in the database
    tags: List of str
        List of tag to use for quering observations
    min_ctime: float
        Minimum timestamp for the beginning of an observation list
    max_ctime: float
        Maximum timestamp for the beginning of an observation list
    obs_id: List of str
        List of obs-ids of particular observations that you want to run
    hwp_angle_version_override: int
        Version of hwp_angle solution to override
    hwp_angle_encoder_override: int
        Encoder of hwp_angle solution to override
    nprocs: int
        Number of processes to use. 1 at site computing
    """

    ctx = core.Context(context_path)
    obs_ids = []
    for tag in tags:
        obslist = ctx.obsdb.query("subtype = 'cal'", tags=[tag])
        for obs in obslist:
            obs_ids.append(obs['obs_id'])

    db_path = os.path.join(output_dir, 'tau_hwp.sqlite')
    h5_fn = 'tau_hwp.h5'
    h5_path = os.path.join(output_dir, 'tau_hwp.h5')
    if os.path.exists(db_path):
        logger.info(f"Mapping {db_path} for the "
                    "archive index.")
        db = core.metadata.ManifestDb(db_path)
    else:
        logger.info(f"Creating {db_path} for the "
                    "archive index.")
        scheme = core.metadata.ManifestScheme()
        scheme.add_exact_match('obs:obs_id')
        scheme.add_data_field('dataset')
        db = core.metadata.ManifestDb(
            db_path,
            scheme=scheme
        )

    futures = []
    for obs_id in obs_ids:
        futures.append(executor.submit(
            run,
            context_path,
            process_pipe,
            obs_id=obs_id,
            ver_override=hwp_angle_version_override,
            enc_override=hwp_angle_encoder_override,
            n_split=n_split,
        ))
    for future in futures:
        try:
            obs_id, rset = future.result()

            logger.info(f'saving {obs_id}...')
            write_dataset(
                data=rset,
                filename=h5_path,
                address=obs_id,
                overwrite=True
            )
            db.add_entry(
                {'obs:obs_id': obs_id, 'dataset': obs_id},
                filename=h5_fn, replace=overwrite,
            )
        except Exception as e:
            logger.error(f'Failed {obs_id} {e}')


def main(cfg):
    rank, executor, as_completed_callable = get_exec_env(nprocs=cfg['nprocs'])
    if rank == 0:
        _main(
            executor=executor,
            as_completed_callable=as_completed_callable,
            **cfg
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to configuration yaml file.")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
