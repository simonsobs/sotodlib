import yaml
import argparse

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.qa import metrics as qa_metrics
from sotodlib.site_pipeline.monitor import Monitor
from sotodlib.site_pipeline.jobdb import JobManager

logger = sp_util.init_logger("qa_metrics")


def main(config):
    """ Update all metrics specified in the config.
    """
    if isinstance(config, str):
        with open(config, "r") as fh:
            config = yaml.safe_load(fh)

    monitor = Monitor.from_configs(config["monitor"])
    context = core.Context(config["context_file"])

    # get JobDB configuration
    jdb_max_retry = 5  # mark a job as failed after this number of tries
    if isinstance(config["jobdb"], str):  # support previous convention
        jdb = JobManager(sqlite_file=config["jobdb"])
    else:
        jdb = JobManager(sqlite_file=config["jobdb"]["db_file"])
        jdb_max_retry = config["jobdb"].get("max_retry", jdb_max_retry)

    # get a list of metrics from the config
    metrics = []
    for m in config["metrics"]:
        # try to get an instance of this metric
        name = m.pop("name")
        try:
            metric_class = getattr(qa_metrics, name)
        except AttributeError:
            raise Exception(f"No metric named {name} was found.")
        # Instantiate class with remaining elements as kwargs
        metrics.append(metric_class(context=context, monitor=monitor, **m))

    # get a list of obs_id to process and add to the JobDB
    jobs = []
    all_obs_id = set()
    for m in metrics:
        # observations that have not been recorded to Influx
        new_obs = m.get_new_obs()

        # check if failed jobs exist already for these
        jclass = f"{m._influx_meas}.{m._influx_field}"
        skipped_jobs = jdb.get_jobs(jclass=jclass, jstate=["done", "failed", "ignored"])
        skipped_obs = set([j.tags["obs_id"] for j in skipped_jobs])
        logger.debug(f"Skipping {len(skipped_obs & new_obs)} / {len(new_obs)} obs_id for metric {m._influx_meas}.{m._influx_field}")
        new_obs -= skipped_obs

        # create list of jobs to process
        new_jobs = {}
        open_jobs = jdb.get_jobs(jclass=jclass, jstate=["open"])
        open_obs = [j.tags["obs_id"] for j in open_jobs]
        for oid in new_obs:
            if oid in open_obs:
                job = open_jobs[open_obs.index(oid)]
            else:
                job = jdb.create_job(jclass, {"obs_id": oid})
            new_jobs[oid] = job

        jobs.append(new_jobs)
        all_obs_id |= new_obs

    logger.info(f"Found {len(all_obs_id)} obs_id with new metrics to record.")

    # convenience function for marking a job failure
    def _mark_failure(job_id):
        with jdb.locked(job_id) as job:
            job.mark_visited()
            if job.visit_count > jdb_max_retry:
                logger.error(f"Reached max retries for {oid}.")
                job.jstate = "failed"
                return True
        return False

    n = len(all_obs_id)
    n_fail = 0
    # process one obs_id at a time
    for i, oid in enumerate(all_obs_id):
        logger.info(f"Recording metrics for obs_id {oid} ({i+1}/{n})...")
        # load metadata once
        meta_fail = False
        try:
            # this can fail if crucial metadata is unavailable
            meta = context.get_meta(oid, ignore_missing=True)
            # check that obsdb info is available
            if len(meta.obs_info.keys()) < 2:
                logger.warning("This observation is missing obs_info. Skipping.")
                raise ValueError("Missing obs_info.")
        except Exception as e:
            logger.error(f"Failed to load metadata for {oid}.\n{type(e)}: {e}")
            meta_fail = True

        # iterate over metrics and record, or mark failure to load metadata
        for m, m_obs in zip(metrics, jobs):
            if oid in m_obs:
                if meta_fail:
                    if _mark_failure(m_obs[oid]):  # returns True if we reached max retries
                        n_fail += 1
                    continue
                try:
                    m.process_and_record(oid, meta)
                    with jdb.locked(m_obs[oid]) as job:
                        job.mark_visited()
                        job.jstate = "done"
                except Exception:
                    logger.error(
                        f"Processing metric {m._influx_meas}.{m._influx_field} failed.",
                        exc_info=True
                    )
                    if _mark_failure(m_obs[oid]):  # returns True if we reached max retries
                        n_fail += 1

    if n_fail > 0:
        raise RuntimeError(f"This run produced {n_fail} failures.")


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Metrics Configuration File")
    return parser


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
