import yaml
import argparse

from sotodlib import core
import sotodlib.site_pipeline.util as sp_util
from sotodlib.qa import metrics as qa_metrics
from sotodlib.site_pipeline.monitor import Monitor

logger = sp_util.init_logger("qa_metrics")


def main(config):
    """ Update all metrics specified in the config.
    """
    if isinstance(config, str):
        with open(config, "r") as fh:
            config = yaml.safe_load(fh)

    monitor = Monitor.from_configs(config["monitor"])
    context = core.Context(config["context_file"])

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

    # get a list of obs_id to process
    obs_id = []
    for m in metrics:
        obs_id.append(m.get_new_obs())
    all_obs_id = list(set.union(*obs_id))

    logger.info(f"Found {len(all_obs_id)} obs_id with new metrics to record.")

    n = len(all_obs_id)
    i = 1
    # process one obs_id at a time
    for oid in all_obs_id:
        logger.info(f"Recording metrics for obs_id {oid} ({i}/{n})...")
        # load metadata once
        meta = context.get_meta(oid, ignore_missing=True)
        for m, m_obs in zip(metrics, obs_id):
            if oid in m_obs:
                m.process_and_record(oid, meta)
        i += 1


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Metrics Configuration File")
    return parser


if __name__ == '__main__':
    sp_util.main_launcher(main, get_parser)
