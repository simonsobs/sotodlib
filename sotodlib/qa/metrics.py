import re
import numpy as np

from ..core import metadata


def _has_tag(root, keys):
    keys = keys.split(".")
    for key in keys:
        if key in root:
            root = root[key]
        else:
            return False
    return True

def _get_tag(root, keys, i):
    keys = keys.split(".")
    for key in keys:
        root = root[key]
    return root[i]


class QAMetric(object):
    """ Base class for quality assurance metrics to be recorded in Influx,
    derived from processed metadata files.

    The base class includes methods for recording a metric value to Influx,
    checking if a given obs_id has been recorded for this metric, and fetching
    a list of obs_id's that can be recorded.

    Metrics are labelled by the observation ID they describe, an Influx 'measurement'
    and 'field', and any number of additional tags. A measurement can have multiple
    fields associated with it, and here we've tried to organize the metrics so that
    a source of information (e.g. HWP angle solutions) is a measurement and relevant
    quantities describing it (e.g. HWP success, mean rate, etc) are fields.

    Subclasses should implement the `_process` and `_get_available_obs` methods to
    access the source of data they are going to tap, as well as define the
    `_influx_meas` and `_influx_field` attributes.
    """

    # these will be defined by child classes
    _influx_meas = None  # measurement to record to
    _influx_field = None  # the field to populate

    def __init__(self, context, monitor, log="qa_metrics_log"):
        """ A QA metric base class.

        Arguments
        ---------
        context : core.Context
            Context that includes all necessary metadata to generate metrics.
        monitor : site_pipeline.monitor.Monitor
            InfluxDB connection.
        log : str
            InfluxDB measurement that is used to log new entries.
        """

        self.context = context
        self.monitor = monitor
        self._influx_log = log

    def exists(self, obs_id, tags={}):
        """ Check if a metric exists for this obs_id in Influx.

        Arguments
        ---------
        obs_id : str
            The observation ID to check.
        tags : dict (optional)
            Further restrict to given tags.

        Returns
        -------
        exists : bool
            Whether it exists or not.
        """
        return self.monitor.check(self._influx_field, obs_id, tags, log=self._influx_log)

    def get_existing_obs(self):
        """ Get a list of observations already recorded to Influx.
        """
        # query influx log measurement for observations of this field
        res = self.monitor.client.query(
            f"select {self._influx_field}, observation from {self._influx_log}"
        ).get_points()
        return [r["observation"] for r in res]

    def process_and_record(self, obs_id, meta=None):
        """ Generate a metric for this obs_id and record it to InfluxDB.

        Arguments
        ---------
        obs_id : str
            The observation ID to process.
        meta : AxisManager (optional)
            The metadata for this observation ID. If not provided will read from file.
        """
        if meta is None:
            meta = self.context.get_meta(obs_id, ignore_missing=True)
        if meta.obs_info.obs_id != obs_id:
            raise Exception(f"Metadata does not correspond to obs_id {obs_id}.")
        line = self._process(meta)
        log_tags = {"observation": obs_id}  # used to identify this entry
        self.monitor.record(**line, log=self._influx_log, measurement=self._influx_meas, log_tags=log_tags)
        self.monitor.write()

    def get_new_obs(self):
        """ Get a list of available observations not yet recorded to InfluxDB.
        """
        # get a list of obs_id to update
        avail_obs = set(self._get_available_obs())
        exist_obs = set(self.get_existing_obs())
        return avail_obs - exist_obs

    def _process(self, meta):
        """ Implement this to process an actual metric."""
        raise NotImplementedError

    def _get_available_obs(self):
        """ Implement this for a given metric."""
        raise NotImplementedError


class PreprocessQA(QAMetric):
    """ A metric derived from a preprocesstod process.
    """

    _influx_meas = "preprocesstod"
    _process_args = {}

    def __init__(self, context, monitor, process_name, process_args={}, **kwargs):
        """ In addition to the context and monitor, pass the name of the
        preprocess process to record. It should have a `gen_metric` method
        implemented and `_influx_field` attribute.
        """
        self._process_args = process_args
        super().__init__(context, monitor, **kwargs)

        from sotodlib.preprocess import Pipeline
        proc = Pipeline.PIPELINE.get(process_name, None)
        if proc is None:
            raise Exception(f"No preprocess process with name {process_name}")
        self._pipe_proc = proc

        # get the field name from the process
        self._influx_field = self._pipe_proc._influx_field

    def _process(self, meta):
        return self._pipe_proc.gen_metric(meta, meta.preprocess, **self._process_args)

    def _get_available_obs(self):
        # find preprocess manifest file
        man_file = [p["db"] for p in self.context["metadata"] if p.get("name", "") == "preprocess"]
        if len(man_file) == 0:
            raise Exception(f"No preprocess metadata block in context {self.context.filename}.")

        # load manifest and read available observations
        man_db = metadata.ManifestDb.from_file(man_file[0])
        return [o[0] for o in man_db.get_entries(["\"obs:obs_id\""]).asarray()]


class HWPSolQA(QAMetric):
    """ Base class for metrics derived from HWP angle solutions. Subclasses should
    implement the `_process` method. Some quantities are derived twice, once for each
    encoder, and these will require and `encoder` parameter to be provided to select
    which one to produce a metric for. This is indicated by setting the `_needs_encoder`
    attribute to `True`.
    """

    _influx_meas = "hwp_solution"
    _needs_encoder = False  # set this flag if encoder needs to be specified

    def __init__(self, context, monitor, encoder=None, **kwargs):
        super().__init__(context, monitor, **kwargs)

        self._encoder = encoder
        self._tags = {}
        if encoder is not None:
            if str(encoder) not in ["1", "2"]:
                raise Exception(f"Invalid value {encoder} for encoder parameter.")
            self._tags = {"encoder": self._encoder}
        elif self._needs_encoder:
            raise Exception("This metric needs an encoder to be specified on creation.")

    def _get_available_obs(self):
        # find preprocess manifest file
        man_file = [p["db"] for p in self.context["metadata"] if p.get("name", "") == "hwp_solution"]
        if len(man_file) == 0:
            raise Exception(f"No hwp_solution metadata block in context {self.context.filename}.")

        # load manifest and read available observations
        man_db = metadata.ManifestDb.from_file(man_file[0])
        obs_re = re.compile("^obs_*.")
        return [o[0] for o in man_db.get_entries(["\"obs:obs_id\""]).asarray() if obs_re.match(o[0])]


class HWPSolSuccess(HWPSolQA):
    """ Records success of the HWP angle solution calculation, for each encode."""

    _influx_field = "logger"
    _needs_encoder = True

    def _process(self, meta):
        success = [meta.hwp_solution[f"logger_{self._encoder}"] == "Angle calculation succeeded"]
        obs_time = [meta.obs_info.timestamp]
        return {
            "field": self._influx_field,
            "values": success,
            "timestamps": obs_time,
            "tags": [self._tags],
        }


class HWPSolPrimaryEncoder(HWPSolQA):
    """ The primary encoder used for the HWP angle calculation."""

    _influx_field = "primary_encoder"

    def _process(self, meta):
        # no tags for this metric
        return {
            "field": self._influx_field,
            "values": [meta.hwp_solution["primary_encoder"]],
            "timestamps": [meta.obs_info.timestamp],
            "tags": [self._tags],
        }


class HWPSolVersion(HWPSolQA):
    """ The version of the solution used for the HWP angle calculation."""

    _influx_field = "version"

    def _process(self, meta):
        # no tags for this metric
        return {
            "field": self._influx_field,
            "values": [meta.hwp_solution["version"]],
            "timestamps": [meta.obs_info.timestamp],
            "tags": [self._tags],
        }


class HWPSolOffcenter(HWPSolQA):
    """ Calculated offcentering of HWP angle solution."""

    _influx_field = "offcenter"

    def _process(self, meta):
        # no tags for this metric
        return {
            "field": self._influx_field,
            "values": [meta.hwp_solution["offcenter"][0]],
            "timestamps": [meta.obs_info.timestamp],
            "tags": [self._tags],
        }


class HWPSolOffcenterErr(HWPSolQA):
    """ Standard error on the offcentering of HWP angle solution."""

    _influx_field = "offcenter_err"

    def _process(self, meta):
        # no tags for this metric
        return {
            "field": self._influx_field,
            "values": [meta.hwp_solution["offcenter"][1]],
            "timestamps": [meta.obs_info.timestamp],
            "tags": [self._tags],
        }


class HWPSolNumSamples(HWPSolQA):
    """ The total number of encoder samples."""

    _influx_field = "num_samples"
    _needs_encoder = True

    def _process(self, meta):
        flag_key = f"filled_flag_{self._encoder}"
        frac = [meta.hwp_solution[flag_key].size]
        obs_time = [meta.obs_info.timestamp]
        return {
            "field": self._influx_field,
            "values": frac,
            "timestamps": obs_time,
            "tags": [self._tags],
        }


class HWPSolNumFlagged(HWPSolQA):
    """ The number of encoder samples that were flagged."""

    _influx_field = "num_flagged"
    _needs_encoder = True

    def _process(self, meta):
        flag_key = f"filled_flag_{self._encoder}"
        frac = [meta.hwp_solution[flag_key].sum()]
        obs_time = [meta.obs_info.timestamp]
        return {
            "field": self._influx_field,
            "values": frac,
            "timestamps": obs_time,
            "tags": [self._tags],
        }


class HWPSolMeanRate(HWPSolQA):
    """ The mean calculated HWP angle rate, for each encoder."""

    _influx_field = "mean_rate"
    _needs_encoder = True

    def _process(self, meta):
        good_samp = ~meta.hwp_solution[f"filled_flag_{self._encoder}"]
        nsamp = good_samp.sum()
        rate = np.nan if nsamp == 0 else (meta.hwp_solution[f"hwp_rate_{self._encoder}"] * good_samp).sum() / nsamp
        obs_time = [meta.obs_info.timestamp]
        return {
            "field": self._influx_field,
            "values": [rate],
            "timestamps": obs_time,
            "tags": [self._tags],
        }


class HWPSolMeanTemplate(HWPSolQA):
    """ The mean of the calculated template magnitude."""

    _influx_field = "mean_template"
    _needs_encoder = True

    def _process(self, meta):
        obs_time = [meta.obs_info.timestamp]
        return {
            "field": self._influx_field,
            "values": [np.mean(np.abs(meta.hwp_solution[f"template_{self._encoder}"]))],
            "timestamps": obs_time,
            "tags": [self._tags],
        }
