from ..core import metadata


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
        for t in line["tags"]:  # make sure this is also recorded in metric itself
            t.update(log_tags)
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

    def __init__(self, context, monitor, process_name, **kwargs):
        """ In addition to the context and monitor, pass the name of the
        preprocess process to record. It should have a `gen_metric` method
        implemented and `_influx_field` attribute.
        """
        super().__init__(context, monitor, **kwargs)

        from sotodlib.preprocess import Pipeline
        proc = Pipeline.PIPELINE.get(process_name, None)
        if proc is None:
            raise Exception(f"No preprocess process with name {process_name}")
        self._pipe_proc = proc

        # get the field name from the process
        self._influx_field = self._pipe_proc._influx_field

    def _process(self, meta):
        return self._pipe_proc.gen_metric(meta, meta.preprocess)

    def _get_available_obs(self):
        # find preprocess manifest file
        man_file = [p["db"] for p in self.context["metadata"] if p.get("name", "") == "preprocess"]
        if len(man_file) == 0:
            raise Exception(f"No preprocess metadata block in context {self.context.filename}.")

        # load manifest and read available observations
        man_db = metadata.ManifestDb.from_file(man_file[0])
        return [o[0] for o in man_db.get_entries(["\"obs:obs_id\""]).asarray()]
