.. py:module:: sotodlib.qa.metrics

.. _metrics-module:

==========
QA Metrics
==========

Defines a structure for recording quality assurance (QA) metrics derived from
data at various stages of processing to an Influx database. It is composed of
individual metric objects that are associated to a single Influx field. These
operate on a metadata ``AxisManager`` for a single observation ID, from which
they calculate and return a quantity to be recorded. A metric for a given source
should be based on the ``QAMetric`` class, which provides an interface to the
Influx database. Two abstract methods should be implemented by subclasses: the
``_process`` method is passed the metadata and should calculate the metric;
``_get_available_obs`` should return a list of observation ID's that are
available to be processed (i.e. have an existing metadata entry in the
``Manifest`` file referenced in the ``Context``).


Base Class
----------

.. autoclass:: sotodlib.qa.metrics.QAMetric
    :members:


Metric-recording Script
-----------------------

.. autofunction:: sotodlib.site_pipeline.record_qa.main

For a list of metrics provided in a config file, this script determines which
observation ID's are available and have yet to be recorded in the Influx
database, loads the metadata defined in the given context file one observation at
a time, calculates each metric, and records it to Influx. The configuration file
should contain three blocks:

    ``monitor``
        Specifies an InfluxDB monitor as defined in
        :meth:`sotodlib.site_pipeline.monitor.Monitor`
    ``context_file``
        The context file that includes all the relevant metadata.
    ``metrics``
        A list of metrics, with each entry including at least a ``name`` field
        giving the name of the class to instantiate, with all other parameters
        passed to its ``__init__`` method.

For example, a config could look like::

    monitor:
        host: "daq-dev"
        port: "8086"
        database: "pipeline-dev"
        username: "monitor"
        password: "crepe-handmade-trio"
        path: ""
        ssl: False
    context_file: "/path/to/context.yaml"
    metrics:
      - name: "PreprocessQA"
        process_name: "det_bias_flags"
      - name: "HWPSolSuccess"
        encoder: "1"
      - name: "HWPSolNumSamples"
        encoder: "1"
      - name: "HWPSolNumFlagged"
        encoder: "1"
      - name: "HWPSolMeanRate"
        encoder: "1"
      - name: "HWPSolMeanTemplate"
        encoder: "1"
      - name: "HWPSolSuccess"
        encoder: "2"
      - name: "HWPSolNumSamples"
        encoder: "2"
      - name: "HWPSolNumFlagged"
        encoder: "2"
      - name: "HWPSolMeanRate"
        encoder: "2"
      - name: "HWPSolMeanTemplate"
        encoder: "2"
      - name: "HWPSolVersion"
      - name: "HWPSolPrimaryEncoder"
      - name: "HWPSolOffcenter"
      - name: "HWPSolOffcenterErr"

In the above config, some of the metrics have parameters like ``process_name``
or ``encoder`` that are specific to their subclass.


Metric Subclasses
-----------------

Preprocess
::::::::::

Metrics to be derived from the output of the ``preprocess`` module share a
generic class. A ``_PreProcess`` subclass that supports QA metrics should
implement its ``gen_metric`` abstract method to calculate a summary quantity for
its output. In the ``record_qa`` configuration, the name of the process to
instantiate for a given metric should be given as the ``process_name`` argument.

.. autoclass:: sotodlib.qa.metrics.PreprocessQA
    :members:


HWP Angle Solution
::::::::::::::::::

Some HWP solution metrics are derived for both encoder solutions separately, so
the ``encoder`` argument should be specified in the config.

.. autoclass:: sotodlib.qa.metrics.HWPSolQA
    :members:
.. autoclass:: sotodlib.qa.metrics.HWPSolSuccess
.. autoclass:: sotodlib.qa.metrics.HWPSolNumSamples
.. autoclass:: sotodlib.qa.metrics.HWPSolNumFlagged
.. autoclass:: sotodlib.qa.metrics.HWPSolMeanRate
.. autoclass:: sotodlib.qa.metrics.HWPSolMeanTemplate
.. autoclass:: sotodlib.qa.metrics.HWPSolVersion
.. autoclass:: sotodlib.qa.metrics.HWPSolPrimaryEncoder
.. autoclass:: sotodlib.qa.metrics.HWPSolOffcenter
.. autoclass:: sotodlib.qa.metrics.HWPSolOffcenterErr
