QDS Monitor
===========
The QDS Monitor is meant to be a simple to use class that allows users to
publish the results of their calculations to a live monitor. The live monitor
backend is an Influx Database, which is used with the SO Data Acquisition
system, known as the Observatory Control System. This allows us to use the same
live monitoring interface, Grafana.

Overview
--------
The ``Monitor`` class wraps the InfluxDB interface, and provide a few simple
methods -- ``check``, ``record``, and ``write`` -- detailed in the
:ref:`API section <monitor_api>`.

``check`` is meant to be used to check if the calculation already has been
performed for the given observation/tag set. This can be used to ensure
expensive calculations are not repeated when running batch jobs. ``record``
takes your calculations, timestamps, and a set of identifying tags, and queues
them for batch writing to the InfluxDB. Finally, ``write`` will write your
recorded results to the InfluxDB, clearing the queue.

This perhaps is best demonstrated with some examples, shown in the next section.

Examples
--------
Simple Pseudocode
`````````````````
The general outline we're aiming for is as follows::

    from qds import qds
    
    # Initialize DB Connection
    monitor = qds.Monitor('localhost', 8086, 'qdsDB')
    
    # Load observation
    tod = so_data_load.load_observation(context,                       
              observation_id, detectors_list)
    
    # Compute statistic
    result = interesting_calculation(tod)
    
    # Tag and write to DB
    tags = {'telescope': 'LAT', 'wafer': wafer_name}
    monitor.record('white_noise_level', result, timestamp, tags)
    monitor.write()

Realworld Example
`````````````````
For an actual working example we look at the script located in
``pwg-scripts/pwg-qds/bin/sim_array_noise_monitor.py``. As we walk through it
here we simplify some parts and omit some descriptive prints. Feel free to look
directly at the script for more details.

To start, we will import the ``qds`` module and create our ``Monitor`` object.
You will need to know the address and port for your InfluxDB, as well as the
name of the database within InfluxDB that you want to write to.::

    from qds import qds

    monitor = qds.Monitor('localhost', 8086, 'qds')

Let's say we want to load some of the sims, we'll create our Context and get
the observations with::

    context = core.Context('pipe_s0001_v2.yaml')
    observations = context.obsfiledb.get_obs()

Then we can, for example, loop over all observations, determining the detectors
and wafers in each observation::

    for obs_id in observations:
        c = context.obsfiledb.conn.execute('select distinct DS.name, DS.det from detsets DS '
                        'join files on DS.name=files.detset '
                        'where obs_id=?', (obs_id,))
        dets_in_obs = [tuple(r) for r in c.fetchall()]
        wafers = np.unique([x[0] for x in dets_in_obs])

We'll run our calculation for each wafer, so let's loop over those now,
building a detector list for the wafer, and loading the TOD for just those
detectors and computing their FFTs::

    for wafer in wafers:
        det_list = build_det_list(dets_in_obs, wafer)
        tod = so_data_load.load_observation(context.obsfiledb, obs_id, dets=det_list)

        # Compute ffts
        ffts, freqs = rfft(tod)
        det_white_noise = calculate_noise(tod, ffts, freqs)

Now we want to save our results to the monitor. To do this, we'll need two
other lists, one for the timestamps associated with each noise value (in this
case, these are all the same, and use the first timestamp in the TOD), and one
for the tags for each noise value (in this example we tag each detector
individually with their detector ID, along with the wafer it is on and what
telescope we're working with -- this probably is in the context somewhere, but
I'm just writing in SAT1)::

        timestamps = np.ones(len(det_white_noise))*tod.timestamps[0]
        base_tags = {'telescope': 'SAT1', 'wafer': wafer}
        tag_list = []
        for det in det_list:
            det_tag = dict(base_tags)
            det_tag['detector'] = det
            tag_list.append(det_tag)
        log_tags = {'observation': obs_id, 'wafer': wafer}
        monitor.record('white_noise_level', det_white_noise, timestamps, tag_list, 'detector_stats', log_tags=log_tags)
        monitor.write()

We also include a set of log tags, these are to record that we've completed
this calculation for this observation and wafer. Lastly we record the
measurement, giving it the name "white_noise_level", passing our three lists of
equal length (``det_white_noise``, ``timestamps``, ``tag_list``), and recording
the measurement as completed in the "detector_stats" log with the observation ID
and wafer log tags.

Where these log tags could come in handy is if we need to stop and restart our
calculation and want to skip recomputing the results. Since we saved the wafer
along with the observation ID it would make sense to check at the wafer level
loop::

    for wafer in wafers:
        # Check calculation completed for this wafer
        check_tags = {'wafer': wafer} 
        if monitor.check('white_noise_level', obs_id, check_tags):
            continue

Add this to the top of our wafer loop would skip already recorded wafers for
this observation id.

.. _monitor_api:

API
---
.. autoclass:: qds.qds.Monitor
    :members:
