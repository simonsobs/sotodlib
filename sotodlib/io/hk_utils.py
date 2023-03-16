import numpy as np
import yaml
import itertools
import logging

from so3g.hk import load_range
from sotodlib import core
import sotodlib.io.load_smurf as ls
from sotodlib.io.load_smurf import Observations

from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def _get_swap_dict(d):
    """
    Swap dictionary keys and values; useful for organizing fields in
    config file by device name instead of alias in get_grouped_hkdata().
    """
    return {v: k for k, v in d.items()}


def get_grouped_hkdata(start, stop, config):
    """
    Takes output from load_range(), reconfigures load_range() dictionary
    slightly. Groups HK field data by corresponding device/feed. Outputs
    that list of data.

    Parameters:
        start: Earliest time to search for data. Can be datetime objects,
            unix timestamps, int, floats.
        stop: Latest time to search for data. See start parameter above for
            note on time formats.
        config (str): Name of a .yaml file for loading fields and parsing
            information for housekeeping devices

    Returns:
        get_grouped_data (list): grouped list of data per HK feed/device;
            useful for turning into a grouped set of axismanagers, or for
            other grouped data analysis purposes.

    Notes
    -----

    An example config file looks like:

        data_dir: '/mnt/so1/data/chicago-latrt/hk/'

        field_list:
            'bf_4k' : 'observatory.LSA22HG.feeds.temperatures.Channel_06_T'
            'xy_stage_x': 'observatory.XYWing.feeds.positions.x'
            'xy_stage_y': 'observatory.XYWing.feeds.positions.y'
    """
    # call load_range()
    logger.debug("running load_range()")
    hkdata = load_range(start, stop, config=config)

    # load all fields from config file
    with open(config, 'r') as file:
        hkconfig = yaml.safe_load(file)

    field_dict = hkconfig['field_list']

    # output a dict of field name, alias, timestamps, and data
    # for fields that are online given start, stop and keys from load_range()
    hk_aliases = list(hkdata.keys())

    online_fields = {}
    for alias in hk_aliases:
        field_info = {alias: field_dict[alias]}
        online_fields.update(field_info)

    fields_data = {}
    for alias in online_fields:
        time = hkdata[alias][0]
        data = hkdata[alias][1]
        field = online_fields[alias]

        info = {field: [alias, time, data]}
        fields_data.update(info)

    # now group field names by feed/device
    sorted_fields = sorted(_get_swap_dict(online_fields))

    grouped_feeds = []
    key_func = lambda field: field.split('.')[0:4]
    for key, group in itertools.groupby(sorted_fields, key_func):
        grouped_feeds.append(list(group))

    # use grouped_feeds and fields_data to output a grouped list of data per
    # feed/device, ready to input to an axismanager
    grouped_data = []
    for group in grouped_feeds:
        device_data = {}
        for field in group:
            field_dict = {field: fields_data[field]}
            device_data.update(field_dict)

        grouped_data.append(device_data)

    return grouped_data


def make_hkaman(grouped_data):
    """
    Takes data from get_grouped_hkdata(), tests whether feed/device is
    cosampled, outputs axismanager(s) for either case.

    Parameters:
        grouped_data (list): grouped list of data per HK feed/device; output
            from get_grouped_hkdata()

    Returns:
        merged_amans (AxisManager): AxisManager of HK AxisManagers. It can
            contain 1 axismanager per cosmapled device, as well as 1 axismanger
            per 'channel' for a non-cosampled device. All HK AxisManagers are
            wrapped in 1 ultimate axismanager to streamline data analysis, and
            make it easier to explore axismanager/device keys.
    """
    amans = []
    device_names = []
    for group in grouped_data:
        data = {}
        aliases = []
        times = []
        for field in group:
            # arrange data per device to input into aman
            alias = group[field][0]
            aliases.append(alias)

            time = group[field][1]
            times.append(time)

            device_data = group[field][2]

            info = {alias: device_data}
            data.update(info)

        try:
            # check whether hk device is cosampled
            assert np.all([len(t) == len(times[0]) for t in times])
            # TODO: diff of times, if less than say 10%??, treat as cosampled
        except AssertionError:
            # if not cosampled, make aman for each field in device/feed
            logger.debug("{} is not cosampled. Making separate axis managers \
                         for this case".format('.'.join(field.split('.')[0:3])))
            for field in group:
                alias = group[field][0]
                time = group[field][1]
                data = group[field][2]

                device_name = field.split('.')[1] + '_' + alias
                device_names.append(device_name)

                device_axis = 'hklabels_' + device_name
                samps_axis = 'hksamps_' + device_name

                hkaman = core.AxisManager(core.LabelAxis(alias, [alias]),
                                          core.OffsetAxis(samps_axis, len(time)))
                hkaman.wrap('timestamps', time, [(0, samps_axis)])
                hkaman.wrap(device_name, np.array([data]),
                            [(0, alias), (1, samps_axis)])
                amans.append(hkaman)
        else:
            # if yes, make one aman per device
            device_name = field.split('.')[1]
            device_names.append(device_name)
            device_axis = 'hklabels_' + device_name
            samps_axis = 'hksamps_' + device_name

            hkaman_cos = core.AxisManager(core.LabelAxis(device_axis, aliases),
                                          core.OffsetAxis(samps_axis, len(time)))
            hkaman_cos.wrap('timestamps', time, [(0, samps_axis)])
            hkaman_cos.wrap(device_name, np.array([data[alias] for alias in aliases]), [(0, device_axis), (1, samps_axis)])
            amans.append(hkaman_cos)

    # make an aman of amans
    merged_amans = core.AxisManager()
    [merged_amans.wrap(name, a) for (name, a) in zip(device_names, amans)]

    return merged_amans


def get_hkaman(start, stop, config):
    """
    Wrapper to combine get_grouped_hkdata() and make_hkaman() to output one
    axismanager of HK axismanagers to streamline data analysis.

    Parameters:
        start: Earliest time to search for data. Can be datetime objects,
            unix timestamps, int, floats. Required for get_grouped_hkdata().
        stop: Latest time to search for data. See start parameter above for
            note on time formats. Required for get_grouped_hkdata()
        config (str): Filename of a .yaml file for loading fields and aliases.
            Required for get_grouped_hkdata()

    Notes
    -----

    An example config file looks like:

        data_dir: '/mnt/so1/data/chicago-latrt/hk/'

        field_list:
            'bf_4k' : 'observatory.LSA22HG.feeds.temperatures.Channel_06_T'
            'xy_stage_x': 'observatory.XYWing.feeds.positions.x'
            'xy_stage_y': 'observatory.XYWing.feeds.positions.y'
    """
    data = get_grouped_hkdata(start, stop, config)
    hk_amans = make_hkaman(data)
    return hk_amans


def detcosample_hkamans(hkconfig, detconfig, obs_id):
    """
    Given a detector observation, output an HK axismanager that cosampled
    with detector timestreams.

    Parameters:
        hkconfig (str): The.yaml filename for loading HK fields and aliases.
        detconfig (str): The .yaml config file matching readout id and detector ids.
        obs_id (str): The observation id for a detector UFM. Helps determine
                      the detector start and stop times needed for interpolation.

    Notes
    -----

    An example detconfig file looks like:

        data_prefix : "/mnt/so1/data/chicago-latrt"
        hwp_prefix : "/mnt/so1/shared/site-pipeline/data_pkg/chicago-latrt/hwp/"
        g3tsmurf_db: "/mnt/so1/shared/site-pipeline/data_pkg/chicago-latrt/g3tsmurf.db"
    """
    # load detector timestamps
    SMURF = ls.G3tSmurf.from_configs(detconfig)
    session = SMURF.Session()
    obs = session.query(Observations).filter(Observations.obs_id == obs_id).one()
    det_aman = ls.load_file([f.name for f in obs.files], archive=SMURF)

    grouped_data = get_grouped_hkdata(obs.start, obs.stop, hkconfig)

    amans = []
    device_names = []
    for group in grouped_data:
        data = {}
        aliases = []
        times = {}
        for field in group:
            # arrange data per device for making an aman per hk device
            alias = group[field][0]
            aliases.append(alias)

            time = group[field][1]
            time_info = {alias: time}
            times.update(time_info)

            device_data = group[field][2]

            info = {alias: device_data}
            data.update(info)

        # make an aman per device
        device_name = field.split('.')[1]
        device_names.append(device_name)
        device_axis = 'hklabels_' + device_name

        data_interp = {}
        for alias in times:
            f_hkvals = interp1d(times[alias], data[alias], fill_value='extrapolate')
            hkdata_interp = f_hkvals(det_aman.timestamps)

            info = {alias: hkdata_interp}
            data_interp.update(info)

        hkaman = core.AxisManager(core.LabelAxis(device_axis, aliases),
                                  core.OffsetAxis('detector_timestamps', len(det_aman.timestamps)))
        hkaman.wrap('timestamps', det_aman.timestamps, [(0, 'detector_timestamps')])
        hkaman.wrap(device_name, np.array([data_interp[alias] for alias in aliases]),
                    [(0, device_axis), (1, 'detector_timestamps')])
        amans.append(hkaman)

    merged_detcos_hkamans = core.AxisManager()
    [merged_detcos_hkamans.wrap(name, a) for (name, a) in zip(device_names, amans)]

    return merged_detcos_hkamans
# TODO: insert a progress bar for get_grouped_hkdata
