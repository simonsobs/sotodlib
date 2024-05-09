import numpy as np
import yaml
import itertools
import logging

from sotodlib.io import hkdb
from sotodlib import core

from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def _get_swap_dict(d):
    """
    Swap dictionary keys and values; useful for organizing fields in
    config file by device name instead of alias in get_grouped_hkdata().
    """
    return {v: k for k, v in d.items()}


def _group_feeds(fields, alias_exists=False):
    """ Sort and group the HK feeds that are output from hkdb.load_hk()
    by device.

    Parameters:
        fields (dict) : dict of names of fields, and any aliases if provided
        alias_exists (bool) : indicates whether the field dictionary includes
            aliases for field names. If True, the dictionary is inverted to
            group by device, as grouping requires exact field names. If False,
            no aliases were provided in the fields dictionary. Default is False.
  
      Returns:
        grouped_feeds (list): : sorted and grouped list of fields by device name,
            a requirement for outputting an axis manager per HK device.

    """
    if alias_exists:
        sorted_fields = sorted(_get_swap_dict(fields))
    else:
        sorted_fields = sorted(fields)

    grouped_feeds = []
    key_func = lambda field: field.split('.')[0]
    for key, group in itertools.groupby(sorted_fields, key_func):
        grouped_feeds.append(list(group))

    return grouped_feeds
    

def _group_data(hkdata, alias_map=None, fields_only=False):
    """ Sort hkdata based off of _group_feeds output.

    Parameters:
        hkdata (dict) : data dict from hkdb.load_hk()
        alias_map (dict or None) :  data dictionary mapping field aliases to
            their corresponding field names. If provided, indicates aliases exist
            for each field, requiring the use of the hkdb config file. If None,
            there are no aliases, hkdb config file not needed. Default is None.
        fields_only (bool) : if True, field names are provided as args to 
            load_hk. Default is False.
        config_exists (bool) : if True, config file is used to extract the fields
            needed for hk loading. This also means that the fields have an alias. 

    Returns:
        get_grouped_data (list) : grouped list of data per HK feed/device;
        useful for turning into a grouped set of axismanagers

    """
    hknames = list(hkdata.keys())
    if fields_only:
        grouped_feeds = _group_feeds(fields=hknames, alias_exists=False)

    if alias_map:
        grouped_feeds = _group_feeds(fields=alias_map, alias_exists=True)

        fields_data = {}
        for alias, field_name in alias_map.items():
            time = hkdata[field_name][0]
            data = hkdata[field_name][1]
            fields_data[field_name] = [time, data, alias]
            
    grouped_data = []
    for group in grouped_feeds:
        device_data = {}
        for field in group:
            if fields_only:
                field_dict = {field: hkdata[field]}
                device_data.update(field_dict)
            if alias_map:
                field_dict = {field: fields_data[field]}
                device_data.update(field_dict)

        grouped_data.append(device_data)

    return grouped_data


def sort_hkdata(config, start, end, fields=None, use_config=False):
    """
    Gets HK data dictionary using hkdb.load_hk() and groups data by HK device.

    Parameters:
        start: earliest time to search for data. Can be datetime objects,
            unix timestamps, int, floats.
        end: latest time to search for data. See start parameter above for
            note on time formats.
        config (str): name of hkdb config .yaml file.
        use_config (bool):  indicates whether to use the config file to 
            group HK data by including fields' aliases. If True, aliases
            are added; if False, only field names are used for grouping.
            Default is False.

    Returns:
        get_grouped_data (list): list of HK data dictionaries organized by
            HK feed/device.

    Notes
    -----

    An example config file looks like:

        hk_root: /so/data/satp1/hk/
        hk_db: satp1_hk.db
        aliases:
            fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T
            bd-dr-port: cryo-ls240-lsa2619.temperatures.Channel_2_T
            urh: cryo-ls240-lsa2619.temperatures.Channel_1_T

    """
    cfg = hkdb.HkConfig.from_yaml(config)
    
    if use_config:
        ls = hkdb.LoadSpec(cfg=cfg, fields=cfg.aliases.keys(), start=start, end=end)
        ld = hkdb.load_hk(ls, show_pb=True)
        hkdata = ld.data
        with open(config, 'r') as file:
            hkconfig = yaml.safe_load(file)

        alias_map = hkconfig['aliases']

        grouped_data = _group_data(hkdata=hkdata, alias_map=alias_map,
                                  fields_only=False)

        return grouped_data
    elif fields and not use_config:
        ls = hkdb.LoadSpec(cfg=cfg, start=start, end=end, fields=fields)
        ld = hkdb.load_hk(ls, show_pb=True)
        hkdata = ld.data

        grouped_data = _group_data(hkdata=hkdata, alias_map=None, fields_only=True)

        return grouped_data
        

def _hk_amans(group, alias_exists=False, hk_cosampled=False):
    """ Generates HK axis managers based on the provided HK data's
        sampling characteristics.
    
        Parameters:
            group (dictionary): dictionary of HK data for one device/feed.
            alias_exists (bool): if True, `group` contains alias information.
                Default is False.
            hk_cosampled (bool): if True, HK device in `group` has cosampled data.
                Default is False. 
        
        Returns:
            amans (list): list of axis managers for a non-cosampled HK device,
                or list of one axis manager for a cosampled HK device.
            device_names (list): either field names of each HK field or their aliases.

    """
    # for non cosampled hk data
    amans = []
    device_names = []
    if not hk_cosampled: 
        for field in group:
            time = group[field][0]
            data = group[field][1]
            if alias_exists is True:
                device_name = group[field][2]
            else:
                device_name = field.split('.')[0]
    
            device_names.append(device_name)
    
            device_axis = 'hklabels_' + device_name
            samps_axis = 'hksamps_' + device_name

            hkaman = core.AxisManager(core.LabelAxis(device_axis, [field]),
                                      core.OffsetAxis(samps_axis, len(time)))
            hkaman.wrap('timestamps', time, [(0, samps_axis)])
            hkaman.wrap(f'{device_name}_data', np.array([data]),
                        [(0, device_axis), (1, samps_axis)])
            amans.append(hkaman)
    
    # for cosampled hk data
    else:
        aliases = [group[field][2] if alias_exists else field for field in group]
        key = list(group.keys())[0]
        time = group[key][0]
        data = {}
        
        for field, value in group.items():
            device_name = field.split('.')[0]
            if len(value) == 2:
                times, values = value
                alias = field
            elif len(value) == 3:
                times, values, alias = value
            
            data[alias] = values
            
            if device_name not in device_names:
                device_names.append(device_name)
            else:
                continue
        
        device_axis = 'hklabels_' + device_name
        samps_axis = 'hksamps_' + device_name
        hkaman_cos = core.AxisManager(core.LabelAxis(device_axis, aliases),
                                      core.OffsetAxis(samps_axis, len(time)))
        hkaman_cos.wrap('timestamps', time, [(0, samps_axis)])
        hkaman_cos.wrap(f'{device_name}_data', np.array([data[alias] for alias in aliases]),
                        [(0, device_axis), (1, samps_axis)])
        amans.append(hkaman_cos)

    return amans, device_names


def _hkdet_amans(group, det_aman, alias_exists=False):
    """ Generates HK axis managers that are cosampled to detector data.
        
        Parameters:
            group (dictionary): dictionary of HK data for one device/feed.
            alias_exists (bool): if True, `group` contains alias information.
                Default is False.
            det_aman (AxisManager): detector data AxisManager  
        
        Returns:
            amans (list): list of axis manager for HK data that are cosampled to detector timestreams.
            device_names (list): either field names of each HK field or their aliases.
            
    """
    device_names = []
    data_interp = {}
    aliases = [group[f][2] if alias_exists else f for f in group]
    amans = []
    for field, value in group.items():
        device_name = field.split('.')[0]
        if len(value) == 2:
            times, values = value
            alias = field 
        elif len(value) == 3:
            times, values, alias = value
    
        f_hkvals = interp1d(times, values, fill_value='extrapolate')
        hkdata_interp = f_hkvals(det_aman.timestamps)

        data_interp[alias] = hkdata_interp
        
        if device_name not in device_names:
            device_names.append(device_name)
        else:
            continue
    
    
    device_axis = 'hklabels_' + device_name

    hkaman = core.AxisManager(core.LabelAxis(device_axis, aliases),
                              core.OffsetAxis('detector_timestamps',
                                              len(det_aman.timestamps)))
    hkaman.wrap('timestamps', det_aman.timestamps, [(0, 'detector_timestamps')])
    hkaman.wrap(f'{device_name}_data', np.array([data_interp[alias] for alias in aliases]),
                [(0, device_axis), (1, 'detector_timestamps')])
    amans.append(hkaman)

    return amans, device_names               


def make_hkaman(grouped_data, alias_exists=False, det_cosampled=False, det_aman=None):
    """
    Takes data from get_grouped_hkdata(), tests whether feed/device is
    cosampled, outputs axismanager(s) for either case.

    Parameters:
        grouped_data (list): grouped list of data per HK feed/device; output
            from get_grouped_hkdata()
        det_cosampled (bool): decides how to organize HK data into an AxisManager
            depending on user preference. Default set to False means HK data is
            output to an AxisManager per cosampled device, as well as 1 AxisManager
            per 'channel' for a non-cosampled device. If set to True, all HK data
            will be interpolated against detector timestreams.
        det_aman (NoneType): AxisManager of detector data from load_smurf().
            Only required if det_cosampled is True.

    Returns:
        merged_amans (AxisManager): AxisManager of HK AxisManagers. Returns one
            ultimate AxisManager containing all individual HK axismanagers generated.
    
    """
    ams = []
    devices = []
    for group in grouped_data:
        aliases = [group[field][2] if alias_exists else field for field in group]
        # only want HK data, checking for cases where HK data is cosampled
        if det_cosampled is False:
            try:
                if len(group) == 1:
                    raise AssertionError("Only one field in the group, treating as not cosampled")
                # check for cosampled HK devices
                keys = list(group.keys())
                t = group[keys[0]][0]
                assert np.all([len(group[f][0]) == len(t) for f in group]) # group[f][-1]
                # TODO: diff of times, if less than say 10%??, treat as cosampled
            except AssertionError:
                # if HK device not cosampled, make 1 aman per 'channel'
                amans, device_names = _hk_amans(group=group, alias_exists=alias_exists, hk_cosampled=False)
                devices.extend(device_names)
                ams.extend(amans)
            else:
                # if HK device cosampled, make 1 aman for entire device
                 amans, device_names = _hk_amans(group=group, alias_exists=alias_exists, hk_cosampled=True)
                 devices.extend(device_names)
                 ams.extend(amans)

        else:
            assert det_aman is not None, "Did not input a detector axis manager as an arg"
            amans, device_names = _hkdet_amans(group=group, alias_exists=alias_exists, det_aman=det_aman)
            devices.extend(device_names)
            ams.extend(amans)

    # make an aman of amans
    merged_amans = core.AxisManager()
    [merged_amans.wrap(name, a) for (name, a) in zip(devices, ams)]

    return merged_amans


def get_hkaman(start, end, config, fields=None):
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

        hk_root: /so/data/satp1/hk/
        hk_db: satp1_hk.db
        aliases:
            fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T
            bd-dr-port: cryo-ls240-lsa2619.temperatures.Channel_2_T
            urh: cryo-ls240-lsa2619.temperatures.Channel_1_T
    
    """
    if fields:
        data = sort_hkdata(config=config, start=start, end=end, fields=fields, use_config=False)
        hkamans = make_hkaman(grouped_data=data, alias_exists=False, det_cosampled=False)
    else:
        data = sort_hkdata(config=config, start=start, end=end, fields=None, use_config=True)
        hkamans = make_hkaman(grouped_data=data, alias_exists=True, det_cosampled=False)
    return hkamans


def get_detcosamp_hkaman(det_aman, config, fields=None):
    """
    Wrapper to combine get_grouped_hkdata() and make_hkaman() to output one
    axismanager of HK axismanagers that are cosampled to detector timestreams.

    Parameters:
        config (str): Filename of a .yaml file for loading HK fields and aliases.
            Required for get_grouped_hkdata().
        det_aman (AxisManager): detector data AxisManager.

    Notes
    -----

    An example config file looks like:

        hk_root: /so/data/satp1/hk/
        hk_db: satp1_hk.db
        aliases:
            fp_temp: cryo-ls372-lsa21yc.temperatures.Channel_02_T
            bd-dr-port: cryo-ls240-lsa2619.temperatures.Channel_2_T
            urh: cryo-ls240-lsa2619.temperatures.Channel_1_T
    
    """
    start = float(det_aman.timestamps[0])
    end = float(det_aman.timestamps[-1])

    if fields:
        data = sort_hkdata(config=config, start=start, end=end, fields=fields, use_config=False)
        amans = make_hkaman(grouped_data=data, alias_exists=True,
                            det_cosampled=True, det_aman=det_aman)
        return amans
    else:
        data = sort_hkdata(config=config, start=start, end=end, fields=None, use_config=True)
        amans = make_hkaman(grouped_data=data, det_aman=det_aman,
                            alias_exists=True, det_cosampled=True)
        return amans

# TODO: fix the docstrings, add log statements, change the arg wording of config_exists cus it always exists now
