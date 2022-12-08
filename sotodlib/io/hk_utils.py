import numpy as np
import so3g.hk as hk
from so3g.hk import tree
import matplotlib.pyplot as plt
from so3g.hk import load_range
import sotodlib.io.load_smurf as ls
from sotodlib import core
import yaml
import itertools
from itertools import groupby

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def get_grouped_hkdata(start, stop, config):
    """ insert docstrings
    """
    # call load_range()
    print("running load_range()")
    hkdata = load_range(start, stop, config=config)

    # load all fields from config file
    with open(config, 'r') as file:
        hkconfig = yaml.safe_load(file)

    field_dict = hkconfig['field_list']

    # output a dict of field name, alias, timestamps, and data
    # for fields that are online given start, stop, and data from load_range()
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
    sorted_fields = sorted(get_swap_dict(online_fields))
    
    grouped_feeds = []
    key_func = lambda field: field.split('.')[0:4]
    for key, group in itertools.groupby(sorted_fields, key_func):
        grouped_feeds.append(list(group))


    # use grouped_feeds and fields_data to output a grouped list of data per
    # feed, ready to input to an axismanager
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
    """

# TODO: insert a progress bar for get_grouped_hkdata
