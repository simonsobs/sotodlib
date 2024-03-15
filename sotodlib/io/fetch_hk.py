import numpy as np
import so3g
from spt3g import core as g3core
import glob
import os
import logging

hk_logger = logging.getLogger(__name__)
hk_logger.setLevel(logging.INFO)

def fetch_hk(start, stop, data_dir, fields=None):
    hk_data = {}
    bases = []

    start_ctime = start #- 3600
    stop_ctime = stop #+ 3600

    for folder in range(int(start_ctime/1e5), int(stop_ctime/1e5)+1):
        base = data_dir+'/'+str(folder)
        if not os.path.exists(base):
            hk_logger.debug(f'{base} does not exist, skipping')
            continue

        for file in sorted(os.listdir(base)):
            try:
                t = int(file[:-3])
                #print(t)
            except:
                hk_logger.debug(f'{file} does not have the right format, skipping')
                continue
            if t >= start_ctime and t <= stop_ctime+60:
                reader = so3g.G3IndexedReader(base+'/'+file)

                while True:
                    frames = reader.Process(None)
                    if not frames:
                        break

                    for frame in frames:
                        if 'address' in frame:
                            for v in frame['blocks']:
                                for k in v.keys():
                                    field = '.'.join([frame['address'], k])

                                    if fields is None or field in fields:
                                        key = field.split('.')[-1]
                                        if k == key:
                                            data = [[t.time / g3core.G3Units.s for t in v.times], v[k]]
                                            hk_data.setdefault(field, ([], []))
                                            hk_data[field] = (
                                                np.concatenate([all_data[field][0], data[0]]),
                                                np.concatenate([all_data[field][1], data[1]])
                                            )
    return hk_data
