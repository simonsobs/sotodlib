import numpy as np
import so3g
from spt3g import core as g3core
import glob
import os
import logging

hk_logger = logging.getLogger(__name__)
hk_logger.setLevel(logging.INFO)

def fetch_hk(start, stop, data_dir, fields=None, folder_patterns=None):
    hk_data = {}
    bases = []

    start_ctime = start #- 3600
    stop_ctime = stop #+ 3600
    
    if folder_patterns is None:
        folder_patterns = ['{folder}', 'hk_{folder}_*']
    for folder in range( int(start_ctime/1e5), int(stop_ctime/1e5)+1):
        bases = []
        for pattern in folder_patterns:
            extended_pattern = pattern.format(folder=folder)

            base = glob.glob(os.path.join(data_dir, extended_pattern))
            bases.extend(base)
        
        if len(bases) > 1:
            bases.sort
            base = bases[0]
            hk_logger.warn(f"Multiple base folders were found for {folder}. The first one, alphabetically, is selected: {base}")
        elif len(bases) == 1:
            base = bases[0]
        elif len(bases) == 0:
            hk_logger.debug(f"No base folder found for {folder}, skipping")
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
                                                np.concatenate([hk_data[field][0], data[0]]),
                                                np.concatenate([hk_data[field][1], data[1]])
                                            )
    return hk_data
