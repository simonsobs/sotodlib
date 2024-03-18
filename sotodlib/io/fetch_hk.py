import so3g
from spt3g import core as g3core


def fetch_hk(path, fields=None):
    """
    Fetches housekeeping (HK) data from a given path.

    Args:
        path (str): Path to the HK .g3 data file.
        fields (list, optional): List of specific fields. If None, fetches
                                 all fields. Default is None.

    Returns:

        Dictionary with structure::

        {
            field[i] : (time[i], data[i])
        }

        Same output format as `load_range`. Masked to only have data from
        start and stop of .g3 file provided in path argument.

    """
    hk_data = {}
    reader = so3g.G3IndexedReader(path)

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
                                hk_data[field][0].extend(data[0])
                                hk_data[field][1].extend(data[1])

    return hk_data
