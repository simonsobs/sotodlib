import numpy as np
import csv
import os


def get_wafer_info(array_name, base_config={}, array_config={}, raw=False,
                   include_no_match=True):
    """Assemble the "wafer info" for a UFM.  This involves reading a few
    csv files and tracing the connections between them.

    Args:
      array_name (str): e.g. 'Mv7'.  Capitalization is important, as
        this is used in filenames.
      base_config (dict): Main config dict.  See notes.
      array_config (dict): Used to override names of specific input
        files (see below).  Normally not necessary; see below.
      raw (bool): If True, return all the data dictionaries and the
        raw data
      include_no_match (bool): If True, include a row with det_id
        "NO_MATCH" to represent resonators with failed association.

    Returns:
       A list where each element contains the data for a single
       readout element (anything that might be associated to a
       resonator; including optical detectors, dark detectors, free
       squids, etc.).

    Notes:
      There are 4 files contributing to the construction of the wafer
      info.  Each may be overridden in the array_config dict; the
      values can be relative to ``base_config['array_info_dir']`` or
      absolute.  See source code for default filenames.

      The base_config dict items used here are:
        - ``array_info_dir``: path to where the wiring files can be
          found.
        - ``bandpass_remap``: a dict specifying how to translate
          bandpass center values provided in the copper_layout into
          values that should be assigned to the dets (and used in
          det_id).  E.g. {90: 220, 150: 280}.

    """
    # Figure out the filenames and read the csv.
    base_dir = base_config.get('array_info_dir', './')
    filenames = [
        os.path.join(base_dir, array_config.get(key, default)) for key, default in [
            ('umux_map_file', 'umux_32_map.csv'),
            ('copper_map_file', 'copper_map_corrected.csv'),
            ('copper_layout_file', 'copper_map_corrected_pixel_num_layout.csv'),
            ('mux_pos_to_band_file', f'{array_name}_mux_pos_num_to_mux_band_num.csv'),
        ]
    ]
    umux_map, copper_map, copper_layout, mux_pos_to_band = \
        map(_read_csv_cols, filenames)

    # Reformat, add some columns.

    ## Sometimes is_north is TRUE/FALSE instead of True/False
    mux_pos_to_band['is_north'][:] = [x.capitalize() for x in mux_pos_to_band['is_north']]

    ## The "pin in copper_map is hidden in SQUID_PIN string.
    copper_map['_pin'] = np.array(
        [k.split('_')[3] for k in copper_map['SQUID_PIN']])

    # Remap the bandpass field, e.g. 150 -> 280 or whatever.
    bandpass_remap = base_config.get('bandpass_remap')
    if bandpass_remap is not None:
        # Do it in two steps to support swapping, shrug.
        masks = []
        for k, v in bandpass_remap.items():
            masks.append((str(v), copper_layout['bandpass'] == str(k)))
        for v, s in masks:
            copper_layout['bandpass'][s] = v

    # The umux_map contains all muxable frequencies from 4-8 GHz.  But
    # SO uses 2 chips, and only the 4-6 GHz band from each.  Therefore
    # reformulate umux_map with a new column 'is_north', and repeat
    # the 4-6 GHz band.
    s = np.array(list(map(int, umux_map['mux_band']))) < 14
    umux_map = {k: np.hstack((v[s], v[s])) for k, v in umux_map.items()}
    umux_map['is_north'] = np.hstack((np.zeros(s.sum(), bool),
                                      np.ones(s.sum(), bool))).astype('str')

    # The four data tables are represented as dicts of arrays of
    # strings.  Strings are ok for now since it's mostly straight
    # table joins on equal values. Re-typing is dealt with in final
    # processing.

    # The umux_map contains all the muxable (is_north, freq) pairs.
    # So loop over entries in that, and find what is attached there.
    idx = []
    data_rows = []
    for umux_i in range(len(umux_map['mux_band'])):
        # Matching the resonator to other tables requires the mux_band
        # (0-27) and bond_pad.  Convert the mux_band to the pair
        # (mod_band, is_highband) to lookup in the mux_pos_to_band
        # table.
        this_umux = _row_from_cols(umux_map, umux_i)
        this_pad = this_umux['bond_pad']  # Note this is a str still.

        # Now get the index into the mux_pos table.
        try:
            mux_map_i = _find_in_cols(mux_pos_to_band,
                                      {'mux_band_num': str(this_umux['mux_band']),
                                       'is_north': this_umux['is_north']})
        except ValueError:
            # This is the only way we should lose resonators.
            continue

        # Use the mux_pos_num to look up detector details.
        this_ptb = _row_from_cols(mux_pos_to_band, mux_map_i)
        this_ptb['array_name'] = array_name  # seems appropriate
        mux_pos_num = this_ptb['mux_pos_num']

        copper_i, layout_i = None, None
        this_copper, this_layout = {}, {}

        try:
            copper_i = _find_in_cols(copper_map,
                                     {'Mux chip position': mux_pos_num,
                                      '_pin': this_pad})
        except:
            pass

        if copper_i is not None:
            layout_i = _find_in_cols(copper_layout,
                                     {'mux_layout_position': mux_pos_num,
                                      'bond_pad': this_pad})
            this_copper = _row_from_cols(copper_map, copper_i)
            this_layout = _row_from_cols(copper_layout, layout_i)

        idx.append((umux_i, mux_map_i, copper_i, layout_i))

        data_rows.append((
            this_umux,
            this_ptb,
            this_copper,
            this_layout,
        ))

    if raw:
        return {
            'filenames': filenames,
            'table_names': ['umux', 'pos_to_band', 'copper', 'layout'],
            'tables': [umux_map, mux_pos_to_band, copper_map, copper_layout],
            'indices': idx,
            'rows': data_rows,
        }

    # Process all rows into final format
    output = []
    for row in data_rows:
        output.append(_process_row(row))

    if include_no_match:
        output.append(_process_row(
            [{'det_id': 'NO_MATCH', 'array_name': array_name},{},{},{}]))

    return output


def _process_row(row):
    # Organize and reformat all the data loaded for a resonator /
    # readout element into a wafer_info entry dict.
    #
    # To get a "NO_MATCH" row pass in
    #   ({'det_id': 'NO_MATCH', 'array_name': ...}, {}, {}, {}).

    umux, pos_to_band, copper, layout = row
    def _get(key, cast=lambda x: x, default=None):
        for src in [umux, copper, layout, pos_to_band]:
            if key in src:
                v = src[key]
                return cast(v)
        return default

    # Construct the output; this sets the ordering of the keys; mark
    # ones we need to post-compute as PLACEHOLDER.
    output = {ko: _get(ki, *args) for ko, ki, *args in [
        ('det_id', 'det_id'),
        ('array', 'array_name'),
        ('bond_pad', 'bond_pad', int, -1),
        ('mux_band', 'mux_band', int, -1),
        ('mux_channel', 'mux_channel', int, -1),
        ('mux_subband', 'mux_subband', str, 'NC'),
        ('mux_position', 'mux_pos_num', int, -1),
        ('design_freq_mhz', 'freq_mhz', float, np.nan),
        ('bias_line', 'Bias line', int, -1),
        ('pol', 'pol', str, 'X'),
        ('bandpass', 'bandpass', str, 'NC'),
        ('det_row', 'DTPixelrow', int, -1),
        ('det_col', 'DTPixelcolumn', int, -1),
        ('rhombus', 'rhomb', str, 'X'),
        ('type', 'PLACEHOLDER'),
        ('x', 'DTPixelxcenter', float, np.nan),
        ('y', 'DTPixelycenter', float, np.nan),
        ('angle', 'DTActualangle', float, np.nan),
        ('crossover', 'DTPadlabel', str, 'X'),
        ('coax', 'is_north', str, None),
    ]}

    if output['bandpass'] != 'NC':
        output['bandpass'] = 'f%03i' % int(output['bandpass'])

    if output['det_id'] == 'NO_MATCH':
        label = 'NO_MATCH'
        det_type = 'NC'
    elif output['det_row'] < 0 or output['pol'] == 'D' or output['bandpass'] == 'NC':
        b, det_type = '%02i' % output['bond_pad'], 'SLOT'
        if b == '64':
            det_type = 'SQID'
        elif b == '-1':
            b, det_type = 'NC', 'BARE'
        elif output['pol'] == 'D':
            det_type = 'DARK'
        elif output['det_row'] == -1:
            det_type = 'UNRT'
        else:
            # SLOT
            output['angle'] = np.nan
            output['crossover'] = 'X'
        mux_pos = '%02i' % _get('mux_pos_num', int)
        label = f'{_get("array_name")}_{det_type}_Mp{mux_pos}b{b}D'
    else:
        det_type = 'OPTC'
        label = '{array}_{bandpass}_{rhombus}r{det_row:02d}c{det_col:02d}{pol}'.format(**output)

    if det_type == 'DARK':
        # The position is taken from 'x' and 'y', not Pixel*center.
        output['x'] = _get('x', float)
        output['y'] = _get('y', float)
        output['angle'] = np.nan

    output.update({
        'det_id': label,
        'type': det_type,
        'coax': {'True': 'N', 'False': 'S'}.get(output['coax'], 'X'),
        'x': output['x'] / 1000.,
        'y': output['y'] / 1000.,
        'array': output['array'].lower(),
        'crossover': output['crossover'][0],
    })

    return output


# CSV -> dict of columns.

def _read_csv_cols(filename):
    rows = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)#, dialect='memberdb')
        for row in reader:
            rows.append(row)
    hdr = rows.pop(0)
    return {h: np.array(c) for h, c in zip(hdr, list(zip(*rows)))}


def _find_in_cols(cols, search_dict):
    idx = [tuple(r) for r in zip(*[cols[k] for k in search_dict])]
    return idx.index(tuple(search_dict.values()))


def _row_from_cols(cols, idx):
    return {k: v[idx] for k, v in cols.items()}
