import numpy as np


def get_apodize_window_for_ends(aman, apodize_samps=1600, apo_type='C1'):
    """
    Generate an apodization window using a cosine taper at the beginning and end.

    Args:
        aman: An axismanager
        apodize_samps (int): Number of samples to apply the cosine taper to at each end.
        apo_type (str): Type of apodization window applied to the edges. Options are:

            - ``'C1'``: Standard cosine (Hann) taper, i.e. a half-cosine that
              goes smoothly from 1 to 0 as ``0.5 * (1 + cos(x))`` over the
              apodization region. This is the default.
            - ``'old_default'``: Legacy quarter-cosine taper, ``cos(x)`` over
              ``[0, pi/2]``. Retained for backward compatibility.

    Returns:
        numpy.ndarray: An array representing the apodization window.
    """
    if apo_type == 'C1':
        cosedge = 0.5 * (np.cos(np.linspace(0, np.pi, apodize_samps)) + 1)
    elif apo_type == 'old_default':
        cosedge = np.cos(np.linspace(0, np.pi/2, apodize_samps))

    w = np.ones(aman.samps.count)
    w[-apodize_samps:] = cosedge
    w[:apodize_samps] = np.flip(cosedge)
    return w


def get_apodize_window_from_flags(aman, flags, apodize_samps=200, apo_type='C1'):
    """
    Generate an apodization window based on flag values. Apply cosine tapering every 
    continuous portion of data between flagged region.

    Args:
        aman: An axismanager
        flags (str or RangesMatrix or Ranges): Flags of mask in RangesMatrix/Ranges. If provided by 
            a string, 'aman.flags[flags]' is used for the flags.
        apodize_samps (int): Number of samples to apply the cosine taper.
        apo_type (str): Type of apodization window applied to the edges. Options are:

            - ``'C1'``: Standard cosine (Hann) taper, i.e. a half-cosine that
              goes smoothly from 1 to 0 as ``0.5 * (1 + cos(x))`` over the
              apodization region. This is the default.
            - ``'old_default'``: Legacy quarter-cosine taper, ``cos(x)`` over
              ``[0, pi/2]``. Retained for backward compatibility.

    Returns:
        numpy.ndarray: An array representing the apodization window.
    """
    if isinstance(flags, str):
        flags = aman.flags[flags]
    flags_mask = flags.mask()

    if flags_mask.ndim == 1:
        flag_is_1d = True
    else:
        all_columns_same = np.all(np.all(flags_mask == flags_mask[0, :], axis=0))
        if all_columns_same:
            flag_is_1d = True
            flags_mask = flags_mask[0]
        else:
            flag_is_1d = False

    if apo_type == 'C1':
        cosedge = 0.5 * (np.cos(np.linspace(0, np.pi, apodize_samps)) + 1)
    elif apo_type == 'old_default':
        cosedge = np.cos(np.linspace(0, np.pi/2, apodize_samps))

    apodizer = ~flags_mask
    apodizer = apodizer.astype(float)

    if flag_is_1d:
        idxes_left = np.where(np.diff(apodizer) == -1)[0]
        idxes_right = np.where(np.diff(apodizer) == 1)[0]

        for _left in idxes_left:
            _apo_idxes_left = (_left-apodize_samps+1, _left+1)
            if _apo_idxes_left[0] < 0:
                apodizer[:_apo_idxes_left[1]] = 0
            else:
                apodizer[_apo_idxes_left[0]:_apo_idxes_left[1]] *= cosedge

        for _right in idxes_right:
            _apo_idxes_right = (_right-1, _right+apodize_samps-1)
            if _apo_idxes_right[1] > aman.samps.count - 1:
                apodizer[_apo_idxes_right[0]:] = 0
            else:
                apodizer[_apo_idxes_right[0]:_apo_idxes_right[1]] *= np.flip(cosedge)
    else:
        for di in range(aman.dets.count):
            idxes_left = np.where(np.diff(apodizer[di]) == -1)[0]
            idxes_right = np.where(np.diff(apodizer[di]) == 1)[0]

            for _left in idxes_left:
                _apo_idxes_left = (_left-apodize_samps+1, _left+1)
                if _apo_idxes_left[0] < 0:
                    apodizer[di][:_apo_idxes_left[1]] = 0
                else:
                    apodizer[di][_apo_idxes_left[0]:_apo_idxes_left[1]] *= cosedge

            for _right in idxes_right:
                _apo_idxes_right = (_right-1, _right+apodize_samps-1)
                if _apo_idxes_right[1] > aman.samps.count - 1:
                    apodizer[di][_apo_idxes_right[0]:] = 0
                else:
                    apodizer[di][_apo_idxes_right[0]:_apo_idxes_right[1]] *= np.flip(cosedge)

    return apodizer


def apodize_cosine(aman, signal_name='signal', apodize_samps=1600, in_place=True,
                   apo_axis='apodized', window=None, flags=None, apo_type='C1'):
    """
    Function to smoothly filter the timestream to 0's on the ends with a
    cosine function. If window is provided, multiply the window function to
    aman[signal_name]. If flags is provided, generate an apodization window
    based on flag values instead of ends of timestream.

    Args:
        signal_name (str): Axis to apodize
        apodize_samps (int): Number of samples on tod ends to apodize.
        in_place (bool): writes over signal with apodized version
        apo_axis (str): Axis to store the apodized signal if not in place.
        window (numpy.ndarray): Precomputed apodization window.
        flags (str or RangesMatrix or Ranges): flag value to compute apodization window.
        apo_type (str): Type of apodization window applied to the edges. Options are:

            - ``'C1'``: Standard cosine (Hann) taper, i.e. a half-cosine that
              goes smoothly from 1 to 0 as ``0.5 * (1 + cos(x))`` over the
              apodization region. This is the default.
            - ``'old_default'``: Legacy quarter-cosine taper, ``cos(x)`` over
              ``[0, pi/2]``. Retained for backward compatibility.

    """
    if window is None:
        if flags is not None:
            window = get_apodize_window_from_flags(aman, flags, apodize_samps, apo_type=apo_type)
        else:
            window = get_apodize_window_for_ends(aman, apodize_samps, apo_type=apo_type)

    if in_place:
        aman[signal_name] *= window
    else:
        aman.wrap_new(apo_axis, dtype='float32', shape=('dets', 'samps'))
        aman[apo_axis] = aman[signal_name] * window
    return
