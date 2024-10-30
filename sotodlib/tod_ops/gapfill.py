import numpy as np
import so3g
from . import pca
import logging

logger = logging.getLogger(__name__)

class Extract:
    """Container for storage of sparse sub-segments of a vector.  This
    ties together a Ranges object and a data array with the same
    length as the total length of the Ranges segments.

    This is useful in cases where a relatively small number of samples
    need to be excised from a signal vector, but stored and perhaps
    restored at a later time.

    In docstrings below a "full data vector" is a 1-d array with
    self.ranges.count elements; the "tracked samples" are a subset
    consisting of self.ranges.mask.sum() elements.

    Attributes:
        ranges: The so3g.proj.Ranges object mapping tracked samples
            into full data vector.
        n_ex: The number of tracked samples.
        data: The data vector containing tracked samples.
    """
    def __init__(self, ranges, init_data=True):
        """Constructor.

        Arguments:
            ranges: a Ranges object; all positive ranges will be
                tracked.
            init_data: array or boolean.  If False, self.data is
                initialized to None.  If True, self.data is
                initialized to zeros.  Otherwise, init_data is stored
                in self.data.  Note this is a reference, not a copy.

        """
        self.ranges = ranges.copy()
        rr = self.ranges.ranges()
        self.n_ex = rr[:, 1].sum() - rr[:, 0].sum()
        if init_data is False or init_data is None:
            self.data = None
        elif init_data is True:
            self.data = np.zeros(self.n_ex)
        else:
            self.data = init_data

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return (self.ranges.count,)

    def offset_iter(self):
        """Returns an iterator for use in expanding / collapsing extracts.
        Each iteration returns a tuple (ex_lo, ex_hi, full_lo,
        full_hi) where ex_lo:ex_hi is a range in extracted data
        vector, and full_lo:full_hi is a range in the full data
        vector.

        """
        offset = 0
        for lo, hi in self.ranges.ranges():
            yield offset, offset + hi - lo, lo, hi
            offset += hi - lo

    def expand(self, fill_value=0):
        """Expands the extract into a full-length data array, filling missing
        bits with fill_value."""
        output = np.zeros(self.ranges.count, self.data.dtype) + fill_value
        for elo, ehi, lo, hi in self.offset_iter():
            output[lo:hi] = self.data[elo:ehi]
        return output

    def swap(self, signal):
        """Swaps the current extract with the tracked samples of full data
        vector signal."""
        to_save = np.empty(self.n_ex, self.data.dtype)
        for elo, ehi, lo, hi in self.offset_iter():
            to_save[elo:ehi] = signal[lo:hi]
            signal[lo:hi] = self.data[elo:ehi]
        self.data[:] = to_save

    def patch(self, signal):
        """Copies the extract into the signal vector.  Untracked samples are
        not modified.

        """
        for elo, ehi, lo, hi in self.offset_iter():
            signal[lo:hi] = self.data[elo:ehi]
        return signal

    def accumulate(self, signal, scale):
        """Adds the extract, scaled by factor scale, into the full length
        signal vector.  Untracked samples are not modified.

        """
        for elo, ehi, lo, hi in self.offset_iter():
            signal[lo:hi] += self.data[elo:ehi] * scale
        return signal


class ExtractMatrix:
    """This class is a simple container for a list (of length n_dets) of
    Extract objects of equal length (n_samps), to provide abstract
    access to data with shape (n_dets, n_samps).  Each child Extract
    can be accessed with [item] indexing.  Most methods have (tod,
    signal=None) signature, and are simple loops that call the method
    of the same name on each child Extract object.

    """
    def __init__(self, items=None):
        """The complete list of Extract objects should be passed in."""
        if items is None:
            items = []
        self.items = items

    def __repr__(self):
        """This repr shows the shape of the full array, along with the
        fraction (as a percentage) of the full space that is actually
        tracked.

        """
        frac = (sum([e.n_ex for e in self.items]) /
                (self.shape[0] * self.shape[1]))
        return 'ExtractMatrix(' + ','.join(map(str, self.shape)) + \
            '@%.1f%%)' % (frac*100)

    def __len__(self):
        return self.shape[0]

    def copy(self):
        return ExtractMatrix([x.copy() for x in self.items])

    @property
    def shape(self):
        if len(self.items) == 0:
            return (0,)
        return (len(self.items),) + self.items[0].shape

    def __getitem__(self, index):
        return self.items[index]

    def expand(self, fill_value=0):
        """Expands the extract into a full-size data array, filling missing
        bits with fill_value."""
        dtype = self.items[0].dtype
        signal = np.zeros(self.shape, dtype) + fill_value
        for o, i in zip(signal, self.items):
            i.patch(o)
        return signal

    def swap(self, tod, signal=None):
        """Swaps the current extract with the tracked samples of full data
        vector signal."""
        if signal is None:
            signal = tod.signal
        for o, i in zip(signal, self.items):
            i.swap(o)
        return signal

    def patch(self, tod, signal=None):
        """Copies the extract into the signal vector.  Untracked samples are
        not modified.

        """
        if signal is None:
            signal = tod.signal
        for o, i in zip(signal, self.items):
            i.patch(o)
        return signal

    def accumulate(self, tod, signal, scale):
        """Adds the extract, scaled by factor scale, into the full length
        signal vector.  Untracked samples are not modified.

        """
        if signal is None:
            signal = tod.signal
        for o, i in zip(signal, self.items):
            i.accumulate(o, scale)
        return signal


def get_gap_fill_single(data, flags, nbuf=10, order=1, swap=False):
    """Computes samples to fill the gaps in data identified by flags.
    Each flagged segment is modeled with a polynomial of the order
    specified, based on up to nbuf points on each side of the segment.

    Arguments:
        data: 1d vector of samples.
        flags: Ranges object consistent with data size.
        nbuf: Maximum number of samples on each side of a flagged span
            to use for model fit.  If the unflagged span between two
            flagged spans is shorter than nbuf, then it is still used
            to anchor one end of the fit.
        order: Maximum order of polynomial model used in
            interpolation.  The actual order may be smaller if
            insufficient data are available to constrain the
            coefficients.  Care should be taken when going higher than
            linear (order=1); this will tend to be unstable unless the
            gaps are much smaller than the anchors used to constrain
            the poly on each end.
        swap: If False, do not modify the input data vector.  If
            True, patch the data with the model.

    Returns:
        An Extract object containing the modeled data.  If swap is
        True, then the input data vector is patched with the model and
        the returned object, instead, contains the samples from data
        that were changed.

    """
    rsegs = (flags.copy().buffer(nbuf) * ~flags)
    rseg_ranges = rsegs.ranges()
    A = np.zeros((order+1, order+1))
    b = np.zeros(order+1)
    t0, y0 = 0, 0
    model = None
    model_i = -1  # Set to trigger update.
    sig_ex = Extract(flags)

    for elo, ehi, lo, hi in sig_ex.offset_iter():
        while (model_i + 1 < len(rseg_ranges) and
               lo > rseg_ranges[model_i+1][0]):
            model = None
            model_i += 1
        if model is None:
            t0, y0 = lo, data[lo-1]
            b[:] = 0
            A[:] = 0
            contrib_count = 0
            for f in [model_i, model_i+1]:
                if f < 0 or f >= len(rseg_ranges):
                    continue
                _lo, _hi = rseg_ranges[f]
                _t = np.arange(_lo, _hi) - t0
                for _j in range(order+1):
                    b[_j] += np.dot(_t**_j, data[_lo:_hi] - y0)
                    for _k in range(_j, order+1):
                        A[_j, _k] += (_t**(_j+_k)).sum()
                        A[_k, _j] = A[_j, _k]
                contrib_count += _hi - _lo
            if contrib_count == 0:
                y0 = 0.
                model = [0.]
            else:
                # Only fit as many terms as you plausibly constrain --
                # 10 data points per term.
                n_keep = 1 + max(0, min(order, contrib_count // 10))
                model = np.linalg.solve(A[:n_keep,:n_keep], b[:n_keep])[::-1]

        t = np.arange(lo, hi) - t0
        sig_ex.data[elo:ehi] = np.polyval(model, t) + y0
    if swap:
        sig_ex.swap(data)
    return sig_ex


def get_gap_fill(tod, nbuf=10, order=1, swap=False, signal=None, flags=None,
                 _method='fast'):
    """See get_gap_fill_single for meaning of arguments not described here.

    Arguments:
        tod: AxisManager with (dets, samps) axes.
        signal: signal to pass to get_gap_fill_single as data
            argument; defaults to tod.signal
        flags: flags to pass to get_gap_fill_single; defaults to
            tod.flags.

    Returns:
        The ExtractMatrix object with per-detector Extracts from
        get_gap_fill_single.

    """
    if signal is None:
        signal = tod.signal
    if flags is None:
        flags = tod.flags
    if _method is None:
        _method = 'fast' if hasattr(so3g, 'get_gap_fill_poly') else 'slow'

    if _method == 'fast':
        sample_counts = [np.dot(r.ranges(), [-1, 1]).sum()
                         for r in flags]
        dest = np.empty(sum(sample_counts), dtype='float32')
        so3g.get_gap_fill_poly(flags, signal, nbuf, order, swap, dest)
        sample_idx = np.cumsum([0] + sample_counts)
        return ExtractMatrix([Extract(r, dest[i:j])
                              for r, i, j in zip(flags, sample_idx[:-1], sample_idx[1:])])
    else:
        return ExtractMatrix([get_gap_fill_single(d, f, order=order, nbuf=nbuf, swap=swap)
                              for d, f in zip(signal, flags)])


def get_gap_model_single(weights, modes, flags):
    """Computes samples to fill gaps in data identified by flags, based on
    weights and modes (such as would be contained in a PCA model).

    Arguments:
        weights: 1-d array of weights (n_mode) to apply to each mode.
        modes: 2-d array of modes (n_mode, n_samps)
        flags: so3g.proj.Ranges object with count == n_samps.

    Returns:
        An Extract object containing the modeled data.

    """
    sig_ex = Extract(flags)
    for w, m in zip(weights, modes):
        for elo, ehi, lo, hi in sig_ex.offset_iter():
            sig_ex.data[elo:ehi] += w * m[lo:hi]
    return sig_ex


def get_gap_model(tod, model, flags=None, weights=None, modes=None):
    """Calls get_gap_model_single on each detector and its corresponding
    model weights.

    Arguments:
        tod: AxisManager with (dets, samps) axes.
        model: AxisManager with (dets, eigen, samps) axes.
        flags: flags to pass to get_gap_model_single; defaults to
            tod.flags.
        weights: array of mode couplings (dets, eigen); defaults to
            model.weights.
        modes: array with time-dependent modes (eigen, samps);
            defaults to model.modes.

    Returns:
        The ExtractMatrix object with per-detector models from
        get_gap_model_single.

    """
    if flags is None:
        flags = tod.flags
    if weights is None:
        weights = model.weights
    if modes is None:
        modes = model.modes
    return ExtractMatrix([get_gap_model_single(w, modes, f)
                          for w, f in zip(weights, flags)])

def get_contaminated_ranges(good_flags, bad_flags):
    """Determine what intervals in good_flags are contaminated (overlap
    with) intervals in bad_flags.  Note this isn't as simple as
    good_flags * bad_flags, because any contiguous region in
    good_flags is considered contaminated if even one sample of it is
    touched by bad_flags.

    Args:
      good_flags (RangesMatrix): The flags to check for contamination.
        Must have shape (dets, samps).
      bad_flags (RangesMatrix): The flags marking bad data, which
        should be considered as a contaminant to good_flags.  Same
        shape as good_flags.

    Returns:
      RangesMatrix with same shape as inputs, indicating the intervals
      from good_flags that overlap at all with some interval of
      bad_flags.

    Example:
      Move contaminated intervals into bad_flags::

        contam = get_contaminated_ranges(source_flags, glitch_flags)
        source_flags *= ~contam
        glitch_flags += contam

    """
    contam = good_flags.zeros_like()
    for r0, r1, rs in zip(good_flags.ranges, bad_flags.ranges,
                          contam.ranges):
        overlap = (r0 * r1).ranges()
        if len(overlap) == 0:
            continue
        # Any interval in r0 that overlaps must be moved
        for i0, i1 in r0.ranges():
            if np.any((i0 <= overlap[:,0]) * (overlap[:,0] < i1)):
                rs.add_interval(int(i0), int(i1))
    return contam


def fill_glitches(aman, nbuf=10, use_pca=False, modes=3, signal=None,
                  glitch_flags=None, in_place=True, wrap=None):
    """
    This function fills pre-computed glitches provided by the caller in
    time-ordered data using either a polynomial (default) or PCA-based
    approach. Wraps the other functions in the ``tod_ops.gapfill`` module.

    Args
    -----
    aman : AxisManager
        AxisManager to fill glitches in
    nbuf : int
        Number of buffer samples to use in polynomial gap filling.
    use_pca : bool
        Whether or not to fill glitches using pca model. Default is False
    modes : int
        Number of modes in the pca to use if pca=True. Default is 3.
    signal : ndarray or None
        Array of data to fill glitches in. If None then uses ``aman.signal``.
        Default is None.
    glitch_flags : RangesMatrix or None
        RangesMatrix containing flags to use for gap filling. If None then
        uses ``aman.flags.glitches``.
    in_place : bool
        If False it makes a copy of signal before gap filling and returns
        the copy.
    wrap : str or None
        If not None, wrap the gap filled data into tod with this name.

    Returns
    -------
    signal : ndarray
        Returns ndarray with gaps filled from input signal.
    """
    # Process Args
    if signal is None:
        if in_place:
            sig = aman.signal
        else:
            sig = np.copy(aman.signal)
    else:
        if in_place:
            sig = signal
        else:
            sig = np.copy(signal)
    
    if glitch_flags is None:
        glitch_flags = aman.flags.glitches

    # Polyfill
    gaps = get_gap_fill(aman, nbuf=nbuf, flags=glitch_flags,
                        signal=np.float32(sig))
    sig = gaps.swap(aman, signal=sig)
    
    #PCA Fill
    if use_pca:
        if modes > aman.dets.count:
            logger.warning(f'modes = {modes} > number of detectors = ' +
                           f'{aman.dets.count}, setting modes = number of ' +
                           'detectors')
            modes = aman.dets.count
        # PCA fill
        mod = pca.get_pca_model(tod=aman, n_modes=modes,
                                signal=sig)
        gfill = get_gap_model(tod=aman, model=mod, flags=glitch_flags)
        sig = gfill.swap(aman, signal=sig)
    
    # Wrap and Return
    if wrap is not None:
        if wrap in aman._assignments:
            aman.move(wrap, None)
        aman.wrap(wrap, sig, [(0, 'dets'), (1, 'samps')])
        
    return sig
