===========
AxisManager
===========

The AxisManager is a container class for numpy arrays and similar
structures, that tracks relationships between axes.  The object can be
made aware, for example, that the 2nd axis of an array called "data"
is tied to the 1st axis of an array called "azimuth"; slices /
sub-selection operations can be performed on a named
on that axis and the entire object will be modified self-consistently.

The data model here is similar to what is provided by the `xarray`_
library.  However, AxisManager makes it possible to store objects
other than numpy ndarrays, if the classes expose a suitable interface.

.. _`xarray`: http://xarray.pydata.org/en/stable/

------------------------
Internal data structures
------------------------

If you're debugging or trying to understand how AxisManager works,
know that the entire state is stored in the following private data
members:

    ``_axes``
        An odict that maps axis name to an Axis-type object (such as
        an instance of LabelAxis or IndexAxis).

    ``_assignments``
        An odict that maps field name to a tuple, where each entry in
        the tuple gives the name of the axis to which that dimension
        of the underlying data is associated (or None, for a dimension
        that is unassociated).  For example an array with shape (100,
        2, 20000) might have an axis assignment of ``('dets', None,
        'samps')``.

    ``_fields``
        An odict that maps field name to the data object.

Consistency of the internal state requires:

- The key under which an Axis-type object in ``_axes`` must match that
  object's ``.name`` attribute.  (I.e. ``_axes[k].name == k``.)
- The ``_assignments`` and ``_fields`` odicts should have the same
  keys, and the dimensionality must agree.
  (I.e. ``len(_assignments[k]) == len(_fields[k].shape)``.)

--------
Tutorial
--------

Here we demonstrate some usage patterns for AxisManager.  For these to
work, you need these imports::

    from sotodlib import core
    import numpy as np

Suppose you have an array of detector readings. It’s 2d, with the first
axis representing some particular detectors and the second axis
representing a time index.

.. code-block:: python

    dets = ['det10', 'det11', 'det12']
    tod = np.zeros((3, 10000)) + [[10],[11],[12]]

AxisManager is a container that can hold numpy arrays (and also it can
hold other AxisManagers, and some other stuff too).

.. code-block:: python

    dset = core.AxisManager().wrap('tod', tod)

AxisManagers can also hold scalars, a value is considered scalar if
``'np.isscalar'`` thinks it is a scalar or if it is ``'None'``.

.. code-block:: python

    dset = dset.wrap('scalar', 1.0)

Inspecting::

    >>> print(dset)
    AxisManager(tod[3,10000], scalar)
    >>> print(dset.tod)
    [[10. 10. 10. ... 10. 10. 10.]
     [11. 11. 11. ... 11. 11. 11.]
     [12. 12. 12. ... 12. 12. 12.]]
    >>> print(dset.scalar)
    1.0


The value that AxisManager adds is an ability to relate an axis in one
child array to an axis in another child. This time, when we add
``'tod'``, we describe the two dimensions of that array with LabelAxis
and IndexAxis objects.

.. code:: python

    dset = core.AxisManager().wrap('tod', tod, [(0, core.LabelAxis('dets', dets)),
                                                (1, core.IndexAxis('samps'))])

Inspecting::

    >>> print(dset)
    AxisManager(tod[dets,samps], dets:LabelAxis(3), samps:IndexAxis(10000))

Now if we add other arrays, we can assign their axes to these existing
ones::

    hwp_angle = np.arange(tod.shape[1]) * 2. / 400 % 360.
    dset.wrap('hwp_angle', hwp_angle, [(0, 'samps')])

The output of the ``wrap`` call is::

    AxisManager(tod[dets,samps], hwp_angle[samps], dets:LabelAxis(3),
      samps:IndexAxis(10000))

To create new arrays in the AxisManager, and have certain dimensions
automatically matched to a particular named axis, use the ``wrap_new``
method::

    events = dset.wrap_new('event_count', shape=('samps', 3), dtype='int')

This object returned by this call is a numpy array of ints with shape
(10000, 3).  It is wrapped into dset under the name ``'event_count'``.
Its first axis is tied to the ``'dets'`` axis.

We can also embed related AxisManagers within the existing one, to
establish a hierarchical structure::

    boresight = core.AxisManager(core.IndexAxis('samps'))
    for k in ['el', 'az', 'roll']:
        boresight.wrap(k, np.zeros(tod.shape[1]), [(0,'samps')])

    dset.wrap('boresight', boresight)

The output of the ``wrap`` cal should be::

    AxisManager(tod[dets,samps], hwp_angle[samps], boresight*[samps],
      dets:LabelAxis(3), samps:IndexAxis(10000))

Note the boresight entry is marked with a ``*``, indicating that it's
an AxisManager rather than a numpy array.

Data access under an AxisManager is done based on field names. For example::

    >>> print(dset.boresight.az)
    [0. 0. 0. ... 0. 0. 0.]

Advanced data access is possible by a path like syntax. This is especially useful when
data access is dynamic and the field name is not known in advance. For example::

    >>> print(dset["boresight.az"])
    [0. 0. 0. ... 0. 0. 0.]

To slice this object, use the restrict() method.  First, let's
restrict in the 'dets' axis.  Since it's an Axis of type LabelAxis,
the restriction selector must be a list of strings::

    dset.restrict('dets', ['det11', 'det12'])

Similarly, restricting in the samps axis::

    dset.restrict('samps', (10, 300))

After those two restrictions, inspect the shapes of contained
objects::

    >>> print(dset.tod.shape)
    (2, 290)
    >>> print(dset.boresight.az.shape)
    (290,)

For debugging, you can write AxisManagers to HDF5 files and then read
them back.  (This is an experimental feature so don't rely on this for
long term stability!)::

    >>> dset.save('output.h5', 'my_axismanager/dset')

    >>> dset_reloaded = AxisManager.load('output.h5', 'my_axismanager/dset')
    >>> dset_reloaded
    AxisManager(tod[dets,samps], hwp_angle[samps], boresight*[samps],
      dets:LabelAxis(2), samps:IndexAxis(290))

Numerical arrays are stored as simple HDF5 datasets, so you can also
use h5py to load the saved arrays::

    >>> import h5py
    >>> f = h5py.File('output.h5')
    >>> f['my_axismanager/dset/tod'][:]
    <HDF5 dataset "tod": shape (2,290), type "<f8">

To save data with `flacarray`_ compression, pass the "encodings"
argument, and use a nested dict to identify the field you want to save
and to specify the storage details.  For example, suppose container
``dset`` has field ``dset.thermometers.diode1`` that is a float array that
can be safely compressed with precision 1e-4, and field
``dset.thermometers.flags`` is an integer array.  Then sensible
encodings request is::

    >>> encodings = {
         'thermometers': {
            'diode1': {
              'type': 'flacarray',
              'args': {
                'quanta': 1e-4
              }
            },
            'flags': {
              'type': 'flacarray'
            }
         }
       }
   >>> dset.save('output.h5', encodings=encodings)


.. _`flacarray`: https://github.com/hpc4cmb/flacarray


--------------------
Standardized Fields
--------------------

As we develop the SO pipeline we will need to standardize field names that have
specific uses within the pipeline so that functions can be written to expect a
specific set of fields. Not all AxisManagers will have all these fields by
default and many fields are linked to documentation locations where more details
can be found. These are meant to prevent naming collisions and more will be 
added here as the code develops. 

* ``dets`` - the axis for detectors  
* ``samps`` - the axis for samples
* ``timestamps`` `[samps]` - the field for UTC timestamps 
* ``signal`` `[dets, samps]` - the field for detector signal
* | ``obs_info`` - AxisManager of scalars with ObsDb information for the loaded 
  | observation. :ref:`Details here. <obsdb-names-section>`
* ``det_info`` `[dets]` - AxisManager containing loaded detector metadata

  * ``readout_id`` - The unique readout ID of the resonator
  * ``det_id`` - The unique detector ID matched to the resonator.
  * | ``wafer`` - An AxisManager of different parameters related to the hardware
    | mapping on the UFM itself. Loaded based on ``det_id`` field in the
    | ``det_info``.

SMuRF fields loaded through :meth:`sotodlib.io.load_file`.

* ``bias_lines`` - the axis for bias lines in a UFM.
* | ``status`` - A SmurfStatus AxisManager containing information from status
  | frames in the .g3 timestreams. :class:`sotodlib.io.load_smurf.SmurfStatus`
* | ``iir_params`` - An AxisManager with the readout filter parameters. Used by
  | :meth:`sotodlib.tod_ops.filters.iir_filter`.
* | ``primary`` `[samps]` - An AxisManager with SMuRF readout information that is
  | synced with the timestreams.
* ``biases`` `[bias_lines, samps]` - Bias voltage applied to TESes over time.

Pointing information required for :mod:`sotodlib.coords`.

* | ``boresight`` `[samps]` - AxisManager with boresight pointing in horizon
  | coordinates. Child fields are ``az``, ``el``, and ``roll``.
* | ``focal_plane`` `[dets]` - AxisManager with detector position and orientation
  | relative to boresight pointing.
* | ``boresight_equ`` `[samps]` - AxisManager with boresight in equitorial
  |  coordinates.


HWP information 

* | ``hwp_angle`` `[samps]` - AxisManager with hwp rotation angle required for
  | :mod:`sotodlib.io.g3tsmurf_utils.load_hwp_data`
* | ``hwpss_model`` `[dets, samps]` - the fields of fitted HWP synchronous signal
  | derived from :mod:`sotodlib.hwp.get_hwpss`

---------
Reference
---------

The class documentation for AxisManager and the basic Axis types
should be rendered here.

AxisManager
===========

.. autoclass:: sotodlib.core.AxisManager
   :special-members: __init__
   :members:

IndexAxis
=========

.. autoclass:: sotodlib.core.IndexAxis
   :special-members: __init__
   :members:

OffsetAxis
==========

.. autoclass:: sotodlib.core.OffsetAxis
   :special-members: __init__
   :members:

LabelAxis
=========

.. autoclass:: sotodlib.core.LabelAxis
   :special-members: __init__
   :members:

