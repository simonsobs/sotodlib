"""Configuration dataclasses for the various AncilEngine subclasses.

These dataclasses are constructed from dicts, often loaded from yaml.
Each AncilEngine subclass registers its prefered config class in
self.config_class.

"""

from dataclasses import dataclass, field
from . import ANCIL_ENGINES


def register_engine(token: str, config_class):
    """Class decorator used to register an AncilEngine in
    ANCIL_ENGINES and to define the config_class class variable and to
    annotate the config_class docstring with a reference to the class.
    Use like this::

      @cc.register_engine(cc.MyPreciousDataConfig, 'my-precious')
      class MyPreciousData(base.AncilEngine):
         ...

    """
    def _deco(cls):
        cls.engine_id = token
        cls.config_class = config_class
        m, n = config_class.__module__, config_class.__name__
        d = config_class.__doc__
        config_class.__doc__ += \
            f"\n\nConfig class for :class:`{m}.{n}`."
        ANCIL_ENGINES[token] = cls
        return cls
    return _deco


# Base classes not intended for direct use.

@dataclass
class AncilEngineConfig:
    """
    The base class from which all engine configs inherit.
    """

    #: Name for the dataset (subclasses usually specify a default
    #: value).
    dataset_name: str = None

    #: Prefix (directory name) for base data archive storage. This
    #: is intended to be set externally to some top-level storage
    #: location for some set of archives.
    data_prefix : str = None

    #: Directory where data archive should be stored. This is taken
    #: relative to data_prefix, unless an absolute path is specified.
    #: Subclasses will usually recommend a value.
    data_dir: str = None

    #: List of "friend" definitions that the class expects to have
    #: registered.
    friends : field(default_factory=list) = None

    #: Format string to use when turning internal field values into
    #: obsdb column names. This must be None or else include
    #: ``{field}``; it may also include ``{dataset}``, which will
    #: be replaced with the ``dataset_name``.
    obsdb_format: str = None

    #: Query string to use on the obsdb for identifying records that
    #: require an update.  Instead of field names, put each internal
    #: fieldname as a variable (e.g. ``{mean} is null``) and it will be
    #: reprocessed according to ``obsdb_format`` spec.
    obsdb_query: str = None


@dataclass
class LowResTableConfig(AncilEngineConfig):
    """Helper class for data archives that consist of a few gathered
    fields, indexed by time.

    """

    #: Number of seconds in archive block.
    archive_block_seconds: int = 2000000

    #: Time range for this dataset to consider.
    dataset_time_range = (1704000000, None)

    gap_size: float = 300.
    filename_pattern: str = '{dataset_name}_{timestamp}.h5'
    dtypes: list = None


@dataclass
class HkExtractConfig(AncilEngineConfig):
    """Helper class for data archives that are based on SO
    Housekeeping data.  Provide support for extraction and repackaging
    based on hkdb.

    """
    dataset_name: str = None

    #: Map from internal field name (e.g. "udp_az") to full hkdb field
    #: name (e.g. "acu.acu_udp_stream.Corrected_Azimuth").  This
    #: defines what fields are extracted and stored in the archive.
    aliases: dict = field(default_factory=dict)

    #: Config file for accessing the relevant hkdb.
    hkdb_config: str = None

    #: Number of seconds in each HDF5 base data file.
    archive_block_seconds: int = 20000000

    #: Number of seconds in each HDF5 base data group (must divide
    #: archive_block_seconds evenly).
    dataset_block_seconds: int = 1000000

    #: Time range for this dataset to consider.
    dataset_time_range: list[float] = (1704000000, None)

    #: Pattern for generating filenames.
    filename_pattern: str = '{dataset_name}_{timestamp}.h5'

    #: Default (numpy) datatype for storing in base data archive.
    default_dtype: str = 'float32'

    #: Datatypes to override specific fields.
    dtypes: dict = field(default_factory=dict)


#
# Specific engine config classes.
#
# Note that the register_engine decorator will automatically add a
# simple class docstring that links back to the Engine class.
#


@dataclass
class ApexPwvConfig(LowResTableConfig):
    # Overrides.
    dataset_name: str = 'apex_pwv'
    obsdb_format: str = '{dataset}_{field}'
    obsdb_query: str = '{mean} is null'


@dataclass
class TocoPwvConfig(LowResTableConfig):
    # Overrides.
    dataset_name: str = 'toco_pwv'
    obsdb_format: str = '{dataset}_{field}'
    obsdb_query: str = '{mean} is null and {start} is null and {end} is null'

    #: Config file for accessing the site hkdb.
    hkdb_config: str = None


@dataclass
class PwvComboConfig(AncilEngineConfig):
    # Overrides.
    dataset_name: str = 'pwv'
    obsdb_format: str = '{dataset}_{field}'
    obsdb_query: str = '{mean} is null'

    #: Name of the friend dataset from which to retrieve Toco
    #: radiometer data.
    toco_dataset: str = 'toco-pwv'

    #: Name of the friend dataset from which to retrieve APEX
    #: radiometer data.
    apex_dataset: str = 'apex-pwv'
