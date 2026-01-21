"""Configuration dataclasses for the various AncilEngine subclasses.

These dataclasses are constructed from dicts, often loaded from yaml.
Each AncilEngine subclass registers its prefered config class in
self.config_class.

"""

from dataclasses import dataclass, field
from . import ANCIL_ENGINES


def register_engine(token: str, config_class):
    """Class decorator used to register an AncilEngine in
    ANCIL_ENGINES and to define the config_class class variable.  Use
    like this::

      @cc.register_engine(cc.MyPreciousDataConfig, 'my-precious')
      class MyPreciousData(base.AncilEngine):
         ...

    """
    def _deco(cls):
        cls.engine_id = token
        cls.config_class = config_class
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

    #: Default prefix (directory) for base data archive storage.
    data_prefix : str = None
    data_dir: str = None

    #: List of "friend" definitions that the class expects to have
    #: registered.
    friends : field(default_factory=list) = None

    #: Test for obsdb query that finds records needing update
    obsdb_format: str = None
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

    #: hkdb config filename to target.
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


# Specific engine config.

@dataclass
class ApexPwvConfig(LowResTableConfig):
    # Overrides.
    dataset_name: str = 'apex_pwv'
    obsdb_format: str = '{dataset}_{field}'
    obsdb_query: str = '{mean} is null'


@dataclass
class PwvComboConfig(AncilEngineConfig):
    # Overrides.
    dataset_name: str = 'pwv'
    obsdb_format: str = '{dataset}_{field}'
    obsdb_query: str = '{mean} is null'

    toco_dataset: str = 'toco-pwv'
    apex_dataset: str = 'apex-pwv'


@dataclass
class TocoPwvConfig(LowResTableConfig):
    obsdb_query: str = '{mean} is null and {start} is null and {end} is null'

    dataset_name: str = 'toco_pwv'
    obsdb_format: str = '{dataset}_{field}'
    hkdb_config: str = None
