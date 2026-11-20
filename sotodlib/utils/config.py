import os
from argparse import Namespace
from copy import deepcopy
from typing import Any, Optional

import yaml


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from `b` take precedence over values from `a`. When a key
    exists in both dictionaries and both corresponding values are
    dictionaries, those dictionaries are merged recursively. All other
    values from `b` replace the corresponding values from `a`.
    Values are deep-copied when inserted into the result, so mutable values
    in the input dictionaries are not shared with the returned dictionary.

    Parameters
    ----------
    a : dict[str, Any]
        The base dictionary.
    b : dict[str, Any]
        The dictionary whose values take precedence.

    Returns
    -------
    dict[str, Any]
        A new dictionary containing the recursively merged values. Neither
        input dictionary is modified.
    """
    result = deepcopy(a)
    for bk, bv in b.items():
        av = result.get(bk)
        if isinstance(av, dict) and isinstance(bv, dict):
            result[bk] = deep_merge(av, bv)
        else:
            result[bk] = deepcopy(bv)
    return result


def load_config(start_cfg: dict[str, Any], cfg_path: str) -> dict[str, Any]:
    """
    Load a configuration file and recursively merge its base configuration.

    The configuration at `cfg_path` is loaded and merged with
    `start_cfg`. If the loaded configuration contains a `base` key then
    the referenced base configuration is loaded recursively and is merged in.

    Values from the more specific configuration take precedence over values
    from its base configuration.

    Note that relative ``"base"`` paths are resolved relative to the directory
    containing the configuration file that references them.

    Parameters
    ----------
    start_cfg : dict[str, Any]
        Configuration values that take precedence over values loaded from `cfg_path`.
    cfg_path : str
        Path to the YAML configuration file to load.

    Returns
    -------
    dict[str, Any]
        The fully merged configuration.

    Raises
    ------
    FileNotFoundError
        If `cfg_path` or a referenced base configuration does not exist.
    yaml.YAMLError
        If a configuration file contains invalid YAML.
    """
    with open(cfg_path) as file:
        new_cfg = yaml.safe_load(file)

    cfg = deep_merge(new_cfg, start_cfg)
    if "base" in new_cfg:
        base_path = new_cfg["base"]
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(cfg_path), base_path)
        return load_config(cfg, base_path)

    return cfg


def load_config_namespace(cfg_or_path: str, defaults: Optional[dict[str, Any]] = None, replace: Optional[dict[str, str]] = None, require: tuple[str,...] = tuple()) -> tuple[Namespace, str]:
    """
    Load configuration into an argparse namespace.

    `cfg_or_path` may be either a path to a YAML configuration file or a
    YAML-formatted string. Configuration keys can optionally be renamed using
    `replace`, and missing keys can be populated from `defaults`.

    Replacement keys are applied before default values are added. If a
    replacement target already exists, its value is overwritten by the
    value from the original key.

    Parameters
    ----------
    cfg_or_path : str
        Path to a YAML configuration file, or a YAML-formatted string.
    defaults : Optional[dict[str, Any]], default: None 
        Default configuration values. A default is only used when its key is
        not already present in the loaded configuration.
    replace : Optional[dict[str, str]], default: None 
        Mapping from existing configuration keys to their replacement names.
        When a key is present, its value is moved to the new key and the
        original key is removed.
    require : tuple[str,...], default: (,)
        List of keys in the final config that are required.
        Will raise a `KeyError` if a required key is not found.

    Returns
    -------
    cfg
        The loaded configuration as an `argparse.Namespace`.
    cfg_str
        The configuration serialized as a YAML string.
        This is the fully collated config with the base configs
        merged in and the renamed and defaults applied.

    Raises
    ------
    FileNotFoundError
        If `cfg_or_path` is interpreted as a file path but the file does
        not exist.
    yaml.YAMLError
        If the configuration contains invalid YAML.
    KeyError
        If we are missing a required key.
    """
    if defaults is None:
        defaults = {}
    if replace is None:
        replace = {}

    if os.path.isfile(cfg_or_path):
        cfg = load_config({}, cfg_or_path)
    else:
        cfg = yaml.safe_load(cfg_or_path)

    for o, n in replace.items():
        if o not in cfg:
            continue
        cfg[n] = cfg[o]
        del cfg[o]

    for k, v in defaults.items():
        cfg[k] = cfg.get(k, v)

    missing = set(require) - set(cfg.keys())
    if len(missing) > 0:
        raise KeyError(f"Missing config keys: {missing}")

    cfg_str = yaml.dump(cfg)
    return Namespace(**cfg), cfg_str

