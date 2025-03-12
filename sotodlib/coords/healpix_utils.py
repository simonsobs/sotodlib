"""
Helper functions for Healpix operations

Functions are provided to transform between "full", "compressed", and "tiled"
Healpix maps. This follows a scheme in which the full map is broken up into
individual tiles defined by a lower resolution map. The setup ("geometry") is
defined by the base nside, nside_tile of the tiling, and the ordering. Tiled
maps have nTile = 12 * nside_tile**2, and must use "NEST" ordering in this
scheme. If no tiling is used, "RING" is allowed.

Individual tiles are also marked as "active" or "inactive" depending on
whether they include any non-zero pixels. This information is contained in an
active_tile_list: a len(nTile) boolean array of which tiles are active.

- Full map: A normal healpix map (in NEST ordering). Numpy arrays with shape
  (...,npix); can have any number of leading dimensions.
- Tiled map: A len(nTile) list of tiles. None for inactive tiles and a numpy
  array of shape (..., npix/nside) for active tiles.
- Compressed map: Numpy array of shape (nActiveTile, ..., npix/nside).
  Requires additional active tile information to recover a full map. The
  position of the tile axis in a compressed map can be controlled with the
  tile_axis_pos argument.
"""

import numpy as np
from types import SimpleNamespace

def get_geometry(nside, nside_tile=None, ordering='NEST'):
    """Make a simple object to contain Healpix geometry information.
    It has attributes nside, nside_tile, and ordering as given.
    If tiled, the additional attribute 'ntile' is automatically set to
    True (nside_tile='auto') or the number of tiles (nside_tile is an int).

    Args:
      nside: int >=1, power of 2. Nside of the base map.
      nside_tile: int or 'auto' for a tiled map. None for an un-tiled map.
      ordering: 'NEST' for tiled maps. 'NEST' or 'RING' for un-tiled.

    """
    hp_geom = SimpleNamespace(nside=nside, nside_tile=nside_tile, ordering=ordering)
    if nside_tile is not None:
        hp_geom.ntile = True
    if isinstance(nside_tile, (int, np.integer)):
        hp_geom.ntile = 12 * nside_tile**2

    check_geometry(hp_geom)
    return hp_geom

def get_active_tile_list(tiled):
    """From a tiled map, get a len(nTile) boolean array of which tiles are active"""
    return np.array([tile is not None for tile in tiled])

def full_to_compressed(full, active_tile_list, tile_axis_pos=0):
    """Get a compressed map from a full NEST map.

    Args:
      full: Full map in NEST.
      active_tile_list: len(nTile) boolean array of active tiles
      tile_axis_pos: Index for the tile axis in the compressed array
    """
    npix = full.shape[-1]
    npix_per_tile = int(npix / len(active_tile_list))
    cmap = []
    for tile_ind in range(len(active_tile_list)):
        if active_tile_list[tile_ind]:
            cmap.append(full[..., npix_per_tile * tile_ind : npix_per_tile * (tile_ind+1)])
    cmap = np.array(cmap, dtype=full.dtype)
    cmap = np.moveaxis(cmap, 0, tile_axis_pos)
    return cmap

def full_to_tiled(full, active_tile_list):
    """Get a tiled map from a full healpix map."""
    compressed = full_to_compressed(full, active_tile_list)
    tiled = compressed_to_tiled(compressed, active_tile_list)
    return tiled

def compressed_to_full(compressed, active_tile_list, tile_axis_pos=0):
    """Get a full NEST map from a compressed map. Empty tiles are filled with zeros.
    See full_to_compressed for args.
    """
    compressed = np.moveaxis(compressed, tile_axis_pos, 0)
    tile_shape = compressed[0].shape
    npix_per_tile = tile_shape[-1]
    super_shape = tile_shape[:-1]
    npix = len(active_tile_list) * npix_per_tile # ntiles * npix_per_tile
    out = np.zeros(super_shape + (npix,))
    tile_inds = [ii for ii in range(len(active_tile_list)) if active_tile_list[ii]]
    for ii in range(len(compressed)):
        tile_ind = tile_inds[ii]
        out[..., npix_per_tile * tile_ind : npix_per_tile * (tile_ind+1)] = compressed[ii]
    return out

def compressed_to_tiled(compressed, active_tile_list, tile_axis_pos=0):
    """Get a tiled map from a compressed map.
    See full_to_compressed for args.
    """
    compressed = np.moveaxis(compressed, tile_axis_pos, 0)
    out = [None] * len(active_tile_list)
    inds = np.where(active_tile_list)[0]
    assert compressed.shape[0] == inds.size
    for ii, ind in enumerate(inds):
        out[ind] = compressed[ii]
    return out

def tiled_to_compressed(tiled, tile_axis_pos=0):
    """Get a compressed map by removing Nones from a tiled map."""
    arr = np.array([tiled[ii] for ii in range(len(tiled)) if tiled[ii] is not None])
    return np.moveaxis(arr, 0, tile_axis_pos)

def tiled_to_full(tiled):
    """Get a full NEST map from a tiled map. Empty tiles are filled with zeros."""
    compressed = tiled_to_compressed(tiled)
    active_tile_list = get_active_tile_list(tiled)
    return compressed_to_full(compressed, active_tile_list)

def check_valid_nside(nside, name='nside'):
    if not isinstance(nside, (int, np.integer)):
        raise TypeError(f"{name} is of type {type(nside)}; should be an integer")
    if not ((nside >= 1) and ((nside & nside-1) == 0)): # Check for power of 2
        raise ValueError(f"Invalid {name} {nside}. Must be a power of 2 and >= 1")
    return True

def check_geometry(geom):
    # Check for required attributes
    for attr in ['nside', 'nside_tile', 'ordering']:
        if not hasattr(geom, attr):
            raise AttributeError(f"Healpix geometry missing required attribute {attr}")

    nside, nside_tile, ordering = geom.nside, geom.nside_tile, geom.ordering
    # Validate nside
    check_valid_nside(nside)

    # Validate nside_tile
    if (nside_tile is not None) and (nside_tile != "auto"):
        if isinstance(nside_tile, str):
            raise ValueError(f"nside_tile is invalid str {nside_tile}; should be integer, 'auto', or None")

        check_valid_nside(nside_tile, 'nside_tile')
        if nside_tile > nside:
            raise ValueError(f"Invalid nside_tile {nside_tile}; cannot be greated than nside {nside}")

    # Validate ordering
    if not (ordering in ['RING', 'NEST']):
        raise ValueError(f"Invalid value of 'ordering': {ordering}; should be 'RING' or 'NEST'")

    # Validate ntile
    is_tiled = (nside_tile is not None)
    ntile = hasattr(geom, 'ntile')
    if not ((is_tiled and ntile) or ((not is_tiled) and (not ntile))):
        raise ValueError("geom.ntile set incorrectly. Should be set for tiled and unset for un-tiled.")

    return True
