import numpy as np

def untile_healpix_compressed(tiled, tile_axis_pos=0):
    """
    Get a compressed healpix map by removing Nones from a tiled map
    Input: list of tiles. None or (..., npix) in NEST ordering
    Out:   *np.array (nActiveTile, ..., npix) of all active tiles
    """
    arr = np.array([tiled[ii] for ii in range(len(tiled)) if tiled[ii] is not None])
    return np.moveaxis(arr, 0, tile_axis_pos)

def tile_healpix_compressed(compressed, active_tile_list, tile_axis_pos=0):
    compressed = np.moveaxis(compressed, tile_axis_pos, 0)
    out = [None] * len(active_tile_list)
    inds = np.where(active_tile_list)[0]
    assert compressed.shape[0] == inds.size
    for ii, ind in enumerate(inds):
        out[ind] = compressed[ii]
    return out

def get_active_tiles(tiled):
    return np.array([tile is not None for tile in tiled])

def decompress_healpix(compressed, active_tiles):
    """Decompress a healpix map
    Input: See outputs of untile_healpix_compressed
    Output: np.array: Full hp map in nest
    """
    tile_shape = compressed[0].shape
    npix_per_tile = tile_shape[-1]
    super_shape = tile_shape[:-1]
    npix = len(active_tiles) * npix_per_tile # ntiles * npix_per_tile
    out = np.zeros(super_shape + (npix,))
    tile_inds = [ii for ii in range(len(active_tiles)) if active_tiles[ii]]
    for ii in range(len(compressed)):
        tile_ind = tile_inds[ii]
        out[..., npix_per_tile * tile_ind : npix_per_tile * (tile_ind+1)] = compressed[ii]
    return out

def untile_healpix(tiled):
    compressed = untile_healpix_compressed(tiled)
    active_tiles = get_active_tiles(tiled)
    return decompress_healpix(compressed, active_tiles)
