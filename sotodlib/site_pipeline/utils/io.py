import os
from typing import Any, Dict

import numpy as np
from pixell import bunch, utils as putils


def write_depth1_map(
    prefix: str,
    data: np.ndarray,
    dtype: np.typing.DTypeLike = np.float32,
    binned: bool = False,
    rhs: bool = False,
    unit: str = "K",
):
    data.signal.write(prefix, "map", data.map.astype(dtype), unit=unit)
    data.signal.write(prefix, "ivar", data.ivar.astype(dtype), unit=f"{unit}^-2")
    data.signal.write(prefix, "time", data.tmap.astype(dtype))

    if binned:
        data.signal.write(prefix, "bin", data.bin.astype(dtype), unit=unit)

    if rhs:
        data.signal.write(
            prefix, "rhs", data.signal.rhs.astype(dtype), unit=f"{unit}^-1"
        )


def write_depth1_info(oname: str, info: Dict[Any, Any]):
    putils.mkdir(os.path.dirname(oname))
    bunch.write(oname, info)
