from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def save_dict_csv(path: str | Path, x: np.ndarray, cols: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["x"] + list(cols.keys())
    arrs = [np.asarray(x).reshape(-1)] + [np.asarray(cols[k]).reshape(-1) for k in cols]
    mat = np.column_stack(arrs)
    header = ",".join(names)
    np.savetxt(path, mat, delimiter=",", header=header, comments="")
