"""Write flattened run/job parameters as HDF5 root attributes."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np


def attr_value_for_h5(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str, np.integer, np.floating, np.bool_)):
        return value.item() if isinstance(value, (np.integer, np.floating, np.bool_)) else value
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def flatten_metadata_for_h5(metadata: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into HDF5-safe scalar attributes."""
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        attr_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_metadata_for_h5(value, prefix=f"{attr_key}_"))
        else:
            converted = attr_value_for_h5(value)
            if converted is not None:
                flat[attr_key] = converted
    return flat


def write_h5_run_metadata(
    h5path: str,
    metadata: dict[str, Any],
    *,
    merge: bool = True,
) -> None:
    """Merge flattened run metadata into an HDF5 file's root attributes."""
    if not metadata or not os.path.isfile(h5path):
        return
    flat = flatten_metadata_for_h5(metadata)
    if not flat:
        return
    with h5py.File(h5path, "a") as f:
        for key, value in flat.items():
            if merge and key in f.attrs:
                continue
            f.attrs[key] = value


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
