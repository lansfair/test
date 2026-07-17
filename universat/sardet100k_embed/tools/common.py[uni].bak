"""Common helpers for UniverSat tooling (mirrors olmoearth/tools/common.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "processing",
    enabled: bool = True,
) -> Iterator[T]:
    """Yield items with tqdm progress, falling back to periodic prints."""
    if not enabled:
        yield from iterable
        return
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    try:
        from tqdm.auto import tqdm

        yield from tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
        return
    except ImportError:
        pass

    if total is None:
        print(f"{desc}: started")
        for idx, item in enumerate(iterable, start=1):
            if idx == 1 or idx % 100 == 0:
                print(f"{desc}: processed {idx}")
            yield item
        print(f"{desc}: done")
        return

    print(f"{desc}: 0/{total}")
    step = max(1, total // 20)
    for idx, item in enumerate(iterable, start=1):
        yield item
        if idx == total or idx % step == 0:
            print(f"{desc}: {idx}/{total}")


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2)


def coerce_affine_transform(transform: Any):
    """Convert a rasterio/Affine-like transform to ``Affine``."""
    from rasterio.transform import Affine

    if transform is None:
        return None
    if isinstance(transform, Affine):
        return transform
    if all(hasattr(transform, key) for key in ("a", "b", "c", "d", "e", "f")):
        return Affine(
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )
    if isinstance(transform, dict):
        if all(key in transform for key in ("a", "b", "c", "d", "e", "f")):
            return Affine(
                float(transform["a"]),
                float(transform["b"]),
                float(transform["c"]),
                float(transform["d"]),
                float(transform["e"]),
                float(transform["f"]),
            )
        if "coefficients" in transform:
            transform = transform["coefficients"]
    if isinstance(transform, np.ndarray):
        transform = transform.tolist()
    if isinstance(transform, (list, tuple)):
        values = [float(value) for value in transform]
        if len(values) >= 6:
            return Affine(*values[:6])
    raise TypeError(f"Unsupported affine transform: {transform!r}")


def affine_to_coefficients(transform: Any) -> list[float] | None:
    """Return JSON-safe six-coefficient Affine representation."""
    affine = coerce_affine_transform(transform)
    if affine is None:
        return None
    return [
        float(affine.a),
        float(affine.b),
        float(affine.c),
        float(affine.d),
        float(affine.e),
        float(affine.f),
    ]


def save_geotiff(
    path: Path,
    array: np.ndarray,
    descriptions: list[str] | tuple[str, ...] | None = None,
    transform: Any | None = None,
    crs: Any | None = None,
) -> None:
    """Save a 2D label/mask or CHW image tensor as a GeoTIFF."""
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    if array.ndim == 2:
        output = array[None, ...]
    elif array.ndim == 3:
        output = array
    else:
        raise ValueError(f"GeoTIFF array must be 2D or CHW, got {array.shape}")

    if output.dtype == np.float64:
        output = output.astype(np.float32)
    elif output.dtype == np.int64:
        if output.size and output.min() >= 0 and output.max() <= 255:
            output = output.astype(np.uint8)
        else:
            output = output.astype(np.int32)
    profile = {
        "driver": "GTiff",
        "height": int(output.shape[1]),
        "width": int(output.shape[2]),
        "count": int(output.shape[0]),
        "dtype": str(output.dtype),
        "transform": coerce_affine_transform(transform) or Affine.identity(),
        "compress": "lzw",
        "BIGTIFF": "IF_SAFER",
    }
    if crs is not None:
        profile["crs"] = crs
    if np.issubdtype(output.dtype, np.floating):
        profile["predictor"] = 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(output)
            if descriptions is not None:
                dst.descriptions = tuple(descriptions)
