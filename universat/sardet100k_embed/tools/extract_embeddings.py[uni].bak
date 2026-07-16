"""Extract dense UniverSat embeddings for offline MMSegmentation probes.

This script is adapted from ``olmoearth/tools/extract_embeddings.py`` to work
with the UniverSat multimodal backbone on the PASTIS-R dataset. It builds the
backbone directly from an MMSegmentation config, runs inference sample by
sample, and writes embeddings (plus optional labels/inputs) as GeoTIFFs
organized by split and sample id.

Run from the MMSegmentation root (the folder containing ``tools/``) with the
UniverSat project on ``PYTHONPATH``::

    export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
    python projects/universat/tools/extract_embeddings.py \
        projects/universat/configs/extract_embeddings_pastisr_universat-base.py \
        --output-root work_dirs/universat_pastisr_embeddings \
        --splits train val test

For distributed extraction launch it with ``torchrun`` / ``python -m torch.distributed.launch``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from common import (
    affine_to_coefficients,
    coerce_affine_transform,
    progress_iter,
    save_geotiff,
    save_json,
)


SCRIPT_DEFAULTS = {
    "config": (
        "projects/universat/configs/extract_embeddings_pastisr_universat-base.py"
    ),
    "output_root": "work_dirs/universat_pastisr_embeddings",
    "splits": ["train", "val", "test"],
    "batch_size": 1,
    "tile_size": 0,
    "tile_overlap": 0.0,
    "device": "auto",
    "precision": "bf16",
    "skip_existing": True,
    "save_inputs": False,
    "save_raw_inputs": False,
    "quiet": False,
    "pipeline_key": "test_pipeline",
    "cfg_options": None,
}


@dataclass(frozen=True)
class DistContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int


def _parse_cfg_options(items: list[str] | None) -> dict[str, Any] | None:
    if items is None:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"cfg option must be key=value, got: {item}")
        key, value = item.split("=", 1)
        if value.lower() == "none":
            parsed = None
        elif value.lower() in {"true", "false"}:
            parsed = value.lower() == "true"
        else:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
        out[key] = parsed
    return out


def _get_dist_context() -> DistContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistContext(
        is_distributed=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def _resolve_device(device_arg: str, ctx: DistContext) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        if ctx.is_distributed:
            return torch.device(f"cuda:{ctx.local_rank}")
        return torch.device("cuda")
    return torch.device("cpu")


def _init_distributed(ctx: DistContext, device: torch.device) -> None:
    if not ctx.is_distributed:
        return
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)


def _barrier(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_distributed(ctx: DistContext) -> None:
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _configure_cuda_fast_math(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _clear_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _safe_sample_id(index: int, metainfo: dict[str, Any]) -> str:
    sample_id = metainfo.get("id_patch")
    if sample_id is None:
        sample_id = metainfo.get("sample_id")
    if sample_id is None:
        sample_id = metainfo.get("img_path")
    if sample_id is None:
        sample_id = f"{index:06d}"
    return str(sample_id).replace("\\", "_").replace("/", "_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _georef_from_metainfo(
    metainfo: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    """UniverSat/PASTIS-R samples are not georeferenced; return identity."""
    return (
        metainfo.get("universat_transform"),
        metainfo.get("universat_crs"),
    )


def _scale_transform_for_shape(
    transform: Any | None,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> list[float] | None:
    if src_hw[0] <= 0 or src_hw[1] <= 0:
        return None
    if dst_hw[0] <= 0 or dst_hw[1] <= 0:
        return None
    from rasterio.transform import Affine

    base = (
        coerce_affine_transform(transform)
        if transform is not None
        else Affine.identity()
    )
    scale_x = float(src_hw[1]) / float(dst_hw[1])
    scale_y = float(src_hw[0]) / float(dst_hw[0])
    return affine_to_coefficients(base * Affine.scale(scale_x, scale_y))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset(
    cfg,
    split: str,
    pipeline_key: str | None,
    save_raw_inputs: bool,
):
    from mmseg.registry import DATASETS

    dataloader_key = f"{split}_dataloader"
    if dataloader_key not in cfg:
        raise KeyError(f"Config does not define {dataloader_key}.")
    dataset_cfg = copy.deepcopy(cfg[dataloader_key]["dataset"])
    if pipeline_key is not None:
        if pipeline_key not in cfg:
            raise KeyError(f"Config does not define {pipeline_key}.")
        dataset_cfg["pipeline"] = copy.deepcopy(cfg[pipeline_key])
    # PASTIS-R pipeline does not support raw-input export; keep the flag for API
    # compatibility but do not modify the pipeline.
    _ = save_raw_inputs
    return DATASETS.build(dataset_cfg)


def _build_backbone(cfg, device: torch.device):
    from mmseg.registry import MODELS

    backbone = MODELS.build(copy.deepcopy(cfg.model.backbone))
    backbone.init_weights()
    backbone.to(device)
    backbone.eval()
    return backbone


def _get_output_grid(cfg) -> int:
    """Return the configured output side length of the backbone.

    UniverSat's ``output_grid`` is the side length of the square output token
    grid (not the total number of tokens). This is used to derive the spatial
    stride between input pixels and output tokens.
    """
    backbone_cfg = cfg.model.get("backbone", {})
    output_grid = backbone_cfg.get("output_grid")
    if output_grid is None:
        raise ValueError(
            "UniverSat extraction requires ``model.backbone.output_grid`` to be "
            "set in the config. Add e.g. ``output_grid=128`` for PASTIS-R."
        )
    return int(output_grid)


def _split_indices(length: int, ctx: DistContext) -> list[int]:
    if not ctx.is_distributed:
        return list(range(length))
    return list(range(ctx.rank, length, ctx.world_size))


def _embedding_shape(path: Path) -> list[int]:
    import rasterio

    with rasterio.open(path) as src:
        return [int(src.count), int(src.height), int(src.width)]


def _make_manifest_sample(
    index: int,
    split: str,
    sample_id: str,
    label: np.ndarray,
    feature_shape: list[int],
    metainfo: dict[str, Any],
    input_paths: dict[str, str] | None = None,
    input_shapes: dict[str, list[int]] | None = None,
    raw_input_shape: list[int] | None = None,
) -> dict[str, Any]:
    embedding_rel = Path(split) / sample_id / "embedding.tif"
    label_rel = Path(split) / sample_id / "label.tif"
    sample = {
        "sample_id": sample_id,
        "source_index": int(index),
        "embedding_path": str(embedding_rel).replace("\\", "/"),
        "seg_map_path": str(label_rel).replace("\\", "/"),
        "dataset_name": metainfo.get("dataset_name"),
        "ori_shape": list(label.shape),
        "embedding_shape": feature_shape,
    }
    if input_paths is not None:
        sample["input_paths"] = {
            str(k): str(v).replace("\\", "/") for k, v in input_paths.items()
        }
    if input_shapes is not None:
        sample["input_shapes"] = {
            str(k): list(v) for k, v in input_shapes.items()
        }
    if raw_input_shape is not None:
        raw_input_rel = Path(split) / sample_id / "raw_input.tif"
        sample["raw_input_path"] = str(raw_input_rel).replace("\\", "/")
        sample["raw_input_shape"] = raw_input_shape
    transform, crs = _georef_from_metainfo(metainfo)
    if crs is not None:
        sample["crs"] = str(crs)
    transform_coefficients = _scale_transform_for_shape(
        transform,
        tuple(int(dim) for dim in label.shape[:2]),
        tuple(int(dim) for dim in label.shape[:2]),
    )
    if transform_coefficients is not None:
        sample["transform"] = transform_coefficients
        sample["embedding_transform"] = _scale_transform_for_shape(
            transform,
            tuple(int(dim) for dim in label.shape[:2]),
            (int(feature_shape[-2]), int(feature_shape[-1])),
        )
        sample["is_georeferenced"] = crs is not None and transform is not None
    if "id_patch" in metainfo:
        sample["id_patch"] = int(metainfo["id_patch"])
    if "fold" in metainfo:
        sample["fold"] = int(metainfo["fold"])
    if "timestamps" in metainfo:
        sample["timestamps"] = _jsonable(metainfo["timestamps"])
    return sample


def _task_from_item(
    index: int,
    split: str,
    output_root: Path,
    item: dict[str, Any],
) -> dict[str, Any]:
    data_sample = item["data_samples"]
    metainfo = data_sample.metainfo
    sample_id = _safe_sample_id(index, metainfo)
    sample_dir = output_root / split / sample_id
    embedding_path = sample_dir / "embedding.tif"
    label_path = sample_dir / "label.tif"
    label = data_sample.gt_sem_seg.data.squeeze(0).cpu().numpy()
    return {
        "index": index,
        "item": item,
        "metainfo": metainfo,
        "sample_id": sample_id,
        "sample_dir": sample_dir,
        "embedding_path": embedding_path,
        "label_path": label_path,
        "label": label,
    }


def _save_task_output(
    task: dict[str, Any],
    feature: np.ndarray,
    embedding_names: list[str],
    save_inputs: bool,
    save_raw_inputs: bool,
    tile_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if save_raw_inputs:
        raise NotImplementedError(
            "Raw input export is not supported for the UniverSat PASTIS-R pipeline. "
            "Run with --no-save-raw-inputs."
        )

    task["sample_dir"].mkdir(parents=True, exist_ok=True)
    transform, crs = _georef_from_metainfo(task["metainfo"])
    label_hw = tuple(int(dim) for dim in task["label"].shape[:2])
    embedding_transform = _scale_transform_for_shape(
        transform,
        label_hw,
        (int(feature.shape[-2]), int(feature.shape[-1])),
    )
    save_geotiff(
        task["embedding_path"],
        feature.astype(np.float32, copy=False),
        descriptions=embedding_names,
        transform=embedding_transform,
        crs=crs,
    )

    input_paths: dict[str, str] | None = None
    input_shapes: dict[str, list[int]] | None = None
    if save_inputs:
        inputs = task["item"]["inputs"]
        input_paths = {}
        input_shapes = {}
        for key, tensor in inputs.items():
            # Skip date tensors; only save modality rasters.
            if key.endswith("_dates"):
                continue
            arr = tensor.detach().cpu().numpy()
            # Time-series modalities are (T, C, H, W); spatial-only are (C, H, W).
            if arr.ndim == 4:
                arr = arr.mean(axis=0)
            path = task["sample_dir"] / f"input_{key}.tif"
            save_geotiff(
                path,
                arr.astype(np.float32, copy=False),
                transform=transform,
                crs=crs,
            )
            input_paths[key] = Path(task["split"]) / task["sample_id"] / path.name
            input_shapes[key] = list(arr.shape)

    save_geotiff(task["label_path"], task["label"], transform=transform, crs=crs)
    sample = _make_manifest_sample(
        index=task["index"],
        split=task["embedding_path"].parent.parent.name,
        sample_id=task["sample_id"],
        label=task["label"],
        feature_shape=list(feature.shape),
        metainfo=task["metainfo"],
        input_paths=input_paths,
        input_shapes=input_shapes,
    )
    if tile_info is not None:
        sample["tiling"] = tile_info
    return sample


def _maybe_manifest_from_existing(
    task: dict[str, Any],
    split: str,
    skip_existing: bool,
    save_inputs: bool,
    save_raw_inputs: bool,
) -> dict[str, Any] | None:
    if not skip_existing:
        return None
    if not task["embedding_path"].exists() or not task["label_path"].exists():
        return None
    if save_raw_inputs:
        # Raw inputs are never written for this pipeline; existing outputs are
        # incomplete if the caller requests them now.
        return None

    input_paths: dict[str, str] | None = None
    input_shapes: dict[str, list[int]] | None = None
    if save_inputs:
        inputs = task["item"]["inputs"]
        input_paths = {}
        input_shapes = {}
        for key in inputs:
            if key.endswith("_dates"):
                continue
            path = task["sample_dir"] / f"input_{key}.tif"
            if not path.exists():
                return None
            input_paths[key] = Path(split) / task["sample_id"] / path.name
            input_shapes[key] = _embedding_shape(path)

    return _make_manifest_sample(
        index=task["index"],
        split=split,
        sample_id=task["sample_id"],
        label=task["label"],
        feature_shape=_embedding_shape(task["embedding_path"]),
        metainfo=task["metainfo"],
        input_paths=input_paths,
        input_shapes=input_shapes,
    )


def _get_first_modality(inputs: dict[str, Any]) -> str:
    """Return the first modality key that is not a date tensor."""
    for key in inputs:
        if not key.endswith("_dates"):
            return key
    raise ValueError("No modality tensor found in inputs dict.")


def _stack_inputs(tasks: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Stack UniverSat multimodal inputs (and dates) across a batch.

    All tasks must have the same set of keys and the same shape per key.
    Modality tensors are cast to float; date tensors keep their original dtype.
    """
    keys = list(tasks[0]["item"]["inputs"].keys())
    for task in tasks:
        if set(task["item"]["inputs"].keys()) != set(keys):
            raise ValueError(
                "Batch contains mixed input keys: "
                f"{set(keys)} vs {set(task['item']['inputs'].keys())}"
            )

    shapes: dict[str, set[tuple[int, ...]]] = {key: set() for key in keys}
    for task in tasks:
        for key, tensor in task["item"]["inputs"].items():
            shapes[key].add(tuple(tensor.shape))
    for key, shape_set in shapes.items():
        if len(shape_set) != 1:
            raise ValueError(
                f"Batch contains mixed shapes for input key '{key}': {sorted(shape_set)}"
            )

    stacked: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [task["item"]["inputs"][key] for task in tasks]
        if key.endswith("_dates"):
            stacked[key] = torch.stack(tensors, dim=0)
        else:
            stacked[key] = torch.stack([t.float() for t in tensors], dim=0)

    if device.type == "cuda":
        stacked = {
            key: tensor.pin_memory() for key, tensor in stacked.items()
        }
    return {
        key: tensor.to(device, non_blocking=device.type == "cuda")
        for key, tensor in stacked.items()
    }


def _forward_tasks(
    backbone,
    tasks: list[dict[str, Any]],
    device: torch.device,
    precision: str,
) -> list[np.ndarray]:
    inputs = _stack_inputs(tasks, device)
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=precision == "bf16" and device.type == "cuda",
    ):
        features = backbone(inputs)[0]
    return [feature.float().contiguous().cpu().numpy() for feature in features]


def _forward_tasks_with_oom_retry(
    backbone,
    tasks: list[dict[str, Any]],
    device: torch.device,
    precision: str,
    rank: int,
) -> list[np.ndarray]:
    try:
        return _forward_tasks(backbone, tasks, device, precision)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(tasks) == 1:
            raise
        print(
            f"[rank {rank}] OOM for batch size {len(tasks)}; "
            "retrying samples one by one."
        )
        _clear_cache(device)
        features = []
        for task in tasks:
            features.extend(_forward_tasks(backbone, [task], device, precision))
        return features


def _resolve_tile_overlap(tile_size: int, tile_overlap: float) -> int:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if tile_overlap < 0:
        raise ValueError(f"tile_overlap must be >= 0, got {tile_overlap}")
    if tile_overlap < 1.0:
        overlap = int(round(tile_size * tile_overlap))
    else:
        overlap = int(round(tile_overlap))
    if overlap >= tile_size:
        raise ValueError(
            "tile overlap must be smaller than tile_size, "
            f"got overlap={overlap}, tile_size={tile_size}"
        )
    return overlap


def _build_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _build_tile_specs(
    height: int,
    width: int,
    tile_size: int,
    tile_overlap: float,
    feature_stride: int,
) -> list[tuple[int, int, int, int]]:
    overlap = _resolve_tile_overlap(tile_size, tile_overlap)
    stride = tile_size - overlap
    if tile_size % feature_stride != 0:
        raise ValueError(
            f"tile_size={tile_size} must be divisible by "
            f"feature_stride={feature_stride}"
        )
    if stride % feature_stride != 0:
        raise ValueError(
            f"tile stride={stride} must be divisible by "
            f"feature_stride={feature_stride}"
        )
    if height % feature_stride != 0 or width % feature_stride != 0:
        raise ValueError(
            "Sliding-window extraction requires input height/width divisible "
            f"by feature_stride={feature_stride}; got {(height, width)}. "
            "Pad in the MMSeg pipeline before extracting embeddings."
        )

    row_starts = _build_tile_starts(height, tile_size, stride)
    col_starts = _build_tile_starts(width, tile_size, stride)
    return [
        (row, col, min(tile_size, height - row), min(tile_size, width - col))
        for row in row_starts
        for col in col_starts
    ]


def _blend_weights_1d(
    length: int,
    left_overlap: int,
    right_overlap: int,
    touches_left: bool,
    touches_right: bool,
) -> np.ndarray:
    weights = np.ones((length,), dtype=np.float32)
    if left_overlap > 0 and not touches_left:
        left_overlap = min(left_overlap, length)
        weights[:left_overlap] = np.minimum(
            weights[:left_overlap],
            np.linspace(
                1.0 / (left_overlap + 1),
                1.0,
                left_overlap,
                dtype=np.float32,
            ),
        )
    if right_overlap > 0 and not touches_right:
        right_overlap = min(right_overlap, length)
        weights[-right_overlap:] = np.minimum(
            weights[-right_overlap:],
            np.linspace(
                1.0,
                1.0 / (right_overlap + 1),
                right_overlap,
                dtype=np.float32,
            ),
        )
    return weights


def _tile_blend_weight(
    tile: tuple[int, int, int, int],
    feature_shape: tuple[int, int],
    full_shape: tuple[int, int],
    overlaps: tuple[int, int, int, int],
    feature_stride: int,
) -> np.ndarray:
    row, col, _, _ = tile
    feat_h, feat_w = feature_shape
    full_h, full_w = full_shape
    top, bottom, left, right = overlaps
    top_feat = top // feature_stride
    bottom_feat = bottom // feature_stride
    left_feat = left // feature_stride
    right_feat = right // feature_stride
    row0 = row // feature_stride
    col0 = col // feature_stride
    wy = _blend_weights_1d(
        feat_h,
        top_feat,
        bottom_feat,
        touches_left=row0 == 0,
        touches_right=row0 + feat_h == full_h,
    )
    wx = _blend_weights_1d(
        feat_w,
        left_feat,
        right_feat,
        touches_left=col0 == 0,
        touches_right=col0 + feat_w == full_w,
    )
    return np.outer(wy, wx).astype(np.float32, copy=False)


def _build_tile_overlap_map(
    tiles: list[tuple[int, int, int, int]],
) -> dict[tuple[int, int, int, int], tuple[int, int, int, int]]:
    row_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    col_groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for tile in tiles:
        row, col, _, _ = tile
        row_groups.setdefault(row, []).append(tile)
        col_groups.setdefault(col, []).append(tile)

    left_right: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for group in row_groups.values():
        group.sort(key=lambda item: item[1])
        for idx, tile in enumerate(group):
            _, col, _, width = tile
            left = 0
            right = 0
            if idx > 0:
                prev = group[idx - 1]
                left = max(0, prev[1] + prev[3] - col)
            if idx + 1 < len(group):
                next_tile = group[idx + 1]
                right = max(0, col + width - next_tile[1])
            left_right[tile] = (left, right)

    top_bottom: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for group in col_groups.values():
        group.sort(key=lambda item: item[0])
        for idx, tile in enumerate(group):
            row, _, height, _ = tile
            top = 0
            bottom = 0
            if idx > 0:
                prev = group[idx - 1]
                top = max(0, prev[0] + prev[2] - row)
            if idx + 1 < len(group):
                next_tile = group[idx + 1]
                bottom = max(0, row + height - next_tile[0])
            top_bottom[tile] = (top, bottom)

    overlap_map = {}
    for tile in tiles:
        top, bottom = top_bottom.get(tile, (0, 0))
        left, right = left_right.get(tile, (0, 0))
        overlap_map[tile] = (top, bottom, left, right)
    return overlap_map


def _tile_task_from_parent(
    task: dict[str, Any],
    tile: tuple[int, int, int, int],
) -> dict[str, Any]:
    row, col, height, width = tile
    inputs = task["item"]["inputs"]
    tiled_inputs = {}
    for key, tensor in inputs.items():
        # Spatial dimensions are the last two for both modality and date tensors.
        if tensor.ndim >= 2:
            tiled_inputs[key] = tensor[..., row : row + height, col : col + width]
        else:
            # Date tensors are 1-D and not spatial; keep unchanged for the tile.
            tiled_inputs[key] = tensor
    return {
        "item": {"inputs": tiled_inputs},
        "metainfo": task["metainfo"],
    }


def _extract_tiled_feature(
    backbone,
    task: dict[str, Any],
    batch_size: int,
    device: torch.device,
    precision: str,
    rank: int,
    tile_size: int,
    tile_overlap: float,
    output_grid: int,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    inputs = task["item"]["inputs"]
    first_mod = _get_first_modality(inputs)
    _, height, width = inputs[first_mod].shape[-3:]

    # UniverSat produces a fixed-size output grid; derive the pixel stride.
    if height % output_grid != 0 or width % output_grid != 0:
        raise ValueError(
            f"Input shape ({height}, {width}) is not divisible by output_grid "
            f"({output_grid}); pad the input or disable tiling."
        )
    feature_stride = max(height // output_grid, 1)

    tiles = _build_tile_specs(
        height,
        width,
        tile_size,
        tile_overlap,
        feature_stride,
    )
    overlap = _resolve_tile_overlap(tile_size, tile_overlap)
    tile_tasks = [_tile_task_from_parent(task, tile) for tile in tiles]
    tile_features = []
    for start in range(0, len(tile_tasks), batch_size):
        batch = tile_tasks[start : start + batch_size]
        tile_features.extend(
            _forward_tasks_with_oom_retry(
                backbone,
                batch,
                device,
                precision,
                rank,
            )
        )

    full_h = height // feature_stride
    full_w = width // feature_stride
    channels = int(tile_features[0].shape[0])
    canvas = np.zeros((channels, full_h, full_w), dtype=np.float32)
    weight_sum = np.zeros((1, full_h, full_w), dtype=np.float32)
    overlap_map = _build_tile_overlap_map(tiles)
    for tile, feature in zip(tiles, tile_features):
        row, col, _, _ = tile
        row0 = row // feature_stride
        col0 = col // feature_stride
        feat_h, feat_w = feature.shape[-2:]
        overlaps = overlap_map[tile]
        for side, value in zip(("top", "bottom", "left", "right"), overlaps):
            if value % feature_stride != 0:
                raise ValueError(
                    f"Tile {side} overlap={value} must be divisible by "
                    f"feature_stride={feature_stride}"
                )
        weight = _tile_blend_weight(
            tile,
            (feat_h, feat_w),
            (full_h, full_w),
            overlaps,
            feature_stride,
        )[None, ...]
        canvas[:, row0 : row0 + feat_h, col0 : col0 + feat_w] += feature * weight
        weight_sum[:, row0 : row0 + feat_h, col0 : col0 + feat_w] += weight

    canvas = canvas / np.clip(weight_sum, 1.0e-6, None)
    if verbose:
        print(
            f"[rank {rank}] tiled {task['sample_id']}: "
            f"{len(tiles)} tiles -> {tuple(canvas.shape)}"
        )
    return canvas, {
        "tile_size": tile_size,
        "tile_overlap": float(tile_overlap),
        "tile_overlap_pixels": overlap,
        "num_tiles": len(tiles),
        "feature_stride": feature_stride,
        "merge_mode": "weighted_overlap_blend",
    }


def _flush_shape_buckets(
    backbone,
    buckets: dict[tuple[int, ...], list[dict[str, Any]]],
    split_samples: list[dict[str, Any]],
    device: torch.device,
    precision: str,
    embedding_names: list[str] | None,
    save_inputs: bool,
    save_raw_inputs: bool,
    verbose: bool,
    rank: int,
) -> list[str] | None:
    for shape, tasks in list(buckets.items()):
        if not tasks:
            continue
        if verbose:
            print(f"[rank {rank}] extracting shape={shape}, batch={len(tasks)}")
        features = _forward_tasks_with_oom_retry(
            backbone,
            tasks,
            device,
            precision,
            rank,
        )

        if embedding_names is None:
            embedding_names = [
                f"embedding_{idx:04d}" for idx in range(features[0].shape[0])
            ]
        for task, feature in zip(tasks, features):
            split_samples.append(
                _save_task_output(
                    task,
                    feature,
                    embedding_names,
                    save_inputs,
                    save_raw_inputs,
                    tile_info=None,
                )
            )
        buckets[shape] = []
        _clear_cache(device)
    return embedding_names


def _extract_split(
    cfg,
    split: str,
    output_root: Path,
    batch_size: int,
    device: torch.device,
    pipeline_key: str | None,
    ctx: DistContext,
    precision: str,
    skip_existing: bool,
    save_inputs: bool,
    save_raw_inputs: bool,
    tile_size: int,
    tile_overlap: float,
    verbose: bool,
) -> dict[str, Any]:
    dataset = _build_dataset(cfg, split, pipeline_key, save_raw_inputs)
    backbone = _build_backbone(cfg, device)
    output_grid = _get_output_grid(cfg)
    local_indices = _split_indices(len(dataset), ctx)
    split_samples: list[dict[str, Any]] = []
    buckets: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    embedding_names: list[str] | None = None
    skipped_count = 0
    tiled_count = 0

    with torch.inference_mode():
        for index in progress_iter(
            local_indices,
            total=len(local_indices),
            desc=f"[rank {ctx.rank}] extracting {split}",
            enabled=verbose,
        ):
            task = _task_from_item(index, split, output_root, dataset[index])
            task["split"] = split  # used by _save_task_output for relative paths
            existing = _maybe_manifest_from_existing(
                task,
                split,
                skip_existing,
                save_inputs,
                save_raw_inputs,
            )
            if existing is not None:
                split_samples.append(existing)
                skipped_count += 1
                continue

            inputs = task["item"]["inputs"]
            first_mod = _get_first_modality(inputs)
            _, height, width = inputs[first_mod].shape[-3:]
            should_tile = tile_size > 0 and (
                height > tile_size or width > tile_size
            )
            if should_tile:
                embedding_names = _flush_shape_buckets(
                    backbone=backbone,
                    buckets=buckets,
                    split_samples=split_samples,
                    device=device,
                    precision=precision,
                    embedding_names=embedding_names,
                    save_inputs=save_inputs,
                    save_raw_inputs=save_raw_inputs,
                    verbose=verbose,
                    rank=ctx.rank,
                )
                feature, tile_info = _extract_tiled_feature(
                    backbone=backbone,
                    task=task,
                    batch_size=batch_size,
                    device=device,
                    precision=precision,
                    rank=ctx.rank,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    output_grid=output_grid,
                    verbose=verbose,
                )
                if embedding_names is None:
                    embedding_names = [
                        f"embedding_{idx:04d}"
                        for idx in range(feature.shape[0])
                    ]
                split_samples.append(
                    _save_task_output(
                        task,
                        feature,
                        embedding_names,
                        save_inputs,
                        save_raw_inputs,
                        tile_info=tile_info,
                    )
                )
                tiled_count += 1
                _clear_cache(device)
                continue

            # Use the shape of every input key as the bucket key so that variable
            # time lengths do not get mixed in the same batch.
            shape = tuple(
                tuple(inputs[key].shape) for key in sorted(inputs.keys())
            )
            bucket = buckets.setdefault(shape, [])
            bucket.append(task)
            if len(bucket) >= batch_size:
                embedding_names = _flush_shape_buckets(
                    backbone=backbone,
                    buckets={shape: bucket},
                    split_samples=split_samples,
                    device=device,
                    precision=precision,
                    embedding_names=embedding_names,
                    save_inputs=save_inputs,
                    save_raw_inputs=save_raw_inputs,
                    verbose=verbose,
                    rank=ctx.rank,
                )
                buckets[shape] = []

        embedding_names = _flush_shape_buckets(
            backbone=backbone,
            buckets=buckets,
            split_samples=split_samples,
            device=device,
            precision=precision,
            embedding_names=embedding_names,
            save_inputs=save_inputs,
            save_raw_inputs=save_raw_inputs,
            verbose=verbose,
            rank=ctx.rank,
        )

    split_samples.sort(key=lambda sample: int(sample["source_index"]))
    manifest = {
        "metainfo": {
            "source_config": str(getattr(cfg, "filename", "")),
            "split": split,
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "format": "universat_embedding_geotiff_manifest",
        },
        "samples": split_samples,
    }
    manifest_path = output_root / (
        f"{split}_rank{ctx.rank}.json" if ctx.is_distributed else f"{split}.json"
    )
    save_json(manifest_path, manifest)
    return {
        "split": split,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "assigned_samples": len(local_indices),
        "num_samples": len(split_samples),
        "skipped_existing": skipped_count,
        "tiled_samples": tiled_count,
        "save_inputs": save_inputs,
        "save_raw_inputs": save_raw_inputs,
        "tile_size": tile_size,
        "tile_overlap": float(tile_overlap),
        "manifest": str(manifest_path),
    }


def _merge_split_manifests(
    output_root: Path,
    split: str,
    world_size: int,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    rank_manifests = []
    for rank in range(world_size):
        path = output_root / f"{split}_rank{rank}.json"
        payload = _load_json(path)
        rank_manifests.append(str(path))
        samples.extend(payload.get("samples", []))
    samples.sort(key=lambda sample: int(sample["source_index"]))
    manifest = {
        "metainfo": {
            "split": split,
            "world_size": world_size,
            "format": "universat_embedding_geotiff_manifest",
            "rank_manifests": rank_manifests,
        },
        "samples": samples,
    }
    out_path = output_root / f"{split}.json"
    save_json(out_path, manifest)
    return {
        "split": split,
        "num_samples": len(samples),
        "manifest": str(out_path),
        "rank_manifests": rank_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dense UniverSat embeddings for offline MMSeg probes."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=SCRIPT_DEFAULTS["config"],
        help="UniverSat MMSeg config for extraction.",
    )
    parser.add_argument(
        "--output-root",
        default=SCRIPT_DEFAULTS["output_root"],
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SCRIPT_DEFAULTS["splits"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SCRIPT_DEFAULTS["batch_size"],
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=SCRIPT_DEFAULTS["tile_size"],
        help=(
            "Enable sliding-window extraction for samples larger than this "
            "input size. 0 disables tiling (default for PASTIS-R 128x128)."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=SCRIPT_DEFAULTS["tile_overlap"],
        help=(
            "Sliding-window overlap. Values in [0, 1) are interpreted as a "
            "ratio of tile_size; values >= 1 are interpreted as pixels."
        ),
    )
    parser.add_argument(
        "--device",
        default=SCRIPT_DEFAULTS["device"],
        help="Use 'auto' for cuda:LOCAL_RANK under torchrun.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp32"],
        default=SCRIPT_DEFAULTS["precision"],
        help="CUDA inference precision. bf16 enables autocast and TF32.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_DEFAULTS["skip_existing"],
        help=(
            "Reuse existing outputs when present. With --save-inputs, "
            "all per-modality input.tif files must also exist."
        ),
    )
    parser.add_argument(
        "--save-inputs",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_DEFAULTS["save_inputs"],
        help=(
            "Also save the pipeline input tensors as input_<modality>.tif for "
            "inspection. Time-series modalities are averaged over the time axis."
        ),
    )
    parser.add_argument(
        "--save-raw-inputs",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_DEFAULTS["save_raw_inputs"],
        help=(
            "Raw input export is not supported for the UniverSat PASTIS-R "
            "pipeline and will raise an error if enabled."
        ),
    )
    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_DEFAULTS["quiet"],
        help="Reduce per-batch logging.",
    )
    parser.add_argument(
        "--pipeline-key",
        default=SCRIPT_DEFAULTS["pipeline_key"],
        help=(
            "Pipeline to use for extraction. Use 'none' to keep each "
            "dataloader's configured pipeline."
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=SCRIPT_DEFAULTS["cfg_options"],
        help="Override config options, e.g. key=value.",
    )
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.utils import import_modules_from_strings

    ctx = _get_dist_context()
    device = _resolve_device(args.device, ctx)
    _init_distributed(ctx, device)
    _configure_cuda_fast_math(device)

    cfg = Config.fromfile(Path(args.config))
    cfg_options = _parse_cfg_options(args.cfg_options)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get("default_scope", "mmseg"))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_key = None if args.pipeline_key.lower() == "none" else args.pipeline_key

    if args.save_raw_inputs:
        parser.error(
            "--save-raw-inputs is not supported for the UniverSat PASTIS-R pipeline."
        )

    try:
        local_summaries = [
            _extract_split(
                cfg=cfg,
                split=split,
                output_root=output_root,
                batch_size=args.batch_size,
                device=device,
                pipeline_key=pipeline_key,
                ctx=ctx,
                precision=args.precision,
                skip_existing=args.skip_existing,
                save_inputs=args.save_inputs,
                save_raw_inputs=args.save_raw_inputs,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                verbose=not args.quiet,
            )
            for split in args.splits
        ]
        save_json(
            output_root / f"summary_rank{ctx.rank}.json",
            {
                "rank": ctx.rank,
                "local_rank": ctx.local_rank,
                "world_size": ctx.world_size,
                "device": str(device),
                "precision": args.precision,
                "batch_size": args.batch_size,
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
                "save_inputs": args.save_inputs,
                "save_raw_inputs": args.save_raw_inputs,
                "splits": local_summaries,
            },
        )
        _barrier(ctx)

        if ctx.rank == 0:
            if ctx.is_distributed:
                summaries = [
                    _merge_split_manifests(output_root, split, ctx.world_size)
                    for split in args.splits
                ]
            else:
                summaries = local_summaries
            save_json(
                output_root / "summary.json",
                {
                    "world_size": ctx.world_size,
                    "device": str(device),
                    "precision": args.precision,
                    "batch_size": args.batch_size,
                    "tile_size": args.tile_size,
                    "tile_overlap": args.tile_overlap,
                    "save_inputs": args.save_inputs,
                    "save_raw_inputs": args.save_raw_inputs,
                    "splits": summaries,
                },
            )
        _barrier(ctx)
    finally:
        _destroy_distributed(ctx)


if __name__ == "__main__":
    main()
