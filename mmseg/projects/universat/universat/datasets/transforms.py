"""Loading and packing transforms for UniverSat multimodal inputs."""

from typing import Dict, Optional

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


def _load_array(path: str) -> np.ndarray:
    """Load a numpy array from ``path``.

    Supports ``.npy`` files. Extend this helper if your modalities are stored
    in a different format (e.g. GeoTIFF via rasterio/gdal).
    """
    if path.endswith(".npy"):
        return np.load(path)
    raise ValueError(
        f"Unsupported modality file format for {path}. "
        f"UniverSat transforms currently support .npy files."
    )


@TRANSFORMS.register_module()
class LoadMultimodalFromFile(BaseTransform):
    """Load each modality raster from ``modality_paths``.

    Required keys:
        - ``modality_paths`` (dict): modality_name -> file path.

    Added keys:
        - ``img`` (dict): modality_name -> float32 tensor.
        - ``img_shape`` / ``ori_shape``: shape of the first modality.
    """

    def __init__(self, modalities: Optional[list] = None):
        self.modalities = modalities

    def transform(self, results: dict) -> dict:
        modality_paths: Dict[str, str] = results["modality_paths"]
        modalities = self.modalities or list(modality_paths.keys())

        img = {}
        for mod in modalities:
            if mod not in modality_paths:
                raise KeyError(
                    f"Modality {mod!r} not found in modality_paths. "
                    f"Available: {list(modality_paths.keys())}"
                )
            array = _load_array(modality_paths[mod]).astype(np.float32)
            tensor = torch.from_numpy(array)
            # Ensure time-series modalities have shape (T, C, H, W).
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            img[mod] = tensor

        results["img"] = img
        # Use the first modality's spatial shape as the reference shape.
        first = next(iter(img.values()))
        results["img_shape"] = tuple(first.shape[-2:])
        results["ori_shape"] = tuple(first.shape[-2:])
        return results


@TRANSFORMS.register_module()
class NormalizeMultimodal(BaseTransform):
    """Normalize each modality independently.

    Args:
        mean: dict mapping modality_name -> list of per-channel means.
        std: dict mapping modality_name -> list of per-channel stds.
    """

    def __init__(
        self,
        mean: Dict[str, list],
        std: Dict[str, list],
    ):
        self.mean = {
            mod: torch.tensor(vals, dtype=torch.float32).view(-1, 1, 1)
            for mod, vals in mean.items()
        }
        self.std = {
            mod: torch.tensor(vals, dtype=torch.float32).view(-1, 1, 1)
            for mod, vals in std.items()
        }

    def transform(self, results: dict) -> dict:
        img = results["img"]
        for mod, tensor in img.items():
            mean = self.mean.get(mod)
            std = self.std.get(mod)
            if mean is None or std is None:
                continue
            # tensor shape is (T, C, H, W) or (1, C, H, W).
            if tensor.ndim == 4:
                m = mean.view(1, -1, 1, 1)
                s = std.view(1, -1, 1, 1)
            else:
                m = mean
                s = std
            img[mod] = (tensor - m) / s.clamp_min(1e-6)
        return results


@TRANSFORMS.register_module()
class PackUniverSatInputs(BaseTransform):
    """Pack multimodal ``img`` dict and seg map into MMSegmentation inputs.

    Required keys:
        - ``img`` (dict of tensors)
        - ``gt_seg_map`` (np.ndarray or tensor), optional

    Added keys:
        - ``inputs`` (dict of tensors)
        - ``data_samples`` (SegDataSample)
    """

    def __init__(
        self,
        meta_keys: tuple = (
            "img_path",
            "seg_map_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed = {}
        packed["inputs"] = results["img"]

        data_sample = SegDataSample()
        if "gt_seg_map" in results:
            gt_seg_map = results["gt_seg_map"]
            if isinstance(gt_seg_map, np.ndarray):
                gt_seg_map = torch.from_numpy(gt_seg_map)
            data_sample.gt_sem_seg = PixelData(data=gt_seg_map.to(torch.long))

        # Attach useful metadata.
        img_meta = {key: results.get(key, None) for key in self.meta_keys}
        data_sample.set_metainfo(img_meta)
        packed["data_samples"] = data_sample
        return packed
