"""DIOR dataset wrapper for UniverSat embedding extraction.

DIOR is a horizontal/oriented object detection dataset. This module converts
its bounding-box XML annotations into a semantic mask so that the existing
UniverSat MMSegmentation extraction pipeline can process DIOR images and save
one dense embedding GeoTIFF per sample.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np

from mmseg.registry import DATASETS

from .universat_dataset import UniverSatSegDataset


# Standard 20 DIOR categories. The exact casing follows the OLMoEarth DIOR
# config; matching is normalized (lower-case, no hyphen/underscore/space).
DIOR_CLASSES = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)


def _normalize_class_name(name: str) -> str:
    """Normalize a class name for robust matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _parse_dior_bboxes(
    xml_path: str,
    cat2label: dict[str, int],
) -> Tuple[List[List[int]], List[int]]:
    """Parse horizontal bounding boxes and labels from a DIOR XML file.

    Args:
        xml_path: Path to the DIOR Pascal-VOC-style annotation XML.
        cat2label: Mapping from normalized class name to zero-based label.

    Returns:
        Tuple of (bboxes, labels) where each bbox is ``[xmin, ymin, xmax, ymax]``
        in pixel coordinates and labels are zero-based class indices.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes: List[List[int]] = []
    labels: List[int] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or name_node.text is None:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
        except (AttributeError, TypeError, ValueError):
            continue

        normalized = _normalize_class_name(name_node.text)
        if normalized not in cat2label:
            raise ValueError(
                f"Unknown DIOR class '{name_node.text}' in {xml_path}. "
                f"Known classes: {list(cat2label.keys())}"
            )

        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(cat2label[normalized])

    return bboxes, labels


def _bbox_to_semantic_mask(
    height: int,
    width: int,
    bboxes: List[List[int]],
    labels: List[int],
) -> np.ndarray:
    """Create a semantic mask by filling each bbox with ``label + 1``.

    Background is ``0``. The ``+1`` offset keeps ``0`` reserved for background
    while class labels remain zero-based in the category mapping.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = label + 1
    return mask


def _image_size_from_xml_or_image(xml_path: str, img_path: str) -> Tuple[int, int]:
    """Return ``(height, width)`` from XML ``<size>`` or by loading the image."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is not None:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            return height, width
    except Exception:
        pass

    from PIL import Image

    with Image.open(img_path) as img:
        return img.height, img.width


@DATASETS.register_module()
class UniverSatDIORDataset(UniverSatSegDataset):
    """UniverSat-compatible DIOR dataset that exposes bbox annotations as masks.

    Expected DIOR layout::

        ${DATA_ROOT}/
          JPEGImages/
            00001.jpg
            ...
          Annotations/
            00001.xml
            ...
          ImageSets/Main/
            train.txt
            val.txt
            test.txt

    Each split file is a plain list of image ids (one per line, without the
    ``.jpg`` extension).

    Args:
        modalities: List of modality names. DIOR is RGB-only, so this should
            be ``['rgb']`` or a registered RGB-like modality name.
        split: Path to a DIOR ``ImageSets/Main/{split}.txt`` file. Relative
            paths are resolved against ``data_root``.
        img_subdir: Subdirectory containing the JPEG images.
        ann_subdir: Subdirectory containing the XML annotations.
        classes: Tuple of class names. Defaults to the 20 DIOR classes.
        bbox_to_mask: Whether to convert bboxes to a filled semantic mask.
            Set to ``False`` to extract embeddings without labels.
        *args, **kwargs: Forwarded to ``UniverSatSegDataset`` / ``BaseSegDataset``.
    """

    def __init__(
        self,
        img_subdir: str = "JPEGImages",
        ann_subdir: str = "Annotations",
        classes: Tuple[str, ...] = DIOR_CLASSES,
        bbox_to_mask: bool = True,
        *args,
        **kwargs,
    ):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.dior_classes = classes
        self.cat2label = {
            _normalize_class_name(cls): idx for idx, cls in enumerate(classes)
        }
        self.bbox_to_mask = bbox_to_mask
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        """Load the image id list from the DIOR split file."""
        split_path = self.split
        if self.data_root is not None:
            split_path = os.path.join(self.data_root, split_path)

        with open(split_path, "r", encoding="utf-8") as f:
            img_ids = [line.strip() for line in f if line.strip()]

        data_list = []
        for img_id in img_ids:
            img_path = os.path.join(
                self.data_root, self.img_subdir, f"{img_id}.jpg"
            )
            xml_path = os.path.join(
                self.data_root, self.ann_subdir, f"{img_id}.xml"
            )
            modality_paths = {mod: img_path for mod in self.modalities}
            data_list.append(
                {
                    "sample_id": img_id,
                    "img_path": img_path,
                    "modality_paths": modality_paths,
                    "xml_path": xml_path,
                    "seg_map_path": xml_path,
                    "height": 0,
                    "width": 0,
                }
            )
        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Return the raw sample info plus a bbox-derived semantic mask."""
        info = super().get_data_info(idx)
        xml_path = info.get("xml_path")
        img_path = info["modality_paths"][self.modalities[0]]

        height, width = _image_size_from_xml_or_image(xml_path, img_path)
        info["height"] = height
        info["width"] = width
        info["ori_shape"] = (height, width)
        info["img_shape"] = (height, width)

        if self.bbox_to_mask and os.path.exists(xml_path):
            bboxes, labels = _parse_dior_bboxes(xml_path, self.cat2label)
            info["gt_seg_map"] = _bbox_to_semantic_mask(height, width, bboxes, labels)

        return info
