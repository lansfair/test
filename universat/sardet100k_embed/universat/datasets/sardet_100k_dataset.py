from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SARDet100kDataset(BaseSegDataset):
    """SARDet-100k dataset that loads COCO-format annotations for OLMoEarth
    embedding extraction.

    COCO JSON annotations are read to extract the image list; bounding-box
    annotations are ignored. Dummy zero segmentation labels are generated
    later in the pipeline via ``GenerateDummySegMap``.
    """

    METAINFO: dict[str, Any] = {
        "classes": (),
    }

    def __init__(
        self,
        data_root: str | Path,
        ann_file: str | Path,
        data_prefix: dict[str, str] | str = "",
        pipeline: list[dict[str, Any]] | None = None,
        test_mode: bool = False,
        lazy_init: bool = False,
        serialize_data: bool = True,
        max_refetch: int = 1000,
        **kwargs,
    ) -> None:
        self._coco_ann_file = ann_file
        super().__init__(
            data_root=str(data_root),
            ann_file=str(ann_file),
            data_prefix=data_prefix,
            pipeline=pipeline or [],
            test_mode=test_mode,
            lazy_init=lazy_init,
            serialize_data=serialize_data,
            max_refetch=max_refetch,
            **kwargs,
        )

    def load_data_list(self) -> list[dict[str, Any]]:
        ann_path = Path(self.data_root) / self._coco_ann_file
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        if not isinstance(images, list):
            raise TypeError(
                f"COCO JSON 'images' must be a list, got {type(images)}"
            )

        data_list: list[dict[str, Any]] = []
        img_prefix = (
            self.data_prefix.get("img", "")
            if isinstance(self.data_prefix, dict)
            else str(self.data_prefix)
        )

        for img_info in images:
            file_name = img_info["file_name"]
            img_path = (
                str(Path(img_prefix) / file_name)
                if img_prefix
                else file_name
            )
            data_list.append(
                {
                    "img_path": img_path,
                    "sample_id": Path(file_name).stem,
                    "dataset_name": "sardet_100k",
                    "ori_shape": (
                        img_info["height"],
                        img_info["width"],
                    ),
                }
            )

        return data_list
