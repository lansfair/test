# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
import warnings
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class FlatJPGDataset(ExtendedVisionDataset):
    Split = str

    def __init__(
        self,
        *,
        split: str,
        root: str,
        seed: int = 42,
        train_ratio: float = 0.9,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # modified by zhoujiwen: flat recursive JPG/PNG dataset for feature distillation.
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        # modified by zhoujiwen: 过滤超过PIL warning阈值的超大图和损坏图，避免训练日志刷屏和DataLoader崩溃。
        # paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and self._valid_image(p)]
        # 优先从 txt 文件加载图片路径列表
        root_path = Path(root)
        txt_path = root_path / "valid_image_paths.txt"
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                paths = [Path(line.strip()) for line in f if line.strip()]
        else:
            # 耗时操作：递归扫描并校验图片
            print("starting valid..")
            paths = [
                p for p in tqdm(root_path.rglob("*"))
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and self._valid_image(p)
            ]
            # 保存路径列表到 txt 文件（每行一个绝对路径）
            with open(txt_path, 'w', encoding='utf-8') as f: 
                for p in paths:
                    f.write(str(p.absolute()) + '\n')
        print("offline valide complete!!!")
        paths = sorted(paths)
        rng = random.Random(seed)
        rng.shuffle(paths)
        split_at = int(len(paths) * train_ratio)
        self._paths = paths[:split_at] if split.upper() == "TRAIN" else paths[split_at:]

    def get_image_data(self, index: int) -> bytes:
        return self._paths[index].read_bytes()

    @staticmethod
    def _valid_image(path: Path) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)  # modified by zhoujiwen: 尺寸探测阶段静默PIL超大图警告，超阈值图片仍会被过滤。
                with Image.open(path) as img: return img.size[0] * img.size[1] <= 89478485  # modified by zhoujiwen: 按PIL warning阈值过滤超大图，避免数据加载变慢。
        except Exception:
            return False  # modified by zhoujiwen: 损坏图片直接过滤，避免训练中断。

    def get_target(self, index: int):
        return 0

    def __len__(self) -> int:
        return len(self._paths)
