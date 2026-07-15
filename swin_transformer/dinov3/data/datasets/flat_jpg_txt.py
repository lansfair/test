# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random
import warnings
from pathlib import Path
from typing import Callable, Optional, Union
import os

from PIL import Image

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset
from tqdm import tqdm
import pickle

class FlatJPGTxt(ExtendedVisionDataset):
    """
    通过一个外部 txt 文件直接初始化数据集，无需递归扫描和校验。

    txt 文件每行一个图片路径（可以是绝对路径或相对路径），
    支持按 train/val 比例切分（使用 seed 保证可复现）。
    """

    def __init__(
        self,
        *,
        image_txt_dir: Union[str, Path] = None,          # 必须指定：包含图片路径列表的 txt 文件
        target_pkl: Union[str, Path] = None,
        seed: int = 42,
        train_ratio: float = 0.9,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        validate_images: bool = False,       # 是否对读取的图片进行有效性校验（默认关闭，信任 txt 内容）
    ) -> None:
        # 调用父类初始化（传入空 root，因为不需要基于 root 扫描）
        super().__init__(
            root="",
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        if target_pkl is not None:
            with open(target_pkl, 'rb') as fr:
                self._paths = pickle.load(fr)
        else:
            txt_paths = Path(image_txt_dir).glob("*.txt")
            pkl_paths = list(Path(image_txt_dir).glob("*.pkl")) 
            self._paths = []
            self.validate_images = validate_images 
            if pkl_paths:   # 列表为空则 False
                # print(f"找到 {len(pkl_paths)} 个 pkl 文件")
                for p in pkl_paths:
                    print(f'loading {str(p)}...')
                    with open(p, 'rb') as fr:
                        self._paths += pickle.load(fr)
                print(f'all images: {len(self._paths)}')
            else:
                for txt_path in txt_paths:
                    self._paths += self.read_one_txt(txt_path)
            rng = random.Random(seed)
            rng.shuffle(self._paths)
        
        # split_at = int(len(paths) * train_ratio)
        # if split.upper() == "TRAIN":
        #     self._paths = paths[:split_at]
        # else:
        #     self._paths = paths[split_at:]

        # print(f"Initialized {len(self._paths)} images for split '{split}' from {txt_path}")

    def get_image_data(self, index: int) -> bytes:
        """返回第 index 张图片的字节数据（用于解码）"""
        return self._paths[index].read_bytes()

    def get_target(self, index: int):
        """本数据集仅用于特征蒸馏，目标值统一为 0"""
        return 0

    def __len__(self) -> int:
        return len(self._paths)

    def read_one_txt(self, txt_path):
        # 读取 txt 文件中的每一行（去除空行和首尾空格）
        print(f"loading {str(txt_path)}")
        # txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"TXT file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            paths = [Path(line.strip()) for line in tqdm(f) if line.strip()]
        # paths = []
        # base_dir = txt_path.parent
        # for line in lines:
        #     p = Path(line)
        #     if not p.is_absolute():
        #         p = base_dir / p
        #     paths.append(p)

        # 可选：校验图片是否有效（可能耗时，默认关闭）
        if self.validate_images:
            print("Validating images ... (this may take a while)")
            valid_paths = []
            for p in paths:
                if self._valid_image(p):
                    valid_paths.append(p)
                else:
                    warnings.warn(f"Invalid or corrupted image skipped: {p}")
            paths = valid_paths

        # 排序并切分
        paths = sorted(paths)
        return paths

    @staticmethod
    def _valid_image(path: Path) -> bool:
        """检查图片是否可读且尺寸未超过 PIL 的警告阈值"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(path) as img:
                    w, h = img.size
                    return w * h <= 89478485  # 约 85MP，与 PIL 默认阈值一致
        except Exception:
            return False