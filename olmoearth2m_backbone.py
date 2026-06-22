from __future__ import annotations

import re
from typing import Any

import torch
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from opencd.registry import MODELS
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from ..utils import build_olmoearth_model, get_modality_bands
from ..utils import get_sample_field


def _import_olmoearth_types():
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    return MaskedOlmoEarthSample, MaskValue, Modality


def _normalize_band_name(value: str) -> str:
    return str(value).strip().upper().replace("_", "").replace(" ", "")


@MODELS.register_module()
class OlmoEarth2mBackbone(BaseModule):
    """
    OLMoEarth backbone with pre-projection layer: 12 input channels -> 4 channels
    Conv1x1 + BN + ReLU before encoder, keep original OlmoEarth encoder logic unchanged.
    Input tensor shape: (B, 12*T, H, W) (your OSCD 12-band input after drop B10)
    After projection: internal feature becomes (B, 4*T, H, W) feed to encoder
    """

    def __init__(
        self,
        model_config_path: str,
        modality: str = "sentinel2_l2a",
        patch_size: int = 4,
        num_timesteps: int = 12,
        out_channels: int = 768,
        pooling_type: str = "mean",
        fast_pass: bool | None = None,
        init_cfg: dict | None = None,
        out_indices=(0, 1, 2, 3),
        # 新增：原始输入波段数 & 投影目标通道数
        raw_bands: int = 12,
        proj_target_bands: int = 4,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.modality = modality
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps
        self.out_channels = out_channels
        self.pooling_type = pooling_type
        self.fast_pass = fast_pass
        self.band_names = list(get_modality_bands(modality))
        self.sample_field = get_sample_field(modality)
        self.out_indices = out_indices
        self.raw_bands = raw_bands
        self.proj_target_bands = proj_target_bands

        # ========== 新增：12 -> 4 可学习投影块 Conv1x1 + BN + ReLU ==========
        # 单时相维度映射：单帧12波段映射为4波段
        self.proj_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.raw_bands,
                out_channels=self.proj_target_bands,
                kernel_size=1,
                bias=True
            ),
            nn.BatchNorm2d(self.proj_target_bands),
            nn.ReLU(inplace=True)
        )

        # 原始OlmoEarth模型与encoder不变
        self.model = build_olmoearth_model(model_config_path)
        self.encoder = self.model.encoder
        self.encoder.remove_masked_tokens = self._remove_masked_tokens_sort_compat
        self._batch_metainfo: list[dict[str, Any]] | None = None

    @staticmethod
    def _remove_masked_tokens_sort_compat(
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sortable_mask = mask
        if mask.dtype == torch.bool:
            sortable_mask = mask.to(torch.uint8)
        sorted_mask, indices = torch.sort(
            sortable_mask,
            dim=1,
            descending=True,
            stable=True,
        )
        sorted_mask = sorted_mask.to(torch.bool)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        seq_lengths = sorted_mask.sum(-1)
        max_length = seq_lengths.max()
        x = x[:, :max_length]
        updated_mask = sorted_mask[:, :max_length]
        return x, indices, updated_mask, seq_lengths, max_length

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "OLMoEarth init_cfg checkpoint must be a state_dict or "
                "contain 'state_dict'/'model'."
            )
        cleaned = {}
        for key, value in checkpoint.items():
            key = re.sub(r"^(module\.)+", "", key)
            key = re.sub(r"^(model\.)+", "", key)
            cleaned[key] = value
        return cleaned

    def init_weights(self) -> None:
        if self.init_cfg is None:
            return
        if not isinstance(self.init_cfg, dict):
            raise TypeError("OlmoEarth2mBackbone init_cfg must be a dict.")
        if self.init_cfg.get("type") != "Pretrained":
            super().init_weights()
            return
        checkpoint_path = self.init_cfg.get("checkpoint")
        if checkpoint_path is None:
            raise ValueError(
                "OlmoEarth2mBackbone init_cfg requires a checkpoint path."
            )
        checkpoint = CheckpointLoader.load_checkpoint(
            checkpoint_path,
            map_location="cpu",
            logger=None,
        )
        state_dict = self._extract_state_dict(checkpoint)
        # proj_block是新增层，strict=False跳过不匹配权重
        self.model.load_state_dict(state_dict, strict=False)
        self._is_init = True

    def set_batch_metainfo(
        self,
        batch_metainfo: list[dict[str, Any]] | None,
    ) -> None:
        self._batch_metainfo = batch_metainfo

    def _get_modality_enum(self):
        _, _, Modality = _import_olmoearth_types()
        return getattr(Modality, self.modality.upper(), None) or getattr(
            Modality, self.sample_field.upper()
        )

    def _get_bandsets(self) -> list[list[str]]:
        modality = self._get_modality_enum()
        for attr in ("band_sets", "bandsets", "band_groups"):
            if not hasattr(modality, attr):
                continue
            value = getattr(modality, attr)
            if value is None:
                continue
            resolved = []
            for group in value:
                if hasattr(group, "bands"):
                    resolved.append(
                        [_normalize_band_name(x) for x in group.bands]
                    )
                else:
                    resolved.append([_normalize_band_name(x) for x in group])
            return resolved
        return [[_normalize_band_name(band)] for band in self.band_names]

    def _default_timestamps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        timestamps = torch.tensor(
            [1, 1, 2025],
            dtype=torch.long,
            device=device,
        )
        return timestamps[None, None, :].repeat(
            batch_size,
            self.num_timesteps,
            1,
        )

    def _timestamps_from_metainfo(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if not self._batch_metainfo:
            return self._default_timestamps(batch_size, device)
        timestamps = []
        for meta in self._batch_metainfo:
            value = meta.get("timestamps")
            if value is None:
                timestamps.append(
                    self._default_timestamps(1, device).squeeze(0)
                )
                continue
            tensor = torch.as_tensor(value, dtype=torch.long, device=device)
            if tensor.ndim != 2 or tensor.shape[-1] != 3:
                raise ValueError(
                    "timestamps must have shape (T, 3), "
                    f"got {tuple(tensor.shape)}"
                )
            timestamps.append(tensor)
        return torch.stack(timestamps, dim=0)

    def _present_bands_from_metainfo(self, batch_size: int) -> list[set[str]]:
        if not self._batch_metainfo:
            all_bands = {
                _normalize_band_name(band)
                for band in self.band_names
            }
            return [all_bands for _ in range(batch_size)]
        out = []
        for meta in self._batch_metainfo:
            present = meta.get("present_bands") or self.band_names
            out.append({_normalize_band_name(band) for band in present})
        return out

    def _build_bandset_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        _, MaskValue, _ = _import_olmoearth_types()
        bandsets = self._get_bandsets()
        present_by_sample = self._present_bands_from_metainfo(batch_size)
        mask = torch.full(
            (batch_size, height, width, self.num_timesteps, len(bandsets)),
            float(MaskValue.MISSING.value),
            dtype=torch.float32,
            device=device,
        )
        for sample_idx, present in enumerate(present_by_sample):
            for bandset_idx, bandset in enumerate(bandsets):
                if any(band in present for band in bandset):
                    mask[sample_idx, :, :, :, bandset_idx] = float(
                        MaskValue.ONLINE_ENCODER.value
                    )
        return mask

    def _has_missing_tokens(self, sample) -> bool:
        _, MaskValue, _ = _import_olmoearth_types()
        for name, value in sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if (value == MaskValue.MISSING.value).any():
                    return True
        return False

    def _make_sample(self, inputs: Tensor):
        MaskedOlmoEarthSample, _, _ = _import_olmoearth_types()
        batch_size, channels, height, width = inputs.shape
        num_bands = self.proj_target_bands
        expected_channels = num_bands * self.num_timesteps
        if channels != expected_channels:
            raise ValueError(
                f"After projection expected {expected_channels} channels "
                f"({num_bands} proj bands x {self.num_timesteps} timesteps), "
                f"got {channels}"
            )
        image = inputs.reshape(
            batch_size,
            num_bands,
            self.num_timesteps,
            height,
            width,
        )
        image = image.permute(0, 3, 4, 2, 1).contiguous()
        bandset_mask = self._build_bandset_mask(
            batch_size, height, width, inputs.device
        )
        kwargs = {
            self.sample_field: image,
            f"{self.sample_field}_mask": bandset_mask,
            "timestamps": self._timestamps_from_metainfo(
                batch_size,
                inputs.device,
            ),
        }
        return MaskedOlmoEarthSample(**kwargs)

    def forward(self, inputs: Tensor) -> tuple[Tensor]:
        """
        Step1: Split time dimension, each timestep is 12 raw bands
        Step2: Apply Conv1x1(12→4)+BN+ReLU per timestep
        Step3: Concat all timesteps back to (B,4*T,H,W) feed original encoder flow
        """
        from olmoearth_pretrain.nn.pooling import (
            PoolingType,
            pool_unmasked_tokens,
        )
        B, C_total, H, W = inputs.shape
        T = self.num_timesteps
        assert C_total == self.raw_bands * T, \
            f"Input total channels must be raw_bands*T = {self.raw_bands}*{T}, get {C_total}"

        # 拆分时序维度：(B, T, 12, H, W)
        x = inputs.view(B, T, self.raw_bands, H, W)
        # 对每个时相单独做12→4投影
        proj_list = []
        for t_idx in range(T):
            x_t = x[:, t_idx, :, :, :]  # (B,12,H,W)
            x_t_proj = self.proj_block(x_t)  # (B,4,H,W)
            proj_list.append(x_t_proj)
        # 重新拼接时序通道维度 (B, T*4, H, W)
        x_proj_all = torch.cat(proj_list, dim=1)

        # 后续完全复用原始流程，送入encoder
        sample = self._make_sample(x_proj_all)
        print(sample.sentinel2_l2a_mask.unique())
        fast_pass = self.fast_pass
        if fast_pass is None:
            fast_pass = not self._has_missing_tokens(sample)
        encoder_out = self.encoder(
            sample,
            fast_pass=fast_pass,
            patch_size=self.patch_size,
        )
        tokens_and_masks = encoder_out["tokens_and_masks"]
        pooled = pool_unmasked_tokens(
            tokens_and_masks,
            PoolingType(self.pooling_type),
            spatial_pooling=True,
            concat_features=False,
        )
        feat = pooled.permute(0, 3, 1, 2).contiguous()  # (B, 768, H, W)

        # 多尺度下采样输出不变
        x1 = feat  # 1x
        x2 = F.avg_pool2d(x1, 2)  # 1/2
        x3 = F.avg_pool2d(x2, 2)  # 1/4
        x4 = F.avg_pool2d(x3, 2)  # 1/8

        outs = [x1, x2, x3, x4]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)