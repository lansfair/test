"""
OlmoEarth 光学+SAR 变化检测模型 v3

架构（v3 关键改进 — 通道 Padding 替代通道切片）：
  前光 RGB (3ch) ──→ pad_to_12ch ──→ PretrainedPatchEmbed(全12ch权重) ──┐
                                                                           ├── OlmoEarth ViT Blocks ──→ diff ──→ UperNet
  后时 SAR (1ch) ──→ pad_to_2ch  ──→ PretrainedPatchEmbed(全 2ch权重) ───┘

v3 vs v2:
  v2: 通道切片 (12ch→3ch) — 破坏波段语义，F1 天花板 ~0.48
  v3: 通道 Padding (3ch→12ch, 填零) — 保留完整波段语义，预期 F1 0.55~0.60

S2_L2A 12 波段映射 (OlmoEarth v1-Base band_order: B02,B03,B04,B08,...):
  idx0=B02(Blue) idx1=B03(Green) idx2=B04(Red) idx3=B08(NIR) ...
  BRIGHT RGB → S2 B04(Red, idx2) B03(Green, idx1) B02(Blue, idx0)
  S1_GRD 2 波段: VV(idx0) VH(idx1)
  BRIGHT SAR → S1 VV(idx0)
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ Channel Padding ============

# S2_L2A 12 波段 → BRIGHT RGB 映射
# OlmoEarth v1-Base band_order: B02:0,B03:1,B04:2,B08:3,... → R=B04(idx2) G=B03(idx1) B=B02(idx0)
S2_RGB_INDICES = [2, 1, 0]  # R→idx2, G→idx1, B→idx0


def pad_optical_3to12(rgb):
    """3ch RGB → 12ch S2_L2A（填充到正确波段位置，其余填零）

    BRIGHT optical: [B, 3, H, W] in RGB order
    → S2_L2A:        [B, 12, H, W]
      B4(idx2) ← R, B3(idx1) ← G, B2(idx0) ← B, 其余波段=0
    """
    B, _, H, W = rgb.shape
    padded = torch.zeros(B, 12, H, W, device=rgb.device, dtype=rgb.dtype)
    padded[:, 2, :, :] = rgb[:, 0, :, :]  # R → B04
    padded[:, 1, :, :] = rgb[:, 1, :, :]  # G → B03
    padded[:, 0, :, :] = rgb[:, 2, :, :]  # B → B02
    return padded


def pad_sar_1to2(sar):
    """1ch SAR → 2ch S1_GRD（VV 填 idx0，VH 填零）

    BRIGHT SAR: [B, 1, H, W]
    → S1_GRD:    [B, 2, H, W]
      VV(idx0) ← SAR, VH(idx1)=0
    """
    B, _, H, W = sar.shape
    padded = torch.zeros(B, 2, H, W, device=sar.device, dtype=sar.dtype)
    padded[:, 0, :, :] = sar[:, 0, :, :]
    return padded


# ============ 预训练 Patch Embedding（v3: 完整权重，不切片） ============

def load_pretrained_patch_embeddings(weight_path):
    """从 OlmoEarth v1-Base checkpoint 提取 Conv2d patch embedding 权重。

    v1-Base 的 FlexiPatchEmbed 用 nn.Conv2d（非 Linear pixel_proj）：
      S2_L2A 分 3 个波段组（band_order 中连续）：
        __0 (768,4,8,8): [B02,B03,B04,B08]         = idx[0:4]
        __1 (768,6,8,8): [B05,B06,B07,B8A,B11,B12] = idx[4:10]
        __2 (768,2,8,8): [B01,B09]                 = idx[10:12]
      沿通道维拼接为单个 (768,12,8,8) Conv2d（等价于三组线性叠加 → 1 token/patch）。
      S1_GRD 单组 (768,2,8,8)。

    Returns:
        opt_weights: (conv_w[768,12,8,8], conv_b[768])
        sar_weights: (conv_w[768,2,8,8],  conv_b[768])
    """
    state = torch.load(weight_path, map_location='cpu', weights_only=True)

    base = 'encoder.patch_embeddings.per_modality_embeddings'

    def _conv(prefix):
        return state[f'{prefix}.proj.weight'].clone(), state[f'{prefix}.proj.bias'].clone()

    # S2: 三个波段组按通道维拼接（band_order 中连续: [0:4],[4:10],[10:12]）
    w0, b0 = _conv(f'{base}.sentinel2_l2a.sentinel2_l2a__0')
    w1, b1 = _conv(f'{base}.sentinel2_l2a.sentinel2_l2a__1')
    w2, b2 = _conv(f'{base}.sentinel2_l2a.sentinel2_l2a__2')
    s2_w = torch.cat([w0, w1, w2], dim=1)   # (768, 12, 8, 8)
    s2_b = b0 + b1 + b2                      # (768,) 偏置叠加

    # S1: 单组
    s1_w, s1_b = _conv(f'{base}.sentinel1.sentinel1__0')

    print(f"[PatchEmbed Conv2d] S2_L2A merged {list(s2_w.shape)} (3 bandsets -> 1 token/patch)")
    print(f"[PatchEmbed Conv2d] S1_GRD {list(s1_w.shape)}")

    del state
    return (s2_w, s2_b), (s1_w, s1_b)


class PretrainedPatchEmbed(nn.Module):
    """OlmoEarth v1-Base 预训练 patch embedding（Conv2d 版）。
    输入 [B, C, H, W] → 输出 [B, N, embed_dim]，N = (H/p) * (W/p)。
    """

    def __init__(self, in_chans, embed_dim, patch_size, conv_w, conv_b):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        assert tuple(self.proj.weight.shape) == tuple(conv_w.shape), \
            f"Conv2d weight {tuple(self.proj.weight.shape)} != pretrained {tuple(conv_w.shape)}"
        self.proj.weight.data = conv_w
        self.proj.bias.data = conv_b

    def forward(self, x):
        x = self.proj(x)                     # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)     # [B, N, embed_dim]
        return x


# ============ ViT Blocks 提取 ============

def load_olmoearth_vit_blocks(config_path, weight_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    enc_cfg = config_dict['model']['encoder_config']

    embed_dim = enc_cfg['embedding_size']
    depth = enc_cfg['depth']
    num_heads = enc_cfg['num_heads']
    mlp_ratio = enc_cfg['mlp_ratio']
    drop_path = enc_cfg.get('drop_path', 0.0)
    qk_norm = enc_cfg.get('qk_norm', False)

    from olmoearth_pretrain.nn.attention import Block
    blocks = nn.ModuleList([
        Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            use_flash_attn=False,
            qk_norm=qk_norm,
        )
        for _ in range(depth)
    ])
    norm = nn.LayerNorm(embed_dim)

    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    block_state = {}
    norm_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.blocks.'):
            block_state[k[len('encoder.blocks.'):]] = v
        elif k.startswith('encoder.norm.'):
            norm_state[k[len('encoder.norm.'):]] = v

    blocks.load_state_dict(block_state, strict=True)
    norm.load_state_dict(norm_state, strict=True)
    print(f"[ViT Blocks] Loaded {depth} blocks, embed_dim={embed_dim}")

    for p in blocks.parameters():
        p.requires_grad = False
    for p in norm.parameters():
        p.requires_grad = False

    return blocks, norm, embed_dim


# ============ CD Model v3 ============

class OlmoEarthCD(nn.Module):
    """OlmoEarth 光学+SAR 变化检测模型 v3 — 通道 Padding 方案。"""

    def __init__(self, config_path, weight_path, num_classes=4, patch_size=8, img_size=224,
                 finetune=False):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.blocks, self.norm, self.embed_dim = load_olmoearth_vit_blocks(config_path, weight_path)
        if not finetune:
            for blk in self.blocks:
                for p in blk.parameters():
                    p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False

        # v3: 完整波段 patch embedding（不切片！）
        opt_w, sar_w = load_pretrained_patch_embeddings(weight_path)
        self.optical_embed = PretrainedPatchEmbed(12, self.embed_dim, patch_size, *opt_w)
        self.sar_embed     = PretrainedPatchEmbed(2,  self.embed_dim, patch_size, *sar_w)

        self.optical_modality_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.sar_modality_token     = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        from models.cd_head import CDHead
        self.decoder = CDHead(
            in_channels=[self.embed_dim] * 4,
            channels=512,
            num_classes=num_classes,
            decoder_type='upernet',
        )

    def _extract_multiscale_features(self, tokens):
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        return [
            F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False),
            F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False),
            F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False),
            F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False),
        ]

    def forward(self, pre_optical, post_sar):
        # v3: 通道 Padding 后再送入 embed
        pre_optical_12ch = pad_optical_3to12(pre_optical)  # 3ch → 12ch
        post_sar_2ch     = pad_sar_1to2(post_sar)          # 1ch → 2ch

        opt_tokens = self.optical_embed(pre_optical_12ch)
        sar_tokens = self.sar_embed(post_sar_2ch)

        opt_tokens = opt_tokens + self.optical_modality_token
        sar_tokens = sar_tokens + self.sar_modality_token

        all_tokens = torch.cat([opt_tokens, sar_tokens], dim=1)
        N = opt_tokens.shape[1]
        pos_embed = torch.cat([self.pos_embed, self.pos_embed], dim=1)
        all_tokens = all_tokens + pos_embed

        for blk in self.blocks:
            all_tokens = blk(all_tokens)
        all_tokens = self.norm(all_tokens)

        opt_out = all_tokens[:, :N, :]
        sar_out = all_tokens[:, N:, :]
        diff_tokens = torch.abs(opt_out - sar_out)

        feats = self._extract_multiscale_features(diff_tokens)
        return self.decoder(feats)


def build_olmoearth_cd(
    config_path=None,
    weight_path=None,
    num_classes=4,
    patch_size=8,
    img_size=224,
    finetune=False,
):
    """工厂函数 v3。"""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if config_path is None:
        config_path = os.path.join(project_dir, 'weights', 'config.json')
    if weight_path is None:
        weight_path = os.path.join(project_dir, 'weights', 'weights.pth')

    model = OlmoEarthCD(
        config_path=config_path,
        weight_path=weight_path,
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        finetune=finetune,
    ).cuda()
    return model
