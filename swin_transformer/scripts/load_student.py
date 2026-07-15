import torch
from pathlib import Path
import os.path as osp

PT_PATH = '/mnt/si000523ygkv/00-model/dinov3-distill-outputs/swin_base_vitl16_ssl_feature_distill_GE+IN22k+ZJSlice1024_16nodes_nowarmup_lowlr/ckpt/30999/merged_weights.pt'
# 加载原始 checkpoint（请替换为实际路径）
checkpoint = torch.load(PT_PATH, map_location='cpu')

# 提取并重命名
new_state_dict = {}
for key, value in checkpoint.items():
    if key.startswith('model.student.backbone.'):
        new_key = key[len('model.student.backbone.'):]  # 去掉前缀
        new_state_dict[new_key] = value

# 保存新 state_dict（可选）
torch.save(new_state_dict, Path(osp.dirname(PT_PATH))/'swintransformer-huge.pt')