"""
olmoearth_cd — 服务器多卡版
OlmoEarth v1.1-Base 光学+SAR 双端变化检测 (BRIGHT 4类)
支持单卡 / 多卡 DDP 自适应

用法:
  单卡: python train.py
  多卡: torchrun --nproc_per_node=4 train.py
  改参数: 编辑下方 CONFIG 字典即可
"""

import os, sys, argparse, logging, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from models.olmoearth_cd import build_olmoearth_cd
from tools.metrics import CDMetrics


# ============================================================
# 🎛️ CONFIG — 在这里改参数，启动命令永远不变
# ============================================================
CONFIG = {
    'data_root':   '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/DFC2025 BRIGHT',           # 数据集路径
    'config_path': 'weights/config.json',              # OlmoEarth config.json
    'weight_path': 'weights/weights.pth',             # OlmoEarth weights.pth
    'num_classes': 4,
    'batch_size':  4,                                  # 每张卡的 batch
    'epochs':      100,
    'lr':          1e-4,
    'img_size':    224,
    'patch_size':  8,                                  # 必须 8 (pretrained proj = 64*8*8)
    'finetune':    True,                               # 全参微调 (差分学习率)
    'class_weight': None,                              # 如 [0.5, 2.0, 2.0, 3.0]
    'num_workers': 0,
    'resume':      None,                               # 断点续跑
    'run_name':    None,                               # None = 自动时间戳
}
# ============================================================


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('olmoearth_cd')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), encoding='utf-8')
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ========== 数据集 ==========
class BrightCDDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, img_size=224, is_train=True):
        self.root = root
        self.is_train = is_train

        # 灵活解析 split 文件: 兼容 {root}/splits/{split}.txt 与 {root}/{split}_set.txt
        split_candidates = [
            os.path.join(root, 'splits', f'{split}.txt'),
            os.path.join(root, f'{split}_set.txt'),
        ]
        split_file = next((p for p in split_candidates if os.path.exists(p)), None)
        if split_file is None:
            raise FileNotFoundError(f'找不到 split 文件，尝试过: {split_candidates}')
        with open(split_file, 'r') as f:
            self.ids = [l.strip() for l in f if l.strip()]

        # 灵活解析图像/标签目录: 兼容 pre-event 与 pre-event_xxx 等变体
        self.pre_dir = self._resolve_subdir(root, 'pre-event')
        self.post_dir = os.path.join(root, 'post-event')
        self.target_dir = os.path.join(root, 'target')

        # 归一化 (与 tools/plot_predictions.py 反归一化保持一致): uint8→float32
        opt_norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        sar_norm = A.Normalize(mean=(0.5,), std=(0.25,))

        if is_train:
            self.optical_transform = A.Compose([
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                opt_norm, ToTensorV2(),
            ])
            self.sar_transform = A.Compose([
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                sar_norm, ToTensorV2(),
            ])
        else:
            self.optical_transform = A.Compose([
                A.CenterCrop(height=img_size, width=img_size),
                opt_norm, ToTensorV2(),
            ])
            self.sar_transform = A.Compose([
                A.CenterCrop(height=img_size, width=img_size),
                sar_norm, ToTensorV2(),
            ])

        import tifffile
        self._load_tiff_rgb = lambda p: tifffile.imread(p)[:, :, :3].transpose(2, 0, 1)
        self._load_tiff = tifffile.imread
        self._np = np

    @staticmethod
    def _resolve_subdir(root, prefix):
        """解析目录名: 优先精确匹配 prefix，否则取 prefix* 中名字最短的目录。"""
        import glob
        exact = os.path.join(root, prefix)
        if os.path.isdir(exact):
            return exact
        matches = sorted(
            (d for d in glob.glob(os.path.join(root, prefix + '*')) if os.path.isdir(d)),
            key=len,
        )
        if not matches:
            raise FileNotFoundError(f'在 {root} 下找不到 {prefix}* 目录')
        return matches[0]

    def _load_sar(self, path):
        img = Image.open(path)
        arr = np.array(img)
        if len(arr.shape) == 3:
            arr = arr[:, :, 0]
        return arr

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        pre = self._load_tiff_rgb(os.path.join(self.pre_dir, f'{sample_id}_pre_disaster.tif'))
        pre = pre.transpose(1, 2, 0)
        sar = self._load_sar(os.path.join(self.post_dir, f'{sample_id}_post_disaster.tif'))
        label = np.array(Image.open(os.path.join(self.target_dir, f'{sample_id}_building_damage.tif')))
        label[label == 255] = 4
        label = label.astype(np.int64)

        aug_opt = self.optical_transform(image=pre, mask=label)
        aug_sar = self.sar_transform(image=sar)
        return aug_opt['image'], aug_sar['image'], aug_opt['mask'].long()

    def __len__(self):
        return len(self.ids)


def create_loader(root_dir, split, batch_size, img_size=224, is_train=True,
                  num_workers=0, sampler=None):
    ds = BrightCDDataset(root_dir, split, img_size, is_train)
    shuffle = is_train and sampler is None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=is_train)


def train_epoch(model, loader, criterion, optimizer, device, epoch, rank):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'E{epoch} Train', leave=False, disable=(rank != 0))
    for pre_opt, post_sar, label in pbar:
        pre_opt = pre_opt.to(device)
        post_sar = post_sar.to(device)
        label = label.to(device).long()

        optimizer.zero_grad()
        out = model(pre_opt, post_sar)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if rank == 0:
            pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def val_epoch(model, loader, device, num_classes, rank):
    model.eval()
    metrics = CDMetrics(num_classes=num_classes)
    pbar = tqdm(loader, desc='Val', leave=False, disable=(rank != 0))
    for pre_opt, post_sar, label in pbar:
        pre_opt = pre_opt.to(device)
        post_sar = post_sar.to(device)
        out = model(pre_opt, post_sar)
        pred = out.argmax(dim=1)
        metrics.update(pred, label.to(device).long())
    return metrics.compute()


def main():
    # —— DDP 初始化 ——
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_ddp = local_rank >= 0

    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    else:
        rank = 0

    device = torch.device(f'cuda:{local_rank}' if is_ddp else 'cuda')

    # —— 参数合并 ——
    parser = argparse.ArgumentParser()
    for k, v in CONFIG.items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f'--{k.replace("_", "-")}', type=lambda x: x.lower() in ('true','1','yes'), default=v)
        elif t == list:
            parser.add_argument(f'--{k.replace("_", "-")}', type=float, nargs='+', default=v)
        else:
            parser.add_argument(f'--{k.replace("_", "-")}', type=t, default=v)
    args = parser.parse_args()

    # —— 时间戳目录 (rank 0 only) ——
    run_name = args.run_name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(proj_dir, 'outputs', run_name)
    log_dir = os.path.join(proj_dir, 'logs', run_name)

    logger = None
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logging(log_dir)

    if rank == 0:
        logger.info(f"DDP={'ON' if is_ddp else 'OFF'} | World={world_size}")
        logger.info(f"Data: {args.data_root} | BS={args.batch_size}x{world_size}={args.batch_size*world_size}")
        logger.info(f"Finetune={args.finetune} | Epochs={args.epochs} | Output: {run_dir}")

    # —— 模型 ——
    model = build_olmoearth_cd(
        config_path=args.config_path, weight_path=args.weight_path,
        num_classes=args.num_classes, patch_size=args.patch_size,
        img_size=args.img_size, finetune=args.finetune,
    ).to(device)

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)

    # —— 数据 ——
    train_dataset = BrightCDDataset(args.data_root, 'train', args.img_size, is_train=True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = create_loader(args.data_root, 'val', args.batch_size,
                                args.img_size, is_train=False, num_workers=args.num_workers)

    # —— 损失 & 优化器 ——
    if args.class_weight:
        cw = torch.tensor(args.class_weight, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, ignore_index=255)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    if args.finetune:
        m = model.module if is_ddp else model
        vit_params = [p for p in m.blocks.parameters()] + [p for p in m.norm.parameters()]
        embed_params = list(m.optical_embed.parameters()) + list(m.sar_embed.parameters()) + \
                       [m.optical_modality_token, m.sar_modality_token, m.pos_embed]
        decoder_params = list(m.decoder.parameters())
        param_groups = [
            {'params': vit_params, 'lr': 5e-6},
            {'params': embed_params, 'lr': 5e-5},
            {'params': decoder_params, 'lr': 1e-4},
        ]
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    start_epoch = 0

    # —— 断点续跑 ——
    if args.resume:
        ckpt_path = os.path.join(args.resume, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            (model.module if is_ddp else model).load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch']
            best_f1 = ckpt.get('f1', 0)
            if rank == 0:
                logger.info(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # —— 训练循环 ——
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, rank)
        result = val_epoch(model, val_loader, device, args.num_classes, rank)
        scheduler.step()

        if rank == 0:
            f1 = result['f1']
            iou = result['iou']
            p = result['precision']
            r = result['recall']

            logger.info(
                f"Epoch {epoch+1:3d} | Train Loss {train_loss:.4f} | "
                f"Val F1 {f1:.4f} | Val IoU {iou:.4f} | Val P {p:.4f} | Val R {r:.4f}"
            )

            # 追加一行指标到 metrics.txt (供出图; parse 按 epoch 去重, 续跑安全)
            with open(os.path.join(log_dir, 'metrics.txt'), 'a') as mf:
                mf.write(f"{epoch+1} {train_loss:.6f} {f1:.6f} {iou:.6f} {p:.6f} {r:.6f}\n")

            model_state = model.module.state_dict() if is_ddp else model.state_dict()
            ckpt = {'epoch': epoch + 1, 'model': model_state,
                    'optimizer': optimizer.state_dict(), 'f1': f1}
            if (epoch + 1) % 10 == 0:
                torch.save(ckpt, os.path.join(run_dir, f'checkpoint_epoch_{epoch+1}.pth'))

            if f1 > best_f1:
                best_f1 = f1
                torch.save(ckpt, os.path.join(run_dir, 'best_model.pth'))
                logger.info(f"  -> Best F1: {f1:.4f} [saved]")

    if is_ddp:
        dist.barrier()

    # —— 测试 + 出图 (rank 0 only) ——
    if rank == 0:
        elapsed = time.time() - t0
        logger.info(f"Training done in {elapsed/3600:.1f}h. Best Val F1: {best_f1:.4f}")

        # 推理/出图用解包后的模型 (best_model.pth 存的是无 module. 前缀的 state_dict)
        infer_model = model.module if is_ddp else model

        logger.info("=== Testing with best model ===")
        best = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device, weights_only=True)
        infer_model.load_state_dict(best['model'])
        test_loader = create_loader(args.data_root, 'test', args.batch_size,
                                     args.img_size, is_train=False)
        test_result = val_epoch(infer_model, test_loader, device, args.num_classes, rank)
        logger.info(
            f"Test F1 {test_result['f1']:.4f} | IoU {test_result['iou']:.4f} | "
            f"P {test_result['precision']:.4f} | R {test_result['recall']:.4f}"
        )

        # 出图
        logger.info("Auto-generating curves & predictions...")
        fig_dir = os.path.join(run_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        try:
            from tools.plot_curves import plot_from_metrics
            metrics_file = os.path.join(log_dir, 'metrics.txt')
            if os.path.exists(metrics_file):
                plot_from_metrics(metrics_file, fig_dir)
                logger.info(f"Training curves saved to: {fig_dir}")
        except Exception as e:
            logger.warning(f"Curve generation failed: {e}")

        try:
            from tools.plot_predictions import visualize_predictions
            visualize_predictions(infer_model, args.data_root, fig_dir, device=device, img_size=args.img_size)
            logger.info(f"Predictions saved to: {fig_dir}")
        except Exception as e:
            logger.warning(f"Prediction generation failed: {e}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
