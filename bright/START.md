# 训练启动命令

## 单卡
python train.py

## 多卡
# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py

# 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py

# 只用 4-7 号卡
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train.py

## 断点续跑
python train.py --resume ./outputs/2026-06-18_10-00-00

## 临时覆盖参数
python train.py --batch-size 8 --epochs 50 --data-root /another/path

## 有效 batch size
# 4 卡 × batch_size=4 → 有效 batch=16
# 8 卡 × batch_size=4 → 有效 batch=32
