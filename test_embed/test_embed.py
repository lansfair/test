import os
import numpy as np
import rasterio
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# ===================== 配置区域 =====================
# ROOT = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_olmo10m"
ROOT = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_copfm"
TRAIN_ROOT = os.path.join(ROOT, "train")
VAL_ROOT = os.path.join(ROOT, "val")
# DETAIL_CSV = "/mnt/qh2-nas3/EO_test/cyz/Euro100_val_retrieval_detail.csv"
# SUMMARY_CSV = "/mnt/qh2-nas3/EO_test/cyz/Euro100_accuracy_summary.csv"
DETAIL_CSV = "/mnt/qh2-nas3/EO_test/cyz/copfm/Euro100_val_retrieval_detail.csv"
SUMMARY_CSV = "/mnt/qh2-nas3/EO_test/cyz/copfm/Euro100_accuracy_summary.csv"

# ===================== 工具函数 =====================
def read_tif(tif_path: str) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        arr = src.read()
    return arr

def get_global_avg_vector(embed: np.ndarray) -> np.ndarray:
    """
    输入 embed shape (768, 16, 16)
    全局平均池化，输出 (768,) 一维场景向量
    """
    return embed.mean(axis=(1, 2))

def get_single_scene_label(lab_arr: np.ndarray) -> int:
    """整张label.tif数值全部相同，取左上角像素"""
    lab_arr = lab_arr.squeeze()
    return int(lab_arr[0, 0])

def load_scene_dataset(folder: str):
    """
    遍历train/val文件夹，批量生成：
        vec_mat: [N, 768] 所有场景向量
        cls_arr: [N,] 对应场景类别
        id_arr: [N,] 样本文件夹名
    """
    vec_list = []
    cls_list = []
    id_list = []
    sub_folders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    for sub in tqdm(sub_folders, desc=f"Loading {os.path.basename(folder)}"):
        emb_path = os.path.join(folder, sub, "embedding.tif")
        lab_path = os.path.join(folder, sub, "label.tif")
        if not (os.path.exists(emb_path) and os.path.exists(lab_path)):
            continue
        # 读取嵌入 + 全局平均池化
        emb = read_tif(emb_path)
        vec = get_global_avg_vector(emb)
        # 读取单图标签
        lab = read_tif(lab_path)
        label = get_single_scene_label(lab)
        vec_list.append(vec)
        cls_list.append(label)
        id_list.append(sub)
    if len(vec_list) == 0:
        raise RuntimeError(f"No valid samples found in {folder}")
    vec_mat = np.stack(vec_list, axis=0)
    cls_arr = np.array(cls_list)
    id_arr = np.array(id_list)
    return vec_mat, cls_arr, id_arr

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # 1. 加载train图库
    train_vecs, train_cls, train_ids = load_scene_dataset(TRAIN_ROOT)
    total_train = len(train_vecs)
    print(f"Train gallery total samples: {total_train}")

    # 2. 加载val查询集
    val_vecs, val_cls, val_ids = load_scene_dataset(VAL_ROOT)
    total_val = len(val_vecs)
    print(f"Val query total samples: {total_val}")

    detail_records = []
    top1_correct = 0
    top5_correct = 0

    # 遍历每一张val图片检索
    for idx in tqdm(range(total_val), desc="Calculating cosine similarity & retrieval"):
        q_vec = val_vecs[idx:idx+1]
        q_label = int(val_cls[idx])
        q_sample_id = val_ids[idx]

        # ========== 余弦相似度计算核心代码 ==========
        # q_vec: (1,768) 当前val全局向量
        # train_vecs: (N_train,768) 全部train向量库
        sim_scores = cosine_similarity(q_vec, train_vecs)[0]
        # ==========================================

        # 相似度从高到低排序
        sort_indices = np.argsort(sim_scores)[::-1]
        sorted_sim = sim_scores[sort_indices]
        sorted_cls = train_cls[sort_indices]
        sorted_sample_ids = train_ids[sort_indices]

        # 提取Top1信息
        t1_cls = sorted_cls[0]
        t1_id = sorted_sample_ids[0]
        t1_sim = round(float(sorted_sim[0]), 4)
        hit1 = 1 if t1_cls == q_label else 0

        # 提取Top5信息
        top5_cls_list = sorted_cls[:5]
        top5_id_list = sorted_sample_ids[:5]
        top5_sim_list = [round(float(s), 4) for s in sorted_sim[:5]]
        hit5 = 1 if (q_label in top5_cls_list) else 0

        # 统计正确数量
        top1_correct += hit1
        top5_correct += hit5

        # 组装单条明细
        record = {
            "val_sample_id": q_sample_id,
            "val_scene_label": q_label,
            "top1_train_id": t1_id,
            "top1_train_label": t1_cls,
            "top1_cosine_sim": t1_sim,
            "hit_top1": hit1,
            "top5_train_ids": ",".join(top5_id_list),
            "top5_train_labels": ",".join([str(c) for c in top5_cls_list]),
            "top5_cosine_sims": ",".join([str(s) for s in top5_sim_list]),
            "hit_top5": hit5
        }
        detail_records.append(record)

    # 输出每张val详细匹配结果
    df_detail = pd.DataFrame(detail_records)
    df_detail.to_csv(DETAIL_CSV, index=False, encoding="utf-8-sig")

    # 计算全局平均准确率
    top1_acc = top1_correct / total_val
    top5_acc = top5_correct / total_val

    # 汇总指标表格
    summary_data = [{
        "total_val_samples": total_val,
        "top1_correct_count": top1_correct,
        "top1_avg_accuracy": round(top1_acc, 4),
        "top5_correct_count": top5_correct,
        "top5_avg_accuracy": round(top5_acc, 4)
    }]
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # 控制台打印最终平均准确率
    print("=" * 75)
    print(f"Total val samples: {total_val}")
    print(f"Top1 Average Retrieval Accuracy: {top1_acc:.4f}")
    print(f"Top5 Average Retrieval Accuracy: {top5_acc:.4f}")
    print("=" * 75)
    print(f"Detail per val sample: {DETAIL_CSV}")
    print(f"Global average accuracy summary: {SUMMARY_CSV}")