import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score
)
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


# ===================== 配置参数 =====================
TRAIN_ROOT = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_copfm/train"
VAL_ROOT   = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_copfm/val"
EMBED_NAME = "embedding.tif"
LABEL_NAME = "label.tif"
PATCH_H, PATCH_W = 16, 16
EMBED_DIM = 768
KNN_K = 5            # KNN近邻数
RANDOM_SEED = 42
SAVE_LOG_PATH = "/mnt/qh2-nas3/EO_test/cyz/copfm/embed_eval_cosine_result_pca.csv"
SIL_SAMPLE_SIZE = 20000  # 轮廓系数采样防OOM

# ---------------- PCA 可配置接口 ----------------
USE_PCA = True        # True开启PCA降维，False使用原始768维
PCA_DIM = 64          # 整数=固定目标维度；0~1浮点数=保留对应方差比例(如0.9)
# ====================================================

def read_tif(file_path):
    with rasterio.open(file_path) as src:
        arr = src.read()
    return arr

def load_dataset_image_level(root_dir):
    """
    图像级加载：每张图输出1个向量（所有Patch均值），1个标签
    返回：X: [图片总数,768], y: [图片总数,]
    """
    all_img_feats = []
    all_img_labels = []
    subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, d))]
    print(f"[{os.path.basename(root_dir)}] 找到 {len(subfolders)} 张图像文件夹，开始读取...")

    for folder in tqdm(subfolders, desc=f"Load {os.path.basename(root_dir)} image embed"):
        embed_path = os.path.join(folder, EMBED_NAME)
        label_path = os.path.join(folder, LABEL_NAME)
        if not (os.path.exists(embed_path) and os.path.exists(label_path)):
            continue

        embed_chw = read_tif(embed_path)   # (768,16,16)
        label_chw = read_tif(label_path)   # (1,16,16)
        label_hw = label_chw[0]

        # 校验整张图只有单一类别
        unique_cls = np.unique(label_hw)
        if len(unique_cls) != 1:
            print(f"【警告】{folder} label.tif包含多个类别: {unique_cls}")
        img_label = int(unique_cls[0])

        # 转换 HWC (16,16,768)，展平 256×768
        embed_hwc = np.transpose(embed_chw, (1, 2, 0))
        patch_num = PATCH_H * PATCH_W
        feats_patch = embed_hwc.reshape(patch_num, EMBED_DIM)

        # 关键：整张图所有Patch取均值，得到单张图像表征 [768,]
        img_feat = np.mean(feats_patch, axis=0)

        all_img_feats.append(img_feat)
        all_img_labels.append(img_label)

    # 拼接全部图像
    X = np.stack(all_img_feats, axis=0)
    y = np.array(all_img_labels, dtype=np.int64)
    print(f"[{os.path.basename(root_dir)}] 图像总数: {X.shape[0]}, 图像表征维度: {X.shape[1]}")
    print(f"[{os.path.basename(root_dir)}] 全局类别数: {len(np.unique(y))}, 类别列表: {sorted(np.unique(y))}")
    return X, y

def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    y_pred = np.array(y_pred).astype(np.int64)
    true_unique = np.unique(y_true)
    pred_unique = np.unique(y_pred)
    cost = np.zeros((len(true_unique), len(pred_unique)), dtype=np.int64)
    for i, t in enumerate(true_unique):
        for j, p in enumerate(pred_unique):
            cost[i, j] = np.sum((y_true == t) & (y_pred == p))
    row_idx, col_idx = linear_sum_assignment(-cost)
    correct = cost[row_idx, col_idx].sum()
    return correct / len(y_true)

def cross_knn_eval(X_train_norm, y_train, X_val_norm, y_val):
    # KNN余弦距离：训练集图像图库，验证集图像查询
    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine")
    knn.fit(X_train_norm, y_train)
    y_pred = knn.predict(X_val_norm)
    acc = accuracy_score(y_val, y_pred)
    return acc

def full_metric_eval(X_norm, y_true, n_cls):
    # 1. 监督轮廓系数：使用图像真实标签分组
    sil_supervised = silhouette_score(X_norm, y_true, sample_size=SIL_SAMPLE_SIZE)
    # 2. 无监督KMeans聚类
    kmeans = KMeans(n_clusters=n_cls, random_state=RANDOM_SEED, n_init=10)
    y_pred_cluster = kmeans.fit_predict(X_norm)
    # 3. 无监督轮廓系数
    sil_unsupervised = silhouette_score(X_norm, y_pred_cluster, sample_size=SIL_SAMPLE_SIZE)
    # 4. 聚类匹配指标
    acc_cluster = cluster_acc(y_true, y_pred_cluster)
    nmi = normalized_mutual_info_score(y_true, y_pred_cluster)
    ari = adjusted_rand_score(y_true, y_pred_cluster)

    return {
        "sil_supervised": sil_supervised,
        "sil_unsupervised": sil_unsupervised,
        "cluster_acc": acc_cluster,
        "nmi": nmi,
        "ari": ari
    }

if __name__ == "__main__":
    # 加载图像级向量
    X_train_raw, y_train = load_dataset_image_level(TRAIN_ROOT)
    X_val_raw, y_val     = load_dataset_image_level(VAL_ROOT)

    # 标准化（仅训练集拟合，防止信息泄露）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)

    # ===================== PCA降维分支 =====================
    if USE_PCA:
        print(f"\n>>> 开启PCA降维，配置参数 PCA_DIM = {PCA_DIM}")
        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca   = pca.transform(X_val_scaled)
        print(f"PCA完成 | 原始维度 {EMBED_DIM} → 降维后 {X_train_pca.shape[-1]}")
        # 替换为降维后特征进入后续归一化
        X_train_scaled = X_train_pca
        X_val_scaled   = X_val_pca
    # ======================================================

    # L2归一化，欧式等价余弦距离
    l2_normalizer = Normalizer(norm="l2")
    X_train_norm = l2_normalizer.transform(X_train_scaled)
    X_val_norm   = l2_normalizer.transform(X_val_scaled)
    final_dim = X_train_norm.shape[-1]

    # 1. KNN图像检索精度
    print("\n===== 图像级KNN余弦评测 (Train图像图库 → Val图像查询) =====")
    knn_val_acc = cross_knn_eval(X_train_norm, y_train, X_val_norm, y_val)
    print(f"KNN K={KNN_K} Cosine Distance Val Image Accuracy: {knn_val_acc:.4f}")

    # 2. 全套指标（监督轮廓+无监督KMeans）
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))
    print(f"\n===== 图像级全套指标 | 总类别数={n_classes} =====")
    train_metrics = full_metric_eval(X_train_norm, y_train, n_classes)
    val_metrics   = full_metric_eval(X_val_norm, y_val, n_classes)

    print("【Train图像集指标】")
    print(f"监督轮廓系数(真实图像标签): {train_metrics['sil_supervised']:.4f} | 无监督轮廓系数(KMeans): {train_metrics['sil_unsupervised']:.4f}")
    print(f"ClusterACC: {train_metrics['cluster_acc']:.4f} | NMI: {train_metrics['nmi']:.4f} | ARI: {train_metrics['ari']:.4f}")
    print("【Val图像集指标】")
    print(f"监督轮廓系数(真实图像标签): {val_metrics['sil_supervised']:.4f} | 无监督轮廓系数(KMeans): {val_metrics['sil_unsupervised']:.4f}")
    print(f"ClusterACC: {val_metrics['cluster_acc']:.4f} | NMI: {val_metrics['nmi']:.4f} | ARI: {val_metrics['ari']:.4f}")

    # 保存结果CSV（新增PCA相关字段）
    result_data = [
        {
            "subset": "train",
            "knn_k": KNN_K,
            "distance_metric": "cosine",
            "use_pca": USE_PCA,
            "pca_param": PCA_DIM,
            "knn_acc": None,
            "sil_supervised": train_metrics["sil_supervised"],
            "sil_unsupervised": train_metrics["sil_unsupervised"],
            "cluster_acc": train_metrics["cluster_acc"],
            "nmi": train_metrics["nmi"],
            "ari": train_metrics["ari"],
            "image_num": X_train_raw.shape[0],
            "ori_embed_dim": EMBED_DIM,
            "final_dim": final_dim
        },
        {
            "subset": "val",
            "knn_k": KNN_K,
            "distance_metric": "cosine",
            "use_pca": USE_PCA,
            "pca_param": PCA_DIM,
            "knn_acc": knn_val_acc,
            "sil_supervised": val_metrics["sil_supervised"],
            "sil_unsupervised": val_metrics["sil_unsupervised"],
            "cluster_acc": val_metrics["cluster_acc"],
            "nmi": val_metrics["nmi"],
            "ari": val_metrics["ari"],
            "image_num": X_val_raw.shape[0],
            "ori_embed_dim": EMBED_DIM,
            "final_dim": final_dim
        }
    ]
    df = pd.DataFrame(result_data)
    df.to_csv(SAVE_LOG_PATH, index=False)
    print(f"\n图像级全部指标已保存至: {SAVE_LOG_PATH}")
    print("评测执行完毕")