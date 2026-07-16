import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score
)
from scipy.optimize import linear_sum_assignment

# ===================== 配置参数 =====================
TRAIN_ROOT = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_olmo10m/train"
VAL_ROOT   = "/mnt/qh2-nas3/EO_test/cyz/EuroSAT100_embed_olmo10m/val"
EMBED_NAME = "embedding.tif"
LABEL_NAME = "label.tif"
PATCH_H, PATCH_W = 16, 16
EMBED_DIM = 768
KNN_K = 5
RANDOM_SEED = 42
SAVE_LOG_PATH = "/mnt/qh2-nas3/EO_test/cyz/olmo10m/embed_eval_kmeans_dbscan_image.csv"
SIL_SAMPLE_SIZE = 20000
# DBSCAN超参，根据你的数据可调整
DBSCAN_EPS = 0.12
DBSCAN_MIN_SAMPLES = 5
# ====================================================

def read_tif(file_path):
    with rasterio.open(file_path) as src:
        arr = src.read()
    return arr

def load_dataset_image_level(root_dir):
    """图像级加载：每张图所有Patch取均值，一张图一个768维向量"""
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

        unique_cls = np.unique(label_hw)
        if len(unique_cls) != 1:
            print(f"【警告】{folder} label.tif包含多个类别: {unique_cls}")
        img_label = int(unique_cls[0])

        embed_hwc = np.transpose(embed_chw, (1, 2, 0))
        patch_num = PATCH_H * PATCH_W
        feats_patch = embed_hwc.reshape(patch_num, EMBED_DIM)
        img_feat = np.mean(feats_patch, axis=0)

        all_img_feats.append(img_feat)
        all_img_labels.append(img_label)

    X = np.stack(all_img_feats, axis=0)
    y = np.array(all_img_labels, dtype=np.int64)
    print(f"[{os.path.basename(root_dir)}] 图像总数: {X.shape[0]}, 表征维度: {X.shape[1]}")
    print(f"[{os.path.basename(root_dir)}] 全局类别数: {len(np.unique(y))}")
    return X, y

def cluster_acc(y_true, y_pred):
    """匈牙利算法计算最优匹配聚类精度"""
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
    """图像级KNN余弦检索，train图库，val查询"""
    knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine")
    knn.fit(X_train_norm, y_train)
    y_pred = knn.predict(X_val_norm)
    acc = accuracy_score(y_val, y_pred)
    return acc

def filter_noise_for_metrics(X, y_true, y_pred_cluster):
    """DBSCAN会输出-1噪声点，计算指标时剔除噪声"""
    mask = y_pred_cluster != -1
    X_clean = X[mask]
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred_cluster[mask]
    return X_clean, y_true_clean, y_pred_clean

def kmeans_metrics(X_norm, y_true, n_cls):
    """KMeans无监督聚类全套指标"""
    kmeans = KMeans(n_clusters=n_cls, random_state=RANDOM_SEED, n_init=10)
    y_pred = kmeans.fit_predict(X_norm)
    sil = silhouette_score(X_norm, y_pred, sample_size=SIL_SAMPLE_SIZE)
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return {
        "sil_unsupervised": sil,
        "cluster_acc": acc,
        "nmi": nmi,
        "ari": ari,
        "cluster_num": len(np.unique(y_pred))
    }

def dbscan_metrics(X_norm, y_true):
    """DBSCAN密度聚类全套指标，余弦距离，自动过滤噪声"""
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine")
    y_pred_all = db.fit_predict(X_norm)
    # 剔除噪声样本
    X_clean, y_clean, y_pred_clean = filter_noise_for_metrics(X_norm, y_true, y_pred_all)
    noise_ratio = 1 - (len(X_clean) / len(X_norm))

    if len(np.unique(y_pred_clean)) <= 1:
        # 全部归为一类/全噪声，无法计算轮廓系数
        sil = np.nan
        acc = np.nan
        nmi = np.nan
        ari = np.nan
    else:
        sil = silhouette_score(X_clean, y_pred_clean, sample_size=min(SIL_SAMPLE_SIZE, len(X_clean)))
        acc = cluster_acc(y_clean, y_pred_clean)
        nmi = normalized_mutual_info_score(y_clean, y_pred_clean)
        ari = adjusted_rand_score(y_clean, y_pred_clean)

    return {
        "sil_unsupervised": sil,
        "cluster_acc": acc,
        "nmi": nmi,
        "ari": ari,
        "cluster_num": len(np.unique(y_pred_clean)),
        "noise_ratio": round(noise_ratio, 4)
    }

def eval_all_cluster(X_norm, y_true, n_real_cls):
    """统一入口：监督轮廓系数 + KMeans指标 + DBSCAN指标"""
    # 监督轮廓系数（仅用真实标签，不属于聚类算法）
    sil_supervised = silhouette_score(X_norm, y_true, sample_size=SIL_SAMPLE_SIZE)
    km_res = kmeans_metrics(X_norm, y_true, n_real_cls)
    db_res = dbscan_metrics(X_norm, y_true)
    return sil_supervised, km_res, db_res

if __name__ == "__main__":
    # 1. 加载图像级均值表征
    X_train_raw, y_train = load_dataset_image_level(TRAIN_ROOT)
    X_val_raw, y_val     = load_dataset_image_level(VAL_ROOT)

    # 2. 标准化（仅训练集拟合，无信息泄露）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)

    # 3. L2归一化，欧式等价余弦距离
    l2_normalizer = Normalizer(norm="l2")
    X_train_norm = l2_normalizer.transform(X_train_scaled)
    X_val_norm   = l2_normalizer.transform(X_val_scaled)

    # ========== 图像级KNN监督检索精度 ==========
    print("\n===== 图像级KNN余弦评测 (Train图库 → Val图像查询) =====")
    knn_val_acc = cross_knn_eval(X_train_norm, y_train, X_val_norm, y_val)
    print(f"KNN K={KNN_K} Cosine Distance Val Image Accuracy: {knn_val_acc:.4f}")

    # ========== 无监督聚类：KMeans + DBSCAN 双算法评估 ==========
    n_real_classes = len(np.unique(np.concatenate([y_train, y_val])))
    print(f"\n===== 无监督聚类评测，真实类别总数={n_real_classes} =====")

    # Train集聚类评估
    train_sil_sup, train_km, train_db = eval_all_cluster(X_train_norm, y_train, n_real_classes)
    # Val集聚类评估
    val_sil_sup, val_km, val_db = eval_all_cluster(X_val_norm, y_val, n_real_classes)

    # 打印Train指标
    print("\n【Train集】")
    print(f"监督轮廓系数(真实标签): {train_sil_sup:.4f}")
    print("--- KMeans无监督聚类 ---")
    print(f"簇数:{train_km['cluster_num']} | Sil:{train_km['sil_unsupervised']:.4f} | ACC:{train_km['cluster_acc']:.4f} | NMI:{train_km['nmi']:.4f} | ARI:{train_km['ari']:.4f}")
    print("--- DBSCAN无监督聚类 ---")
    print(f"簇数:{train_db['cluster_num']} | 噪声占比:{train_db['noise_ratio']:.4f} | Sil:{train_db['sil_unsupervised']:.4f} | ACC:{train_db['cluster_acc']:.4f} | NMI:{train_db['nmi']:.4f} | ARI:{train_db['ari']:.4f}")

    # 打印Val指标
    print("\n【Val集】")
    print(f"监督轮廓系数(真实标签): {val_sil_sup:.4f}")
    print("--- KMeans无监督聚类 ---")
    print(f"簇数:{val_km['cluster_num']} | Sil:{val_km['sil_unsupervised']:.4f} | ACC:{val_km['cluster_acc']:.4f} | NMI:{val_km['nmi']:.4f} | ARI:{val_km['ari']:.4f}")
    print("--- DBSCAN无监督聚类 ---")
    print(f"簇数:{val_db['cluster_num']} | 噪声占比:{val_db['noise_ratio']:.4f} | Sil:{val_db['sil_unsupervised']:.4f} | ACC:{val_db['cluster_acc']:.4f} | NMI:{val_db['nmi']:.4f} | ARI:{val_db['ari']:.4f}")

    # ========== 保存全部指标到CSV ==========
    result_rows = [
        {
            "subset": "train",
            "knn_k": KNN_K,
            "distance_metric": "cosine",
            "knn_acc": None,
            "sil_supervised": train_sil_sup,
            # KMeans
            "km_cluster_num": train_km["cluster_num"],
            "km_sil": train_km["sil_unsupervised"],
            "km_acc": train_km["cluster_acc"],
            "km_nmi": train_km["nmi"],
            "km_ari": train_km["ari"],
            # DBSCAN
            "db_cluster_num": train_db["cluster_num"],
            "db_noise_ratio": train_db["noise_ratio"],
            "db_sil": train_db["sil_unsupervised"],
            "db_acc": train_db["cluster_acc"],
            "db_nmi": train_db["nmi"],
            "db_ari": train_db["ari"],
            "image_num": X_train_raw.shape[0],
            "embed_dim": EMBED_DIM
        },
        {
            "subset": "val",
            "knn_k": KNN_K,
            "distance_metric": "cosine",
            "knn_acc": knn_val_acc,
            "sil_supervised": val_sil_sup,
            # KMeans
            "km_cluster_num": val_km["cluster_num"],
            "km_sil": val_km["sil_unsupervised"],
            "km_acc": val_km["cluster_acc"],
            "km_nmi": val_km["nmi"],
            "km_ari": val_km["ari"],
            # DBSCAN
            "db_cluster_num": val_db["cluster_num"],
            "db_noise_ratio": val_db["noise_ratio"],
            "db_sil": val_db["sil_unsupervised"],
            "db_acc": val_db["cluster_acc"],
            "db_nmi": val_db["nmi"],
            "db_ari": val_db["ari"],
            "image_num": X_val_raw.shape[0],
            "embed_dim": EMBED_DIM
        }
    ]
    df = pd.DataFrame(result_rows)
    df.to_csv(SAVE_LOG_PATH, index=False)
    print(f"\n完整指标已保存至: {SAVE_LOG_PATH}")
    print("评测全部完成")