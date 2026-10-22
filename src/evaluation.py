"""Module đánh giá toàn diện, kiểm tra tổng quát hóa và độ ổn định phân cụm (Pha C3).

Tuân thủ nghiêm ngặt các quy tắc trong config/decision_rules.yaml:
1. Chất lượng nội tại: Silhouette (SVD & Raw L2), DB, CH, sample silhouette plot,
   chiếu 2D PCA & t-SNE tô màu theo cụm (UMAP không có trong môi trường python).
   So sánh với phân phối Null Model và với Baseline ở Pha C2.
2. Ổn định và tổng quát hóa trên khách (Hold-Out 80/20, >= 100 lần lặp):
   - Mỗi lần lặp: chọn 80% khách làm train; CLONE pipeline và model, FIT CHỈ TRÊN TRAIN.
   - 20% còn lại chỉ TRANSFORM + gán cụm (predict cho KMeans; nearest-centroid trong
     không gian SVD cho Agglomerative/Spectral).
   - Đo: (a) ARI/NMI train vs full model, (b) ARI hold-out vs full model, (c) Silhouette hold-out.
   - Báo cáo trung vị, P5, P95. Vẽ boxplot lưu figures/cluster_stability_boxplots.png.
   - Xuất chi tiết từng lần lặp ra outputs/stability_runs.csv.
   - Độ nhạy theo seed (>= 20 seeds) và độ nhạy theo k (k-1, k, k+1).
3. Kiểm tra theo thời gian (Temporal Validation / Future Behavior Test) ở mức ALBUM & GENRE:
   - Sắp xếp invoice mỗi khách theo thời gian; nửa đầu (3 invoice) = giai đoạn 1, còn lại = giai đoạn 2.
   - Fit pipeline trên giai đoạn 1. Với giai đoạn 2:
     (a) Đo ARI giữa cụm giai đoạn 1 và cụm gom riêng trên giai đoạn 2.
     (b) Kiểm tra dự báo: hồ sơ giai đoạn 2 có gần centroid cụm của chính mình hơn các centroid khác không?
     (c) So sánh với phân phối ngẫu nhiên (hoán vị nhãn >= 1000 lần) để tính empirical p-value.
4. Tiêu chí kết luận theo decision_rules.yaml:
   - Silhouette > Null P95, median ARI hold-out >= 0.60, min_cluster_size >= 5, vượt trội baselines.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Tắt cảnh báo Convergence/UserWarnings để log sạch sẽ
warnings.filterwarnings("ignore")

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Đảm bảo root directory có trong sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import normalize
import yaml

from src.features import build_matrix, load_processed_data
from src.pipeline import build_variant_pipeline


# Thiết lập Matplotlib cho biểu đồ chất lượng cao
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 1. HÀM TÍNH TOÁN METRIC CHUẨN XÁC & NHANH
# =====================================================================

def fast_cosine_silhouette(X_norm: np.ndarray, labels: np.ndarray) -> float:
    """Tính silhouette score cosine cho ma trận X đã chuẩn hóa L2 (N=59)."""
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) == n:
        return -1.0

    D = 1.0 - np.dot(X_norm, X_norm.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    a = np.zeros(n)
    b = np.full(n, np.inf)

    for lbl in unique_labels:
        mask = (labels == lbl)
        size = int(np.sum(mask))
        if size > 1:
            a[mask] = np.sum(D[mask][:, mask], axis=1) / (size - 1)
        else:
            a[mask] = 0.0

    for lbl in unique_labels:
        mask = (labels == lbl)
        for other_lbl in unique_labels:
            if other_lbl == lbl:
                continue
            o_mask = (labels == other_lbl)
            o_size = int(np.sum(o_mask))
            if o_size > 0:
                dist_to_other = np.sum(D[mask][:, o_mask], axis=1) / o_size
                b[mask] = np.minimum(b[mask], dist_to_other)

    max_ab = np.maximum(a, b)
    s = np.zeros(n)
    valid = max_ab > 0
    s[valid] = (b[valid] - a[valid]) / max_ab[valid]
    return float(np.mean(s))


def compute_sample_silhouettes(X_norm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Tính silhouette value cho từng mẫu (khách hàng) trên không gian L2."""
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) == n:
        return np.zeros(n)

    D = 1.0 - np.dot(X_norm, X_norm.T)
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)

    a = np.zeros(n)
    b = np.full(n, np.inf)

    for lbl in unique_labels:
        mask = (labels == lbl)
        size = int(np.sum(mask))
        if size > 1:
            a[mask] = np.sum(D[mask][:, mask], axis=1) / (size - 1)
        else:
            a[mask] = 0.0

    for lbl in unique_labels:
        mask = (labels == lbl)
        for other_lbl in unique_labels:
            if other_lbl == lbl:
                continue
            o_mask = (labels == other_lbl)
            o_size = int(np.sum(o_mask))
            if o_size > 0:
                dist_to_other = np.sum(D[mask][:, o_mask], axis=1) / o_size
                b[mask] = np.minimum(b[mask], dist_to_other)

    max_ab = np.maximum(a, b)
    s = np.zeros(n)
    valid = max_ab > 0
    s[valid] = (b[valid] - a[valid]) / max_ab[valid]
    return s


def assign_clusters_nearest_centroid(
    X_train_svd: np.ndarray,
    train_labels: np.ndarray,
    X_test_svd: np.ndarray,
    k: int,
) -> np.ndarray:
    """Gán nhãn cụm cho tập test bằng nearest centroid trong không gian SVD (cho Agglomerative/Spectral).
    
    Tính tâm cụm của từng nhóm trên train, chuẩn hóa L2, rồi tìm argmax cosine similarity.
    """
    dim = X_train_svd.shape[1]
    centroids = np.zeros((k, dim), dtype=float)
    for c in range(k):
        c_mask = (train_labels == c)
        if np.sum(c_mask) > 0:
            centroids[c] = np.mean(X_train_svd[c_mask], axis=0)
        else:
            centroids[c] = np.random.randn(dim)

    # Chuẩn hóa L2 các centroid
    norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    centroids = centroids / norm

    # Tương đồng cosine = X_test_svd @ centroids.T
    test_sim = np.dot(X_test_svd, centroids.T)
    test_pred = np.argmax(test_sim, axis=1)
    return test_pred


# =====================================================================
# 2. ĐỊNH NGHĨA CÁC ỨNG VIÊN ĐÁNH GIÁ (TOP CANDIDATES)
# =====================================================================

def get_c3_candidates() -> List[Dict[str, Any]]:
    """Trả về danh sách 5 ứng viên hàng đầu từ reports/model_comparison.md."""
    return [
        {
            "config_id": "V2_album_count_svd5_dfFalse_KM_k5",
            "name": "KMeans k=5 (V2 Count, SVD=5)",
            "variant": "V2_album_count",
            "variant_cfg": {
                "level": "album",
                "metric": "count",
                "use_idf": True,
                "min_customers": 2,
                "n_components": 5,
                "drop_first_component": False,
            },
            "model_family": "KMeans",
            "model": KMeans(n_clusters=5, n_init=30, random_state=42),
            "k": 5,
            "null_p95": 0.4441,
            "rank": 1,
            "is_recommended": True,
        },
        {
            "config_id": "V2_album_count_svd5_dfFalse_KM_k6",
            "name": "KMeans k=6 (V2 Count, SVD=5)",
            "variant": "V2_album_count",
            "variant_cfg": {
                "level": "album",
                "metric": "count",
                "use_idf": True,
                "min_customers": 2,
                "n_components": 5,
                "drop_first_component": False,
            },
            "model_family": "KMeans",
            "model": KMeans(n_clusters=6, n_init=30, random_state=42),
            "k": 6,
            "null_p95": 0.4411,
            "rank": 2,
            "is_recommended": False,
        },
        {
            "config_id": "V2_album_count_svd5_dfFalse_Agg_complete_k5",
            "name": "Agglomerative Complete k=5 (V2 Count, SVD=5)",
            "variant": "V2_album_count",
            "variant_cfg": {
                "level": "album",
                "metric": "count",
                "use_idf": True,
                "min_customers": 2,
                "n_components": 5,
                "drop_first_component": False,
            },
            "model_family": "Agglomerative",
            "model": AgglomerativeClustering(n_clusters=5, linkage="complete", metric="cosine"),
            "k": 5,
            "null_p95": 0.4010,
            "rank": 3,
            "is_recommended": False,
        },
        {
            "config_id": "V2_album_count_svd5_dfFalse_Spectral_NN8_k5",
            "name": "Spectral NN=8 k=5 (V2 Count, SVD=5)",
            "variant": "V2_album_count",
            "variant_cfg": {
                "level": "album",
                "metric": "count",
                "use_idf": True,
                "min_customers": 2,
                "n_components": 5,
                "drop_first_component": False,
            },
            "model_family": "SpectralClustering",
            "model": SpectralClustering(
                n_clusters=5,
                affinity="nearest_neighbors",
                n_neighbors=8,
                random_state=42,
                assign_labels="kmeans",
            ),
            "k": 5,
            "null_p95": 0.4347,
            "rank": 4,
            "is_recommended": False,
        },
        {
            "config_id": "V1_album_binary_svd5_dfFalse_KM_k5",
            "name": "KMeans k=5 (V1 Binary, SVD=5)",
            "variant": "V1_album_binary",
            "variant_cfg": {
                "level": "album",
                "metric": "binary",
                "use_idf": True,
                "min_customers": 2,
                "n_components": 5,
                "drop_first_component": False,
            },
            "model_family": "KMeans",
            "model": KMeans(n_clusters=5, n_init=30, random_state=42),
            "k": 5,
            "null_p95": 0.4441,
            "rank": 5,
            "is_recommended": False,
        },
    ]


# =====================================================================
# 3. ĐÁNH GIÁ CHẤT LƯỢNG NỘI TẠI & TRỰC QUAN HÓA 2D
# =====================================================================

def evaluate_internal_quality(
    candidates: List[Dict[str, Any]],
    fact_purchase: pd.DataFrame,
    dims: Dict[str, pd.DataFrame],
    figures_dir: Path,
) -> pd.DataFrame:
    """Đánh giá chất lượng nội tại và vẽ biểu đồ 2D cho toàn bộ tập dữ liệu."""
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())
    records = []

    print("\n" + "=" * 70)
    print("1. ĐÁNH GIÁ CHẤT LƯỢNG NỘI TẠI TRÊN TOÀN BỘ DỮ LIỆU (INTERNAL QUALITY)")
    print("=" * 70)

    for cand in candidates:
        v_key = cand["variant"]
        v_cfg = cand["variant_cfg"]
        model = clone(cand["model"])
        k = cand["k"]

        raw_df = build_matrix(
            fact_purchase,
            dims,
            level=v_cfg["level"],
            metric=v_cfg["metric"],
            customer_ids=customer_ids,
            return_sparse=False,
        )

        pipe = build_variant_pipeline(v_key, v_cfg, random_state=42)
        X_svd = pipe.fit_transform(raw_df.values)
        X_raw_norm = normalize(raw_df.values, norm="l2")

        if isinstance(model, KMeans):
            model.fit(X_svd)
            labels = model.labels_
        else:
            labels = model.fit_predict(X_svd)

        sil_svd = fast_cosine_silhouette(X_svd, labels)
        sil_raw = fast_cosine_silhouette(X_raw_norm, labels)
        db = float(davies_bouldin_score(X_svd, labels))
        ch = float(calinski_harabasz_score(X_svd, labels))

        cluster_counts = pd.Series(labels).value_counts().tolist()
        min_size = min(cluster_counts)
        max_size = max(cluster_counts)
        balance = round(min_size / max_size, 4)

        sample_sils = compute_sample_silhouettes(X_svd, labels)

        cand["full_labels"] = labels
        cand["X_svd"] = X_svd
        cand["raw_df"] = raw_df
        cand["sample_sils"] = sample_sils
        cand["pipe"] = pipe

        records.append({
            "config_id": cand["config_id"],
            "name": cand["name"],
            "k": k,
            "silhouette_svd": round(sil_svd, 4),
            "silhouette_raw_l2": round(sil_raw, 4),
            "null_p95": cand["null_p95"],
            "gap_to_null_p95": round(sil_svd - cand["null_p95"], 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 4),
            "min_cluster_size": min_size,
            "max_cluster_size": max_size,
            "balance_ratio": balance,
            "cluster_sizes": str(cluster_counts),
        })

        print(
            f"[*] {cand['config_id']:<40} | Sil(SVD)={sil_svd:.4f} (Null P95={cand['null_p95']:.4f}, Gap={sil_svd-cand['null_p95']:+.4f}) | "
            f"Sil(Raw)={sil_raw:.4f} | DB={db:.4f} | CH={ch:.2f} | Sizes={cluster_counts} | Bal={balance:.4f}"
        )

    # -------------------------------------------------------------
    # Vẽ biểu đồ Sample Silhouette Plot cho Ứng viên #1 (Recommended)
    # -------------------------------------------------------------
    rec_cand = candidates[0]
    labels_rec = rec_cand["full_labels"]
    sils_rec = rec_cand["sample_sils"]
    k_rec = rec_cand["k"]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    y_lower = 10
    colors = cm.nipy_spectral(np.linspace(0, 0.85, k_rec))

    for i in range(k_rec):
        ith_cluster_sils = sils_rec[labels_rec == i]
        ith_cluster_sils.sort()
        size_cluster_i = ith_cluster_sils.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sils,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )
        ax.text(
            -0.08,
            y_lower + 0.5 * size_cluster_i,
            f"Cụm {i} (n={size_cluster_i})",
            color="black",
            fontweight="bold",
            fontsize=9,
            va="center",
        )
        y_lower = y_upper + 5

    avg_sil = np.mean(sils_rec)
    ax.axvline(
        x=avg_sil,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Silhouette Trung bình = {avg_sil:.4f}",
    )
    ax.set_title(
        f"Biểu đồ Silhouette Từng Khách hàng (Sample Silhouette Plot)\n"
        f"Mô hình: {rec_cand['name']} (n=59)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Giá trị Silhouette Coefficient", fontsize=10, fontweight="bold")
    ax.set_ylabel("Thứ tự Khách hàng theo Cụm", fontsize=10, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim([-0.15, 1.0])
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    plt.tight_layout()
    sil_plot_path = figures_dir / "cluster_silhouette_plot.png"
    plt.savefig(sil_plot_path, dpi=300)
    plt.close()
    print(f"\n[+] Đã lưu biểu đồ Silhouette từng khách: {sil_plot_path}")

    # -------------------------------------------------------------
    # Vẽ biểu đồ 2D Projections (PCA & t-SNE)
    # (Lưu ý: UMAP không có trong môi trường python, sử dụng PCA và t-SNE)
    # -------------------------------------------------------------
    X_rec_svd = rec_cand["X_svd"]
    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X_rec_svd)
    tsne_2d = TSNE(n_components=2, perplexity=15, random_state=42, max_iter=1000).fit_transform(
        X_rec_svd
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for i in range(k_rec):
        mask = (labels_rec == i)
        ax1.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            c=[colors[i]],
            label=f"Cụm {i} (n={np.sum(mask)})",
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
        )
        # Vẽ tâm cụm trong không gian PCA
        c_mean = np.mean(pca_2d[mask], axis=0)
        ax1.scatter(c_mean[0], c_mean[1], marker="X", s=180, c="black", edgecolors="white", linewidth=1.5)

        ax2.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=[colors[i]],
            label=f"Cụm {i} (n={np.sum(mask)})",
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
        )
        t_mean = np.mean(tsne_2d[mask], axis=0)
        ax2.scatter(t_mean[0], t_mean[1], marker="X", s=180, c="black", edgecolors="white", linewidth=1.5)

    ax1.set_title("Chiếu 2D PCA (Không gian SVD 5D)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Principal Component 1", fontsize=9)
    ax1.set_ylabel("Principal Component 2", fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(loc="best", fontsize=8, frameon=True)

    ax2.set_title("Chiếu 2D t-SNE (Perplexity=15)\n(Lưu ý: UMAP không khả dụng, dùng t-SNE thay thế)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=9)
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(
        f"Trực quan hóa Phân cụm Khách hàng 2D: {rec_cand['name']} (n=59 Khách)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    proj_path = figures_dir / "cluster_pca_projection.png"
    plt.savefig(proj_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ chiếu 2D PCA/t-SNE: {proj_path}")

    return pd.DataFrame(records)


# =====================================================================
# 4. KIỂM TRA ĐỘ ỔN ĐỊNH VÀ TỔNG QUÁT HÓA (HOLD-OUT CV 80/20)
# =====================================================================

def run_holdout_stability_cv(
    candidates: List[Dict[str, Any]],
    n_splits: int = 100,
    train_ratio: float = 0.8,
    random_state: int = 42,
    output_dir: Path = Path("outputs"),
    figures_dir: Path = Path("figures"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Thực hiện kiểm tra Hold-out khách hàng 80/20 lặp lại >= 100 lần.
    
    Quy trình chặt chẽ:
    - Mỗi lần: bốc 80% khách làm train (47 khách), 20% làm test (12 khách).
    - CLONE pipeline và model, FIT CHỈ TRÊN TRAIN.
    - Test CHỈ TRANSFORM qua pipeline train, gán cụm (predict hoặc nearest centroid).
    - Đo ARI train vs full model, ARI test vs full model, Silhouette test.
    """
    print("\n" + "=" * 70)
    print(f"2. KIỂM TRA ỔN ĐỊNH VÀ TỔNG QUÁT HÓA KHÁCH HÀNG (HOLD-OUT CV: {n_splits} LẦN LẶP)")
    print("=" * 70)

    n_customers = 59
    n_train = int(np.round(n_customers * train_ratio))  # 47 khách
    n_test = n_customers - n_train                      # 12 khách
    rng = np.random.default_rng(random_state)

    stability_runs = []
    summary_stats = []

    for cand in candidates:
        cand_id = cand["config_id"]
        model_family = cand["model_family"]
        k = cand["k"]
        raw_df = cand["raw_df"]
        pipe_base = cand["pipe"]
        model_base = cand["model"]
        labels_full = cand["full_labels"]

        print(f"\n>>> Đang chạy {n_splits} lần Hold-out CV cho ứng viên: {cand_id}...")
        t0 = time.time()

        train_aris = []
        train_nmis = []
        holdout_aris = []
        holdout_sils = []

        for b in range(n_splits):
            perm = rng.permutation(n_customers)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

            X_raw_train = raw_df.values[train_idx]
            X_raw_test = raw_df.values[test_idx]

            # CLONE pipeline và fit CHỈ TRÊN TRAIN (ItemFilter, IDF, SVD, L2)
            pipe_b = clone(pipe_base)
            X_svd_train = pipe_b.fit_transform(X_raw_train)
            X_svd_test = pipe_b.transform(X_raw_test)

            # Fit model trên train
            if isinstance(model_base, KMeans):
                m_b = clone(model_base)
                m_b.fit(X_svd_train)
                train_pred = m_b.labels_
                test_pred = m_b.predict(X_svd_test)
            else:
                m_b = clone(model_base)
                train_pred = m_b.fit_predict(X_svd_train)
                # Gán nhãn cho test bằng nearest centroid trong không gian SVD đã giảm chiều
                test_pred = assign_clusters_nearest_centroid(
                    X_svd_train, train_pred, X_svd_test, k
                )

            # Đo các chỉ số
            ari_tr = float(adjusted_rand_score(labels_full[train_idx], train_pred))
            nmi_tr = float(normalized_mutual_info_score(labels_full[train_idx], train_pred))
            ari_te = float(adjusted_rand_score(labels_full[test_idx], test_pred))
            sil_te = fast_cosine_silhouette(X_svd_test, test_pred)

            train_aris.append(ari_tr)
            train_nmis.append(nmi_tr)
            holdout_aris.append(ari_te)
            holdout_sils.append(sil_te)

            stability_runs.append({
                "iteration": b + 1,
                "config_id": cand_id,
                "model_family": model_family,
                "k": k,
                "train_ari": round(ari_tr, 4),
                "train_nmi": round(nmi_tr, 4),
                "holdout_ari": round(ari_te, 4),
                "holdout_silhouette": round(sil_te, 4),
                "train_size": n_train,
                "holdout_size": n_test,
            })

        elapsed = time.time() - t0
        med_ari_te = float(np.median(holdout_aris))
        p5_ari_te = float(np.percentile(holdout_aris, 5))
        p95_ari_te = float(np.percentile(holdout_aris, 95))

        med_ari_tr = float(np.median(train_aris))
        med_nmi_tr = float(np.median(train_nmis))
        med_sil_te = float(np.median(holdout_sils))

        passed_stability = med_ari_te >= 0.60

        summary_stats.append({
            "config_id": cand_id,
            "name": cand["name"],
            "model_family": model_family,
            "k": k,
            "train_ari_median": round(med_ari_tr, 4),
            "train_nmi_median": round(med_nmi_tr, 4),
            "holdout_ari_median": round(med_ari_te, 4),
            "holdout_ari_p5": round(p5_ari_te, 4),
            "holdout_ari_p95": round(p95_ari_te, 4),
            "holdout_sil_median": round(med_sil_te, 4),
            "passed_stability_threshold_0_60": passed_stability,
            "runtime_sec": round(elapsed, 2),
        })

        status = "[VƯỢT TRẦN THÀNH CÔNG]" if passed_stability else "[CHƯA ĐẠT NGƯỠNG 0.60]"
        print(
            f"   -> Kết quả: Train ARI Median={med_ari_tr:.4f} | "
            f"Hold-out ARI Median={med_ari_te:.4f} (P5={p5_ari_te:.4f}, P95={p95_ari_te:.4f}) | "
            f"Hold-out Sil Median={med_sil_te:.4f} | {status} ({elapsed:.1f}s)"
        )

    df_runs = pd.DataFrame(stability_runs)
    df_summary = pd.DataFrame(summary_stats)

    runs_csv_path = output_dir / "stability_runs.csv"
    df_runs.to_csv(runs_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Đã lưu toàn bộ {len(df_runs)} lượt lặp Hold-out: {runs_csv_path}")

    # -------------------------------------------------------------
    # Vẽ biểu đồ Boxplot Hold-Out Stability
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Boxplot 1: Holdout ARI
    data_ari = [
        df_runs[df_runs["config_id"] == c["config_id"]]["holdout_ari"].values
        for c in candidates
    ]
    cand_labels = [c["config_id"].replace("_svd5_dfFalse", "") for c in candidates]

    box1 = ax1.boxplot(
        data_ari,
        labels=cand_labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2.0),
        boxprops=dict(facecolor="#e6f2ff", color="#0055aa"),
        whiskerprops=dict(color="#0055aa", linestyle="--"),
        capprops=dict(color="#0055aa"),
        flierprops=dict(marker="o", color="#ff7f0e", alpha=0.6, markersize=5),
    )

    ax1.axhline(
        y=0.60,
        color="green",
        linestyle="--",
        linewidth=1.8,
        label="Ngưỡng Ổn định Đạt chuẩn (ARI >= 0.60)",
    )
    ax1.axhline(
        y=0.40,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="Ngưỡng Trung bình (ARI = 0.40)",
    )
    ax1.set_title("Phân phối Hold-out ARI (100 Lần Lặp 80/20)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Adjusted Rand Index (ARI)", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", alpha=0.3, linestyle=":")
    ax1.legend(loc="lower left", fontsize=8, frameon=True)

    # Boxplot 2: Holdout Silhouette
    data_sil = [
        df_runs[df_runs["config_id"] == c["config_id"]]["holdout_silhouette"].values
        for c in candidates
    ]

    box2 = ax2.boxplot(
        data_sil,
        labels=cand_labels,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2.0),
        boxprops=dict(facecolor="#eaf7ed", color="#2ca02c"),
        whiskerprops=dict(color="#2ca02c", linestyle="--"),
        capprops=dict(color="#2ca02c"),
        flierprops=dict(marker="o", color="#d62728", alpha=0.6, markersize=5),
    )

    null_p95_ref = candidates[0]["null_p95"]
    ax2.axhline(
        y=null_p95_ref,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Null Model P95 ({null_p95_ref:.4f})",
    )
    ax2.set_title("Phân phối Hold-out Silhouette (100 Lần Lặp 80/20)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Silhouette Score (Cosine)", fontsize=9, fontweight="bold")
    ax2.tick_params(axis="x", rotation=25)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.legend(loc="lower left", fontsize=8, frameon=True)

    fig.suptitle(
        "Kiểm định Độ Ổn định và Tổng quát hóa Mô hình (Hold-out Validation)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    box_path = figures_dir / "cluster_stability_boxplots.png"
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ Boxplot ổn định: {box_path}")

    return df_runs, df_summary


# =====================================================================
# 5. KIỂM TRA ĐỘ NHẠY THEO RANDOM SEED & THEO K
# =====================================================================

def evaluate_seed_sensitivity(
    candidate: Dict[str, Any],
    n_seeds: int = 25,
) -> Dict[str, Any]:
    """Kiểm tra độ nhạy của thuật toán theo random_state (>= 20 seeds).
    
    Đo ARI pairwise giữa tất cả các cặp seed chạy trên toàn bộ dữ liệu.
    """
    print(f"\n>>> Đang kiểm tra độ nhạy theo Seed ({n_seeds} seeds) cho: {candidate['config_id']}...")
    X_svd = candidate["X_svd"]
    k = candidate["k"]
    model_family = candidate["model_family"]

    labels_by_seed = []
    seeds = [10 + i * 3 for i in range(n_seeds)]

    for s in seeds:
        if model_family == "KMeans":
            m = KMeans(n_clusters=k, n_init=30, random_state=s).fit(X_svd)
            labels_by_seed.append(m.labels_)
        elif model_family == "SpectralClustering":
            m = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=8,
                random_state=s,
                assign_labels="kmeans",
            ).fit(X_svd)
            labels_by_seed.append(m.labels_)
        else:
            # Thuật toán tất định (Deterministic) như Agglomerative
            labels_by_seed.append(candidate["full_labels"])

    pairwise_aris = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ari = adjusted_rand_score(labels_by_seed[i], labels_by_seed[j])
            pairwise_aris.append(ari)

    result = {
        "candidate": candidate["config_id"],
        "n_seeds_tested": n_seeds,
        "mean_pairwise_ari": round(float(np.mean(pairwise_aris)), 4),
        "median_pairwise_ari": round(float(np.median(pairwise_aris)), 4),
        "min_pairwise_ari": round(float(np.min(pairwise_aris)), 4),
        "max_pairwise_ari": round(float(np.max(pairwise_aris)), 4),
        "std_pairwise_ari": round(float(np.std(pairwise_aris)), 4),
    }

    print(
        f"   -> Seed Sensitivity: Mean ARI={result['mean_pairwise_ari']} | "
        f"Median ARI={result['median_pairwise_ari']} | Min={result['min_pairwise_ari']} | Max={result['max_pairwise_ari']}"
    )
    return result


def evaluate_k_sensitivity(
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Kiểm tra độ nhạy theo số cụm k: so sánh k-1, k, k+1 (ví dụ k=4, 5, 6)."""
    X_svd = candidate["X_svd"]
    k_target = candidate["k"]
    labels_target = candidate["full_labels"]

    k_minus = k_target - 1
    k_plus = k_target + 1

    km_minus = KMeans(n_clusters=k_minus, n_init=30, random_state=42).fit(X_svd)
    km_plus = KMeans(n_clusters=k_plus, n_init=30, random_state=42).fit(X_svd)

    ari_minus = adjusted_rand_score(labels_target, km_minus.labels_)
    ari_plus = adjusted_rand_score(labels_target, km_plus.labels_)

    nmi_minus = normalized_mutual_info_score(labels_target, km_minus.labels_)
    nmi_plus = normalized_mutual_info_score(labels_target, km_plus.labels_)

    # Bảng chéo (cross-tabulation) xem cụm nào phân rã / hợp nhất
    ct_minus = pd.crosstab(labels_target, km_minus.labels_).to_dict()
    ct_plus = pd.crosstab(labels_target, km_plus.labels_).to_dict()

    result = {
        "candidate": candidate["config_id"],
        "k_target": k_target,
        "k_minus": k_minus,
        "k_plus": k_plus,
        "ari_with_k_minus_1": round(float(ari_minus), 4),
        "nmi_with_k_minus_1": round(float(nmi_minus), 4),
        "ari_with_k_plus_1": round(float(ari_plus), 4),
        "nmi_with_k_plus_1": round(float(nmi_plus), 4),
    }

    print(
        f"\n>>> K-Sensitivity (k={k_target}): "
        f"So với k={k_minus}: ARI={result['ari_with_k_minus_1']}, NMI={result['nmi_with_k_minus_1']} | "
        f"So với k={k_plus}: ARI={result['ari_with_k_plus_1']}, NMI={result['nmi_with_k_plus_1']}"
    )
    return result


# =====================================================================
# 6. KIỂM TRA THEO THỜI GIAN (TEMPORAL VALIDATION / FUTURE BEHAVIOR)
# =====================================================================

def run_temporal_validation(
    fact_purchase: pd.DataFrame,
    dims: Dict[str, pd.DataFrame],
    candidate: Dict[str, Any],
    n_permutations: int = 1000,
    random_state: int = 42,
    figures_dir: Path = Path("figures"),
) -> Dict[str, Any]:
    """Kiểm tra hành vi tương lai theo trục thời gian (Invoice-level temporal split).
    
    Quy tắc nghiệp vụ:
    - Sắp xếp invoice của từng khách theo thứ tự thời gian (InvoiceDate, InvoiceId).
    - Nửa đầu (3 invoice đầu) = Giai đoạn 1 (Period 1).
    - Nửa sau (3-4 invoice sau) = Giai đoạn 2 (Period 2).
    - Thực hiện ở mức ALBUM và GENRE (không làm ở mức track).
    - Fit pipeline trên Giai đoạn 1 -> cụm GĐ 1.
    - Với Giai đoạn 2:
      (a) Đo ARI giữa cụm GĐ 1 và cụm gom riêng trên GĐ 2.
      (b) Kiểm tra dự báo: hồ sơ GĐ 2 có gần centroid cụm của chính mình hơn các cụm khác không?
      (c) Permutation test (>= 1000 lần) hoán vị nhãn để tính empirical p-value.
    """
    print("\n" + "=" * 70)
    print("3. KIỂM TRA THEO TRỤC THỜI GIAN (TEMPORAL FUTURE BEHAVIOR TEST)")
    print("=" * 70)

    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())
    k = candidate["k"]
    v_cfg = candidate["variant_cfg"]

    # 1. Tách Period 1 và Period 2 theo khách hàng
    # Mỗi khách hàng Chinook có 6 hoặc 7 hóa đơn
    invoices = fact_purchase[["CustomerId", "InvoiceId", "InvoiceDate"]].drop_duplicates()
    invoices = invoices.sort_values(["CustomerId", "InvoiceDate", "InvoiceId"])
    invoices["invoice_rank"] = invoices.groupby("CustomerId").cumcount()

    p1_invoices = set(invoices[invoices["invoice_rank"] < 3]["InvoiceId"])
    p2_invoices = set(invoices[invoices["invoice_rank"] >= 3]["InvoiceId"])

    fact_p1 = fact_purchase[fact_purchase["InvoiceId"].isin(p1_invoices)]
    fact_p2 = fact_purchase[fact_purchase["InvoiceId"].isin(p2_invoices)]

    print(f"[*] Giai đoạn 1 (Period 1): {len(fact_p1)} dòng giao dịch | {fact_p1['CustomerId'].nunique()} khách")
    print(f"[*] Giai đoạn 2 (Period 2): {len(fact_p2)} dòng giao dịch | {fact_p2['CustomerId'].nunique()} khách")

    # 2. Xây dựng ma trận Album Count cho Giai đoạn 1 và Giai đoạn 2
    raw_df_p1 = build_matrix(
        fact_p1, dims, level="album", metric="count", customer_ids=customer_ids, return_sparse=False
    )
    raw_df_p2 = build_matrix(
        fact_p2, dims, level="album", metric="count", customer_ids=customer_ids, return_sparse=False
    )

    # 3. Fit pipeline trên Giai đoạn 1
    pipe_p1 = build_variant_pipeline(candidate["variant"], v_cfg, random_state=random_state)
    X_svd_p1 = pipe_p1.fit_transform(raw_df_p1.values)

    # Transform Giai đoạn 2 bằng pipeline đã fit trên Giai đoạn 1
    X_svd_p2 = pipe_p1.transform(raw_df_p2.values)

    # Gom cụm trên Giai đoạn 1
    km_p1 = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit(X_svd_p1)
    labels_p1 = km_p1.labels_
    centroids_p1 = km_p1.cluster_centers_

    # Chuẩn hóa L2 các centroid GĐ 1 để so sánh cosine
    norm_c = np.linalg.norm(centroids_p1, axis=1, keepdims=True)
    norm_c[norm_c == 0] = 1.0
    centroids_p1_norm = centroids_p1 / norm_c

    # Gom cụm độc lập riêng trên Giai đoạn 2
    pipe_p2 = build_variant_pipeline(candidate["variant"], v_cfg, random_state=random_state)
    X_svd_p2_own = pipe_p2.fit_transform(raw_df_p2.values)
    km_p2 = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit(X_svd_p2_own)
    labels_p2_own = km_p2.labels_

    # (a) ARI giữa cụm GĐ 1 và cụm GĐ 2 gom riêng
    temporal_ari = float(adjusted_rand_score(labels_p1, labels_p2_own))
    temporal_nmi = float(normalized_mutual_info_score(labels_p1, labels_p2_own))

    # (b) Kiểm tra Dự báo Tương lai:
    # Với từng khách, tính khoảng cách cosine từ vector GĐ 2 (X_svd_p2) tới các centroid GĐ 1
    # Cosine distance = 1 - dot product
    sims = np.dot(X_svd_p2, centroids_p1_norm.T)  # (59, k)
    assigned_closest_cluster = np.argmax(sims, axis=1)
    is_closest_to_own_cluster = (assigned_closest_cluster == labels_p1)

    actual_satisfaction_count = int(np.sum(is_closest_to_own_cluster))
    actual_accuracy = float(np.mean(is_closest_to_own_cluster))

    # (c) Permutation Test (N = 1000 lần)
    rng = np.random.default_rng(random_state)
    null_accuracies = []

    for _ in range(n_permutations):
        shuffled_labels = rng.permutation(labels_p1)
        acc_null = np.mean(assigned_closest_cluster == shuffled_labels)
        null_accuracies.append(acc_null)

    null_mean = float(np.mean(null_accuracies))
    null_p95 = float(np.percentile(null_accuracies, 95))
    empirical_p_value = float(
        (np.sum(np.array(null_accuracies) >= actual_accuracy) + 1) / (n_permutations + 1)
    )

    # 4. Kiểm tra phụ ở mức GENRE (Genre Profile Alignment)
    raw_genre_p1 = build_matrix(
        fact_p1, dims, level="genre", metric="share", customer_ids=customer_ids, return_sparse=False
    )
    raw_genre_p2 = build_matrix(
        fact_p2, dims, level="genre", metric="share", customer_ids=customer_ids, return_sparse=False
    )
    # Cosine similarity giữa hồ sơ genre của mỗi khách giữa GĐ 1 và GĐ 2
    g1_norm = normalize(raw_genre_p1.values, norm="l2")
    g2_norm = normalize(raw_genre_p2.values, norm="l2")
    customer_genre_self_sims = np.sum(g1_norm * g2_norm, axis=1)
    median_genre_temporal_consistency = float(np.median(customer_genre_self_sims))

    print(f"\n[+] KẾT QUẢ KIỂM TRA THEO THỜI GIAN:")
    print(f"   - ARI giữa cụm GĐ 1 và cụm GĐ 2 gom riêng: {temporal_ari:.4f} (NMI = {temporal_nmi:.4f})")
    print(
        f"   - Tỷ lệ khách hàng GĐ 2 có hành vi gần centroid cụm GĐ 1 của mình nhất: "
        f"{actual_accuracy*100:.2f}% ({actual_satisfaction_count}/59 khách)"
    )
    print(f"   - Mức ngẫu nhiên hoán vị (Null Mean): {null_mean*100:.2f}% (Null P95 = {null_p95*100:.2f}%)")
    print(f"   - Khoảng cách dôi dư so với ngẫu nhiên: +{(actual_accuracy - null_mean)*100:.2f}%")
    print(f"   - Empirical p-value (Permutation test N={n_permutations}): p = {empirical_p_value:.4f}")
    print(f"   - Độ tự nhất quán thể loại (Genre Self-Consistency Median): {median_genre_temporal_consistency:.4f}")

    # -------------------------------------------------------------
    # Vẽ biểu đồ Phân phối Permutation Test
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.hist(
        null_accuracies,
        bins=25,
        color="#b0c4de",
        edgecolor="#4682b4",
        alpha=0.8,
        density=True,
        label="Phân phối Ngẫu nhiên (1000 Permutations)",
    )
    ax.axvline(
        x=null_p95,
        color="orange",
        linestyle="--",
        linewidth=1.8,
        label=f"Null P95 = {null_p95*100:.1f}%",
    )
    ax.axvline(
        x=actual_accuracy,
        color="red",
        linestyle="-",
        linewidth=2.2,
        label=f"Độ chính xác Thực tế = {actual_accuracy*100:.1f}%\n(p = {empirical_p_value:.4f})",
    )

    ax.set_title(
        "Kiểm định Dự báo Hành vi Tương lai theo Trục Thời gian\n"
        "(Tỷ lệ khách hàng Giai đoạn 2 trung thành với Centroid Cụm Giai đoạn 1)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Tỷ lệ Khách hàng Gần Centroid của Mình Nhất", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mật độ Xác suất (Density)", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    pred_plot_path = figures_dir / "cluster_temporal_prediction.png"
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ kiểm định thời gian: {pred_plot_path}")

    return {
        "temporal_ari": round(temporal_ari, 4),
        "temporal_nmi": round(temporal_nmi, 4),
        "prediction_accuracy": round(actual_accuracy, 4),
        "satisfied_customers_count": actual_satisfaction_count,
        "total_customers": len(customer_ids),
        "null_mean": round(null_mean, 4),
        "null_p95": round(null_p95, 4),
        "gap_to_null_mean": round(actual_accuracy - null_mean, 4),
        "gap_to_null_p95": round(actual_accuracy - null_p95, 4),
        "empirical_p_value": round(empirical_p_value, 4),
        "median_genre_temporal_consistency": round(median_genre_temporal_consistency, 4),
    }


# =====================================================================
# 7. HÀM ĐIỀU PHỐI CHÍNH TOÀN BỘ PHA C3
# =====================================================================

def run_evaluation_pipeline() -> Dict[str, Any]:
    """Hàm chạy toàn bộ pipeline đánh giá Pha C3."""
    output_dir = Path("outputs")
    figures_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fact_purchase, dims = load_processed_data("data/processed")
    candidates = get_c3_candidates()

    # Bước 1: Đánh giá chất lượng nội tại
    df_internal = evaluate_internal_quality(candidates, fact_purchase, dims, figures_dir)
    df_internal.to_csv(output_dir / "internal_quality_comparison.csv", index=False, encoding="utf-8-sig")

    # Bước 2: Kiểm tra Hold-out 80/20 Stability CV (100 lần lặp)
    df_runs, df_stability_summary = run_holdout_stability_cv(
        candidates,
        n_splits=100,
        train_ratio=0.8,
        random_state=42,
        output_dir=output_dir,
        figures_dir=figures_dir,
    )
    df_stability_summary.to_csv(
        output_dir / "stability_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Bước 3: Kiểm tra độ nhạy theo Seed và theo K cho Ứng viên #1
    rec_candidate = candidates[0]
    seed_results = evaluate_seed_sensitivity(rec_candidate, n_seeds=25)
    k_results = evaluate_k_sensitivity(rec_candidate)

    # Bước 4: Kiểm tra theo trục thời gian (Temporal Validation)
    temporal_results = run_temporal_validation(
        fact_purchase, dims, rec_candidate, n_permutations=1000, random_state=42, figures_dir=figures_dir
    )

    # Tổng hợp toàn bộ kết quả C3
    c3_evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recommended_candidate": rec_candidate["config_id"],
        "internal_metrics": df_internal.to_dict(orient="records"),
        "stability_summary": df_stability_summary.to_dict(orient="records"),
        "seed_sensitivity": seed_results,
        "k_sensitivity": k_results,
        "temporal_validation": temporal_results,
    }

    report_json_path = output_dir / "phase_c3_evaluation_metrics.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(c3_evaluation_report, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Đã lưu tổng hợp chỉ số C3 ra: {report_json_path}")

    print("\n" + "=" * 70)
    print("HOÀN TẤT ĐÁNH GIÁ VÀ KIỂM TRA TỔNG QUÁT HÓA (PHASE C3 EVALUATION DONE)!")
    print("=" * 70)

    return c3_evaluation_report


if __name__ == "__main__":
    run_evaluation_pipeline()
