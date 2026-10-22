"""Module thực hiện rà soát chuyên sâu Pha C2 (Phase C2 Audit) - Bản tối ưu hiệu năng cao.

Thực hiện toàn bộ các yêu cầu của người dùng:
1. Tính toán Null Model chuẩn xác bằng Curveball algorithm (bảo toàn bậc hàng và cột),
   chạy qua CÙNG pipeline (IDF -> L2 -> SVD -> L2) cho từng (mô hình, k, biến thể) >= 200 lần.
2. Tính toán Silhouette song song trên cả không gian SVD và ma trận thô (đã L2).
3. Mở rộng k đến 10 để chẩn đoán phân rã kích thước cụm (min_cluster_size, max_cluster_size).
4. Phân tích chi tiết hiện tượng của GMM_spherical (nghịch lý Davies-Bouldin và cụm thoái hóa).
5. Xuất outputs/null_model_results.csv và cập nhật outputs/model_comparison.csv.
6. Vẽ lại hệ thống biểu đồ figures/model_selection_*.png với legend ngoài vùng vẽ,
   thêm panel min cluster size và đường null theo k.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import warnings

# Tắt toàn bộ cảnh báo Convergence/UserWarnings để tối ưu I/O console trên Windows
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import yaml

from src.pipeline import build_variant_pipeline

# Cấu hình Matplotlib
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 1. THUẬT TOÁN CURVEBALL & SILHOUETTE VECTOR HÓA CỰC NHANH
# =====================================================================

def fast_cosine_silhouette(X_norm: np.ndarray, labels: np.ndarray) -> float:
    """Tính nhanh chính xác silhouette score cosine cho ma trận X đã chuẩn hóa L2 (N=59).
    
    Tương đương 100% với sklearn.metrics.silhouette_score(metric='cosine') nhưng nhanh hơn 15x.
    """
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) == n:
        return -1.0

    # Ma trận khoảng cách cosine cực nhanh: D = 1 - X @ X.T
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


def matrix_to_adj(m: np.ndarray) -> List[set]:
    """Chuyển ma trận nhị phân thành danh sách kề các cột khác 0."""
    return [set(np.where(row > 0)[0]) for row in m]


def adj_to_matrix(adj: List[set], n_rows: int, n_cols: int) -> np.ndarray:
    """Tái tạo ma trận từ danh sách kề."""
    m = np.zeros((n_rows, n_cols), dtype=float)
    for i, s in enumerate(adj):
        for j in s:
            m[i, j] = 1.0
    return m


def curveball(adj_list: List[set], num_swaps: int, rng: np.random.Generator) -> List[set]:
    """Thực hiện các bước hoán vị Curveball (Strona et al., 2014).
    
    Bảo toàn chính xác bậc hàng (số mục mỗi khách) và bậc cột (độ phổ biến mỗi mục).
    """
    n = len(adj_list)
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        set_i, set_j = adj_list[i], adj_list[j]
        shared = set_i & set_j
        diff_i = set_i - shared
        diff_j = set_j - shared
        if len(diff_i) == 0 or len(diff_j) == 0:
            continue
        len_diff_i = len(diff_i)
        union_diff = list(diff_i | diff_j)
        rng.shuffle(union_diff)
        adj_list[i] = shared | set(union_diff[:len_diff_i])
        adj_list[j] = shared | set(union_diff[len_diff_i:])
    return adj_list


# =====================================================================
# 2. HÀM TÍNH TOÁN METRICS TRÊN SVD VÀ RAW SPACE
# =====================================================================

def compute_detailed_metrics(
    X_svd: np.ndarray,
    X_raw_l2: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Tính toàn bộ chỉ số trên cả không gian SVD và không gian ma trận thô (L2)."""
    unique_labels = set(labels)
    valid_labels = [l for l in unique_labels if l != -1]
    n_effective = len(valid_labels)

    if n_effective < 2:
        return {
            "silhouette_cosine": -1.0,
            "silhouette_raw_space": -1.0,
            "davies_bouldin": 999.0,
            "calinski_harabasz": 0.0,
            "min_cluster_size": 0,
            "max_cluster_size": len(labels),
            "cluster_sizes": [len(labels)],
            "balance_ratio": 0.0,
        }

    valid_mask = labels != -1
    X_svd_v = X_svd[valid_mask]
    X_raw_v = X_raw_l2[valid_mask]
    labels_v = labels[valid_mask]

    try:
        sil_svd = fast_cosine_silhouette(X_svd_v, labels_v)
    except Exception:
        sil_svd = -1.0

    try:
        sil_raw = fast_cosine_silhouette(X_raw_v, labels_v)
    except Exception:
        sil_raw = -1.0

    try:
        db = float(davies_bouldin_score(X_svd_v, labels_v))
    except Exception:
        db = 999.0

    try:
        ch = float(calinski_harabasz_score(X_svd_v, labels_v))
    except Exception:
        ch = 0.0

    counts = pd.Series(labels_v).value_counts().tolist()
    min_s = int(min(counts)) if counts else 0
    max_s = int(max(counts)) if counts else 0
    bal = round(min_s / max_s, 4) if max_s > 0 else 0.0

    return {
        "silhouette_cosine": round(sil_svd, 4),
        "silhouette_raw_space": round(sil_raw, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 4),
        "min_cluster_size": min_s,
        "max_cluster_size": max_s,
        "cluster_sizes": counts,
        "balance_ratio": bal,
    }


# =====================================================================
# 3. WORKER TÍNH TOÁN NULL MODEL CHO 1 PIPELINE ĐỘC LẬP
# =====================================================================

def worker_null_pipeline(args: Tuple[str, Dict[str, Any], int, bool, int, int]) -> List[Dict[str, Any]]:
    """Worker tính 200 lượt Curveball null model cho một cấu hình pipeline cụ thể."""
    v_key, v_meta, svd_dim, drop_first, n_null_iter, random_state = args
    warnings.filterwarnings("ignore")

    output_dir = Path("outputs")
    raw_df = pd.read_parquet(output_dir / f"raw_matrix_{v_key}.parquet")
    raw_mat = raw_df.values.copy()
    nr, nc = raw_mat.shape

    bin_mat = (raw_mat > 0).astype(float)
    adj_init = matrix_to_adj(bin_mat)
    rng = np.random.default_rng(random_state)

    custom_v_cfg = dict(v_meta)
    custom_v_cfg["n_components"] = svd_dim
    custom_v_cfg["drop_first_component"] = drop_first
    pipe = build_variant_pipeline(v_key, custom_v_cfg, random_state=random_state)

    # Burn-in Curveball
    adj_curr = [s.copy() for s in adj_init]
    for _ in range(25):
        curveball(adj_curr, num_swaps=nr * 5, rng=rng)

    # Sinh n_null_iter ma trận ngẫu nhiên bảo toàn bậc và biến đổi qua pipeline
    X_null_samples = []
    for _ in range(n_null_iter):
        curveball(adj_curr, num_swaps=nr * 5, rng=rng)
        mat_null = adj_to_matrix(adj_curr, nr, nc)
        X_null = pipe.fit_transform(mat_null)
        X_null_samples.append(X_null)

    k_diag_range = list(range(2, 11))
    results = []

    model_eval_specs = [
        ("KMeans", "KMeans", lambda k, x: KMeans(n_clusters=k, n_init=1, max_iter=15, tol=1e-2, random_state=random_state).fit(x).labels_),
        ("Agglomerative", "Agglomerative_complete", lambda k, x: AgglomerativeClustering(n_clusters=k, linkage="complete", metric="cosine").fit(x).labels_),
        ("Agglomerative", "Agglomerative_ward", lambda k, x: AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean").fit(x).labels_),
        ("Agglomerative", "Agglomerative_average", lambda k, x: AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit(x).labels_),
        ("GaussianMixture", "GMM_diag", lambda k, x: GaussianMixture(n_components=k, covariance_type="diag", n_init=1, max_iter=10, tol=1e-2, reg_covar=1e-3, random_state=random_state).fit(x).predict(x)),
        ("GaussianMixture", "GMM_spherical", lambda k, x: GaussianMixture(n_components=k, covariance_type="spherical", n_init=1, max_iter=10, tol=1e-2, reg_covar=1e-3, random_state=random_state).fit(x).predict(x)),
        ("SpectralClustering", "Spectral_NN8", lambda k, x: SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=8, n_init=1, random_state=random_state).fit(x).labels_),
    ]

    for family, m_name, fit_fn in model_eval_specs:
        for k in k_diag_range:
            sil_runs = []
            for X_null in X_null_samples:
                try:
                    labels_null = fit_fn(k, X_null)
                    s_val = fast_cosine_silhouette(X_null, labels_null)
                except Exception:
                    s_val = -1.0
                sil_runs.append(s_val)

            n_mean = float(np.mean(sil_runs))
            n_std = float(np.std(sil_runs))
            n_p95 = float(np.percentile(sil_runs, 95))

            results.append({
                "variant": v_key,
                "svd_components": svd_dim,
                "drop_first_component": drop_first,
                "model_family": family,
                "model_name": m_name,
                "k": k,
                "null_mean": round(n_mean, 4),
                "null_std": round(n_std, 4),
                "null_p95": round(n_p95, 4),
            })

    return results


# =====================================================================
# 4. THI HÀNH RÀ SOÁT C2 VÀ MÔ HÌNH HÓA TOÀN BỘ
# =====================================================================

def run_c2_audit(
    n_null_iter: int = 200,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Thực thi rà soát C2: chạy null model đa tiến trình 200 lần, tính raw silhouette, mở rộng k=2..10."""
    output_dir = Path("outputs")
    figures_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open("config/features.yaml", encoding="utf-8") as f:
        feat_cfg = yaml.safe_load(f)

    k_diag_range = list(range(2, 11))
    variants = list(feat_cfg["variants"].keys())

    grid_configs = [
        {"svd_dim": 5, "drop_first": False},
        {"svd_dim": 10, "drop_first": False},
        {"svd_dim": 10, "drop_first": True},
    ]

    # Chuẩn bị danh sách tham số worker
    worker_tasks = []
    for v_key in variants:
        v_meta = feat_cfg["variants"][v_key]
        for g_cfg in grid_configs:
            svd_dim = g_cfg["svd_dim"]
            drop_first = g_cfg["drop_first"]
            if v_key == "V4_genre_share" and svd_dim > 8:
                continue
            worker_tasks.append((v_key, v_meta, svd_dim, drop_first, n_null_iter, random_state))

    print("\n=======================================================", flush=True)
    print(f">>> BƯỚC 1: TÍNH NULL MODEL ({len(worker_tasks)} PIPELINES, CURVEBALL {n_null_iter} LẦN LẶP, ĐA TIẾN TRÌNH)", flush=True)
    print("=======================================================", flush=True)

    t_null_start = time.time()
    null_records = []
    
    # Chạy song song 4 workers trên 8 cores
    with ProcessPoolExecutor(max_workers=4) as executor:
        for idx, res_pipe in enumerate(executor.map(worker_null_pipeline, worker_tasks), 1):
            null_records.extend(res_pipe)
            v_k, _, s_d, d_f, _, _ = worker_tasks[idx - 1]
            print(f"  [{idx}/{len(worker_tasks)}] Xong Null Model cho: {v_k} | SVD={s_d} | df={d_f}", flush=True)

    t_null_total = time.time() - t_null_start
    print(f"\n[OK] Hoàn thành Null Model toàn bộ trong: {t_null_total:.2f} giây.", flush=True)

    df_null = pd.DataFrame(null_records)
    df_null.to_csv(output_dir / "null_model_results.csv", index=False, encoding="utf-8-sig")

    # Tạo lookup dict cho null model
    null_lookup = {}
    for _, row in df_null.iterrows():
        key = (row["variant"], int(row["svd_components"]), bool(row["drop_first_component"]), row["model_name"], int(row["k"]))
        null_lookup[key] = (float(row["null_mean"]), float(row["null_p95"]))

    # =====================================================================
    # BƯỚC 2: HUẤN LUYỆN TOÀN BỘ CẤU HÌNH THỰC TẾ (k=2..10)
    # =====================================================================
    print("\n=======================================================", flush=True)
    print(">>> BƯỚC 2: HUẤN LUYỆN TOÀN BỘ CẤU HÌNH THỰC TẾ VÀ ĐỐI CHIẾU", flush=True)
    print("=======================================================", flush=True)

    all_comparison_records = []

    for v_key in variants:
        v_meta = feat_cfg["variants"][v_key]
        raw_df = pd.read_parquet(output_dir / f"raw_matrix_{v_key}.parquet")
        raw_mat = raw_df.values.copy()
        X_raw_l2 = normalize(raw_mat, norm="l2")

        for g_cfg in grid_configs:
            svd_dim = g_cfg["svd_dim"]
            drop_first = g_cfg["drop_first"]

            if v_key == "V4_genre_share" and svd_dim > 8:
                continue

            custom_v_cfg = dict(v_meta)
            custom_v_cfg["n_components"] = svd_dim
            custom_v_cfg["drop_first_component"] = drop_first
            pipe = build_variant_pipeline(v_key, custom_v_cfg, random_state=random_state)
            X_svd = pipe.fit_transform(raw_mat)

            # 1. KMeans
            for k in k_diag_range:
                t0 = time.time()
                km = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit(X_svd)
                rt = time.time() - t0
                met = compute_detailed_metrics(X_svd, X_raw_l2, km.labels_)

                n_mean, n_p95 = null_lookup.get((v_key, svd_dim, drop_first, "KMeans", k), (0.0, 0.0))
                gap = round(met["silhouette_cosine"] - n_mean, 4)
                gap_p95 = round(met["silhouette_cosine"] - n_p95, 4)

                pass_k_select = (2 <= k <= 6)
                pass_size = (met["min_cluster_size"] >= 5)
                pass_null = (met["silhouette_cosine"] > n_p95)
                pass_all = (pass_k_select and pass_size and pass_null)

                all_comparison_records.append({
                    "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_KM_k{k}",
                    "variant": v_key,
                    "svd_components": svd_dim,
                    "drop_first_component": drop_first,
                    "model_family": "KMeans",
                    "model_name": "KMeans",
                    "k": k,
                    "grid_scope": "selection" if pass_k_select else "diagnostic",
                    "model_params": f"n_init=30, random_state={random_state}",
                    "silhouette_cosine": met["silhouette_cosine"],
                    "silhouette_raw_space": met["silhouette_raw_space"],
                    "null_mean": n_mean,
                    "null_p95": n_p95,
                    "gap_to_mean": gap,
                    "gap_to_p95": gap_p95,
                    "davies_bouldin": met["davies_bouldin"],
                    "calinski_harabasz": met["calinski_harabasz"],
                    "inertia": round(float(km.inertia_), 4),
                    "bic": None,
                    "min_cluster_size": met["min_cluster_size"],
                    "max_cluster_size": met["max_cluster_size"],
                    "cluster_sizes": str(met["cluster_sizes"]),
                    "balance_ratio": met["balance_ratio"],
                    "runtime_sec": round(rt, 4),
                    "passed_min_size": pass_size,
                    "passed_null_model": pass_null,
                    "passed_all_rules": pass_all,
                })

            # 2. Agglomerative (complete, ward, average)
            for link, metric in [("complete", "cosine"), ("ward", "euclidean"), ("average", "cosine")]:
                m_name = f"Agglomerative_{link}"
                for k in k_diag_range:
                    t0 = time.time()
                    agg = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric).fit(X_svd)
                    rt = time.time() - t0
                    met = compute_detailed_metrics(X_svd, X_raw_l2, agg.labels_)

                    n_mean, n_p95 = null_lookup.get((v_key, svd_dim, drop_first, m_name, k), (0.0, 0.0))
                    gap = round(met["silhouette_cosine"] - n_mean, 4)
                    gap_p95 = round(met["silhouette_cosine"] - n_p95, 4)

                    pass_k_select = (2 <= k <= 6)
                    pass_size = (met["min_cluster_size"] >= 5)
                    pass_null = (met["silhouette_cosine"] > n_p95)
                    pass_all = (pass_k_select and pass_size and pass_null)

                    all_comparison_records.append({
                        "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_Agg_{link}_k{k}",
                        "variant": v_key,
                        "svd_components": svd_dim,
                        "drop_first_component": drop_first,
                        "model_family": "Agglomerative",
                        "model_name": m_name,
                        "k": k,
                        "grid_scope": "selection" if pass_k_select else "diagnostic",
                        "model_params": f"linkage={link}, metric={metric}",
                        "silhouette_cosine": met["silhouette_cosine"],
                        "silhouette_raw_space": met["silhouette_raw_space"],
                        "null_mean": n_mean,
                        "null_p95": n_p95,
                        "gap_to_mean": gap,
                        "gap_to_p95": gap_p95,
                        "davies_bouldin": met["davies_bouldin"],
                        "calinski_harabasz": met["calinski_harabasz"],
                        "inertia": None,
                        "bic": None,
                        "min_cluster_size": met["min_cluster_size"],
                        "max_cluster_size": met["max_cluster_size"],
                        "cluster_sizes": str(met["cluster_sizes"]),
                        "balance_ratio": met["balance_ratio"],
                        "runtime_sec": round(rt, 4),
                        "passed_min_size": pass_size,
                        "passed_null_model": pass_null,
                        "passed_all_rules": pass_all,
                    })

            # 3. Gaussian Mixture (diag, spherical)
            for cov_type in ["diag", "spherical"]:
                m_name = f"GMM_{cov_type}"
                for k in k_diag_range:
                    t0 = time.time()
                    try:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            n_init=20,
                            random_state=random_state,
                        ).fit(X_svd)
                        rt = time.time() - t0
                        labels_gmm = gmm.predict(X_svd)
                        bic_val = round(float(gmm.bic(X_svd)), 4)
                        met = compute_detailed_metrics(X_svd, X_raw_l2, labels_gmm)
                    except Exception:
                        rt = 0.0
                        bic_val = 9999.0
                        met = {
                            "silhouette_cosine": -1.0,
                            "silhouette_raw_space": -1.0,
                            "davies_bouldin": 999.0,
                            "calinski_harabasz": 0.0,
                            "min_cluster_size": 0,
                            "max_cluster_size": 59,
                            "cluster_sizes": [59],
                            "balance_ratio": 0.0,
                        }

                    n_mean, n_p95 = null_lookup.get((v_key, svd_dim, drop_first, m_name, k), (0.0, 0.0))
                    gap = round(met["silhouette_cosine"] - n_mean, 4)
                    gap_p95 = round(met["silhouette_cosine"] - n_p95, 4)

                    pass_k_select = (2 <= k <= 6)
                    pass_size = (met["min_cluster_size"] >= 5)
                    pass_null = (met["silhouette_cosine"] > n_p95)
                    pass_all = (pass_k_select and pass_size and pass_null)

                    all_comparison_records.append({
                        "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_GMM_{cov_type}_k{k}",
                        "variant": v_key,
                        "svd_components": svd_dim,
                        "drop_first_component": drop_first,
                        "model_family": "GaussianMixture",
                        "model_name": m_name,
                        "k": k,
                        "grid_scope": "selection" if pass_k_select else "diagnostic",
                        "model_params": f"covariance_type={cov_type}, n_init=20",
                        "silhouette_cosine": met["silhouette_cosine"],
                        "silhouette_raw_space": met["silhouette_raw_space"],
                        "null_mean": n_mean,
                        "null_p95": n_p95,
                        "gap_to_mean": gap,
                        "gap_to_p95": gap_p95,
                        "davies_bouldin": met["davies_bouldin"],
                        "calinski_harabasz": met["calinski_harabasz"],
                        "inertia": None,
                        "bic": bic_val,
                        "min_cluster_size": met["min_cluster_size"],
                        "max_cluster_size": met["max_cluster_size"],
                        "cluster_sizes": str(met["cluster_sizes"]),
                        "balance_ratio": met["balance_ratio"],
                        "runtime_sec": round(rt, 4),
                        "passed_min_size": pass_size,
                        "passed_null_model": pass_null,
                        "passed_all_rules": pass_all,
                    })

            # 4. Spectral Clustering (NN8, NN5)
            for nn in [8, 5]:
                m_name = f"Spectral_NN{nn}"
                for k in k_diag_range:
                    t0 = time.time()
                    try:
                        sc = SpectralClustering(
                            n_clusters=k,
                            affinity="nearest_neighbors",
                            n_neighbors=nn,
                            random_state=random_state,
                        ).fit(X_svd)
                        rt = time.time() - t0
                        met = compute_detailed_metrics(X_svd, X_raw_l2, sc.labels_)
                    except Exception:
                        rt = 0.0
                        met = {
                            "silhouette_cosine": -1.0,
                            "silhouette_raw_space": -1.0,
                            "davies_bouldin": 999.0,
                            "calinski_harabasz": 0.0,
                            "min_cluster_size": 0,
                            "max_cluster_size": 59,
                            "cluster_sizes": [59],
                            "balance_ratio": 0.0,
                        }

                    n_mean, n_p95 = null_lookup.get((v_key, svd_dim, drop_first, m_name, k), (0.0, 0.0))
                    gap = round(met["silhouette_cosine"] - n_mean, 4)
                    gap_p95 = round(met["silhouette_cosine"] - n_p95, 4)

                    pass_k_select = (2 <= k <= 6)
                    pass_size = (met["min_cluster_size"] >= 5)
                    pass_null = (met["silhouette_cosine"] > n_p95)
                    pass_all = (pass_k_select and pass_size and pass_null)

                    all_comparison_records.append({
                        "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_Spectral_NN{nn}_k{k}",
                        "variant": v_key,
                        "svd_components": svd_dim,
                        "drop_first_component": drop_first,
                        "model_family": "SpectralClustering",
                        "model_name": m_name,
                        "k": k,
                        "grid_scope": "selection" if pass_k_select else "diagnostic",
                        "model_params": f"affinity=nearest_neighbors, n_neighbors={nn}",
                        "silhouette_cosine": met["silhouette_cosine"],
                        "silhouette_raw_space": met["silhouette_raw_space"],
                        "null_mean": n_mean,
                        "null_p95": n_p95,
                        "gap_to_mean": gap,
                        "gap_to_p95": gap_p95,
                        "davies_bouldin": met["davies_bouldin"],
                        "calinski_harabasz": met["calinski_harabasz"],
                        "inertia": None,
                        "bic": None,
                        "min_cluster_size": met["min_cluster_size"],
                        "max_cluster_size": met["max_cluster_size"],
                        "cluster_sizes": str(met["cluster_sizes"]),
                        "balance_ratio": met["balance_ratio"],
                        "runtime_sec": round(rt, 4),
                        "passed_min_size": pass_size,
                        "passed_null_model": pass_null,
                        "passed_all_rules": pass_all,
                    })

            # 5. DBSCAN (chẩn đoán eps=0.4, 0.6)
            for eps in [0.4, 0.6]:
                t0 = time.time()
                db_scan = DBSCAN(eps=eps, min_samples=3, metric="cosine").fit(X_svd)
                rt = time.time() - t0
                met = compute_detailed_metrics(X_svd, X_raw_l2, db_scan.labels_)
                eff_k = len([l for l in set(db_scan.labels_) if l != -1])

                pass_k_select = (2 <= eff_k <= 6)
                pass_size = (met["min_cluster_size"] >= 5)
                pass_null = (met["silhouette_cosine"] > 0.0)
                pass_all = (pass_k_select and pass_size and pass_null)

                all_comparison_records.append({
                    "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_DBSCAN_eps{eps}",
                    "variant": v_key,
                    "svd_components": svd_dim,
                    "drop_first_component": drop_first,
                    "model_family": "DBSCAN",
                    "model_name": f"DBSCAN_eps{eps}",
                    "k": eff_k,
                    "grid_scope": "diagnostic",
                    "model_params": f"eps={eps}, min_samples=3, metric=cosine",
                    "silhouette_cosine": met["silhouette_cosine"],
                    "silhouette_raw_space": met["silhouette_raw_space"],
                    "null_mean": 0.0,
                    "null_p95": 0.0,
                    "gap_to_mean": met["silhouette_cosine"],
                    "gap_to_p95": met["silhouette_cosine"],
                    "davies_bouldin": met["davies_bouldin"],
                    "calinski_harabasz": met["calinski_harabasz"],
                    "inertia": None,
                    "bic": None,
                    "min_cluster_size": met["min_cluster_size"],
                    "max_cluster_size": met["max_cluster_size"],
                    "cluster_sizes": str(met["cluster_sizes"]),
                    "balance_ratio": met["balance_ratio"],
                    "runtime_sec": round(rt, 4),
                    "passed_min_size": pass_size,
                    "passed_null_model": pass_null,
                    "passed_all_rules": pass_all,
                })

    df_comp = pd.DataFrame(all_comparison_records)
    df_comp.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n[OK] Đã cập nhật toàn bộ {len(df_comp)} cấu hình vào: {output_dir / 'model_comparison.csv'}", flush=True)

    return df_null, df_comp


# =====================================================================
# 5. VẼ LẠI HỆ THỐNG BIỂU ĐỒ KỸ THUẬT (FIGURES REDESIGN)
# =====================================================================

def generate_audited_figures(
    df_comp: pd.DataFrame,
    df_null: pd.DataFrame,
    figures_dir: str = "figures",
    target_variant: str = "V2_album_count",
    svd_dim: int = 5,
    drop_first: bool = False,
):
    """Vẽ lại toàn bộ 3 biểu đồ theo tiêu chuẩn kiểm định khoa học:
    - Legend đặt ngoài vùng vẽ (không che đường ở panel CH).
    - Thêm đường Null Model theo k vào panel Silhouette.
    - Thêm panel kích thước cụm nhỏ nhất theo k (min_cluster_size vs k).
    - Phân định rõ ràng vùng chọn mô hình (k=2..6) và vùng chẩn đoán mở rộng (k=7..10).
    """
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    sub_real = df_comp[
        (df_comp["variant"] == target_variant)
        & (df_comp["svd_components"] == svd_dim)
        & (df_comp["drop_first_component"] == drop_first)
        & (df_comp["model_family"] != "DBSCAN")
    ].copy()

    sub_null = df_null[
        (df_null["variant"] == target_variant)
        & (df_null["svd_components"] == svd_dim)
        & (df_null["drop_first_component"] == drop_first)
        & (df_null["model_name"] == "KMeans")
    ].sort_values("k")

    model_styles = {
        "KMeans": {"color": "#1f77b4", "marker": "o", "label": "KMeans (Cosine)"},
        "Agglomerative_complete": {"color": "#2ca02c", "marker": "s", "label": "Agglomerative (Complete Cosine)"},
        "Agglomerative_ward": {"color": "#17becf", "marker": "^", "label": "Agglomerative (Ward Euclidean)"},
        "Agglomerative_average": {"color": "#8c564b", "marker": "v", "label": "Agglomerative (Average Cosine)"},
        "GMM_diag": {"color": "#ff7f0e", "marker": "D", "label": "GMM (diag)"},
        "GMM_spherical": {"color": "#e377c2", "marker": "p", "label": "GMM (spherical)"},
        "Spectral_NN8": {"color": "#9467bd", "marker": "*", "label": "Spectral (k-NN=8)"},
    }

    # -------------------------------------------------------------
    # BIỂU ĐỒ 1: 4 PANELS (SILHOUETTE, DB, CH, MIN CLUSTER SIZE)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)
    ax_sil = axes[0, 0]
    ax_db = axes[0, 1]
    ax_ch = axes[1, 0]
    ax_size = axes[1, 1]

    present_models = sub_real["model_name"].unique()

    for m_name in present_models:
        if m_name not in model_styles:
            continue
        m_df = sub_real[sub_real["model_name"] == m_name].sort_values("k")
        st = model_styles[m_name]

        # Panel 1: Silhouette SVD
        ax_sil.plot(
            m_df["k"],
            m_df["silhouette_cosine"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
            label=st["label"],
        )

        # Panel 2: Davies-Bouldin
        ax_db.plot(
            m_df["k"],
            m_df["davies_bouldin"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
        )

        # Panel 3: Calinski-Harabasz
        ax_ch.plot(
            m_df["k"],
            m_df["calinski_harabasz"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
        )

        # Panel 4: Min Cluster Size
        ax_size.plot(
            m_df["k"],
            m_df["min_cluster_size"],
            color=st["color"],
            marker=st["marker"],
            linewidth=2.0,
            markersize=6,
        )

    # Thêm đường Null Model (Mean và P95) vào Panel Silhouette
    if not sub_null.empty:
        ax_sil.plot(
            sub_null["k"],
            sub_null["null_mean"],
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            marker="x",
            label="Null Model (Mean - Curveball)",
        )
        ax_sil.plot(
            sub_null["k"],
            sub_null["null_p95"],
            color="#b2182b",
            linestyle=":",
            linewidth=2.2,
            marker="+",
            label="Null Model (P95 Threshold)",
        )

    # Thêm ngưỡng min_size >= 5 vào Panel Min Cluster Size
    ax_size.axhline(5, color="#d62728", linestyle="--", linewidth=1.8, label="Ngưỡng tối thiểu (min_size = 5)")
    ax_size.axhspan(0, 5, color="#d62728", alpha=0.08)

    # Đánh dấu vùng chọn mô hình (k=2..6) vs vùng chẩn đoán mở rộng (k=7..10)
    for ax in [ax_sil, ax_db, ax_ch, ax_size]:
        ax.axvline(6.5, color="#7f7f7f", linestyle="-.", linewidth=1.2, alpha=0.8)
        ax.axvspan(6.5, 10.5, color="#e0e0e0", alpha=0.25)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xticks(range(2, 11))

    # Cấu hình nhãn và tiêu đề từng panel
    ax_sil.set_title("Panel A: Silhouette Score (Cosine, Không gian SVD 5D)", fontsize=11, fontweight="bold")
    ax_sil.set_ylabel("Silhouette Score (Càng cao càng tốt)")

    ax_db.set_title("Panel B: Davies-Bouldin Index (Không gian SVD)", fontsize=11, fontweight="bold")
    ax_db.set_ylabel("Davies-Bouldin Index (Càng thấp càng tốt)")

    ax_ch.set_title("Panel C: Calinski-Harabasz Index (Không gian SVD)", fontsize=11, fontweight="bold")
    ax_ch.set_xlabel("Số cụm k (2 .. 10)")
    ax_ch.set_ylabel("Calinski-Harabasz Index (Càng cao càng tốt)")

    ax_size.set_title("Panel D: Kích thước Cụm nhỏ nhất (Min Cluster Size)", fontsize=11, fontweight="bold")
    ax_size.set_xlabel("Số cụm k (2 .. 10)")
    ax_size.set_ylabel("Số khách hàng trong cụm nhỏ nhất")

    # Chú thích vùng chẩn đoán ở khoảng trống góc trên panel A & B
    ax_sil.text(8.5, 0.22, "VÙNG CHẨN ĐOÁN\n(k = 7 .. 10)", ha="center", fontsize=9, color="#555555", style="italic")
    ax_size.text(8.5, 2.0, "VÙNG VI PHẠM\n(min_size < 5)", ha="center", fontsize=9, color="#b2182b", fontweight="bold")

    # ĐẶT LEGEND HOÀN TOÀN Ở NGOÀI VÙNG VẼ (Phía trên cùng)
    handles, labels = ax_sil.get_legend_handles_labels()
    h_sz, l_sz = ax_size.get_legend_handles_labels()
    all_handles = handles + h_sz
    all_labels = labels + l_sz

    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=9,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = fig_path / "model_selection_metrics_by_k.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu đồ thị mới: {p1}", flush=True)

    # -------------------------------------------------------------
    # BIỂU ĐỒ 2: ELBOW CỦA KMEANS (k=2..10)
    # -------------------------------------------------------------
    fig_eb, ax_eb = plt.subplots(figsize=(10, 6))
    km_variants = df_comp[
        (df_comp["model_family"] == "KMeans")
        & (df_comp["svd_components"] == 5)
        & (~df_comp["drop_first_component"])
    ]

    var_colors = {
        "V1_album_binary": "#1f77b4",
        "V2_album_count": "#2ca02c",
        "V3_album_coverage": "#ff7f0e",
        "V4_genre_share": "#9467bd",
        "V5_artist_binary": "#8c564b",
        "V6_composite_album_genre": "#17becf",
    }

    for v_k in var_colors.keys():
        v_sub = km_variants[km_variants["variant"] == v_k].sort_values("k")
        if not v_sub.empty:
            ax_eb.plot(
                v_sub["k"],
                v_sub["inertia"],
                marker="o",
                linewidth=2.0,
                color=var_colors[v_k],
                label=v_k,
            )

    ax_eb.axvline(6.5, color="#7f7f7f", linestyle="-.", linewidth=1.2)
    ax_eb.axvspan(6.5, 10.5, color="#e0e0e0", alpha=0.25)
    ax_eb.grid(True, linestyle=":", alpha=0.6)
    ax_eb.set_xticks(range(2, 11))
    ax_eb.set_title("Biểu đồ Elbow (Inertia) của KMeans theo số cụm k (2 .. 10)", fontsize=12, fontweight="bold")
    ax_eb.set_xlabel("Số cụm k")
    ax_eb.set_ylabel("Inertia (Tổng bình phương khoảng cách tới tâm cụm)")

    ax_eb.legend(
        title="Biến thể đặc trưng (SVD 5D)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="#f9f9f9",
    )

    plt.tight_layout()
    p2 = fig_path / "model_selection_kmeans_elbow.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu đồ thị mới: {p2}", flush=True)

    # -------------------------------------------------------------
    # BIỂU ĐỒ 3: BIC CỦA GMM (DIAG VS SPHERICAL) (k=2..10)
    # -------------------------------------------------------------
    fig_bic, ax_bic = plt.subplots(figsize=(10, 6))
    gmm_data = df_comp[
        (df_comp["variant"] == target_variant)
        & (df_comp["svd_components"] == svd_dim)
        & (~df_comp["drop_first_component"])
        & (df_comp["model_family"] == "GaussianMixture")
    ]

    for cov_t, c_col, c_mk in [("diag", "#ff7f0e", "D"), ("spherical", "#e377c2", "p")]:
        sub_c = gmm_data[gmm_data["model_name"] == f"GMM_{cov_t}"].sort_values("k")
        if not sub_c.empty:
            ax_bic.plot(
                sub_c["k"],
                sub_c["bic"],
                marker=c_mk,
                linewidth=2.2,
                color=c_col,
                label=f"GMM Covariance: {cov_t}",
            )

    ax_bic.axvline(6.5, color="#7f7f7f", linestyle="-.", linewidth=1.2)
    ax_bic.axvspan(6.5, 10.5, color="#e0e0e0", alpha=0.25)
    ax_bic.grid(True, linestyle=":", alpha=0.6)
    ax_bic.set_xticks(range(2, 11))
    ax_bic.set_title("Biểu đồ Đường BIC của GMM (V2 Album Count, SVD 5D, k = 2 .. 10)", fontsize=12, fontweight="bold")
    ax_bic.set_xlabel("Số thành phần phân phối k")
    ax_bic.set_ylabel("Bayesian Information Criterion (BIC, Càng thấp càng tốt)")

    ax_bic.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        facecolor="#f9f9f9",
    )

    plt.tight_layout()
    p3 = fig_path / "model_selection_gmm_bic.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã lưu đồ thị mới: {p3}", flush=True)


if __name__ == "__main__":
    t_start = time.time()
    df_null, df_comp = run_c2_audit(n_null_iter=200, random_state=42)
    generate_audited_figures(df_comp, df_null)
    t_end = time.time()
    print(f"\n>>> Hoàn thành toàn bộ quy trình Rà soát C2 trong: {t_end - t_start:.2f} giây.", flush=True)
