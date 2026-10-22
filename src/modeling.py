"""Module huấn luyện, tinh chỉnh (tuning) và so sánh mô hình gom cụm (Phase C2).

Tuân thủ nghiêm ngặt config/decision_rules.yaml:
- k trong khoảng 2..6.
- Loại bỏ cấu hình có cụm nhỏ nhất < 5 khách hàng.
- Chỉ xem xét cấu hình có Silhouette (cosine) vượt phân vị 95 (P95) của Null Model.
- Đánh giá trên 4 họ mô hình: KMeans, Agglomerative, GaussianMixture, SpectralClustering (+ DBSCAN).
- Đối chiếu với 4 mốc Baseline (Random, Raw Binary KMeans, Null Model, Dominant Genre, Country).
- Lưu toàn bộ cấu hình đã thử vào outputs/model_comparison.csv.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Cấu hình UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

from src.features import (
    build_composite_raw_matrix,
    build_matrix,
    get_customer_interpretation_metadata,
    load_processed_data,
)
from src.pipeline import build_variant_pipeline


# Thiết lập cấu hình matplotlib cho biểu đồ khoa học
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "cosine",
) -> Tuple[float, float, float, int, int, List[int], float]:
    """Tính toán bộ chỉ số đánh giá gom cụm tiêu chuẩn."""
    unique_labels = set(labels)
    # Loại bỏ nhãn nhiễu (-1) nếu có trong DBSCAN
    valid_labels = [l for l in unique_labels if l != -1]
    n_effective_clusters = len(valid_labels)

    if n_effective_clusters < 2:
        return -1.0, 999.0, 0.0, 0, len(labels), [len(labels)], 0.0

    # Lọc các điểm không phải nhiễu để tính metric
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    try:
        sil = float(silhouette_score(X_valid, labels_valid, metric=metric))
    except Exception:
        sil = -1.0

    try:
        db = float(davies_bouldin_score(X_valid, labels_valid))
    except Exception:
        db = 999.0

    try:
        ch = float(calinski_harabasz_score(X_valid, labels_valid))
    except Exception:
        ch = 0.0

    cluster_counts = pd.Series(labels_valid).value_counts().tolist()
    min_size = int(min(cluster_counts)) if cluster_counts else 0
    max_size = int(max(cluster_counts)) if cluster_counts else 0
    balance_ratio = round(min_size / max_size, 4) if max_size > 0 else 0.0

    return sil, db, ch, min_size, max_size, cluster_counts, balance_ratio


def evaluate_baselines(
    processed_data: Tuple[pd.DataFrame, Dict[str, pd.DataFrame]],
    feature_matrices: Dict[str, pd.DataFrame],
    reference_variant: str = "V1_album_binary",
    random_state: int = 42,
    n_random_runs: int = 100,
) -> pd.DataFrame:
    """Đánh giá 4 họ baseline tiêu chuẩn để làm mốc so sánh (benchmarks)."""
    fact_purchase, dims = processed_data
    X_ref = feature_matrices[reference_variant].values
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())

    baseline_records = []

    # ---------------------------------------------------------
    # Baseline 1: Phân nhóm theo Thể loại trội nhất (Dominant Genre)
    # ---------------------------------------------------------
    genre_counts = (
        fact_purchase.groupby(["CustomerId", "GenreId"]).size().unstack(fill_value=0)
    )
    dominant_genre = genre_counts.idxmax(axis=1).reindex(customer_ids).values
    sil_dg, db_dg, ch_dg, min_dg, max_dg, sizes_dg, bal_dg = compute_clustering_metrics(
        X_ref, dominant_genre, metric="cosine"
    )
    baseline_records.append({
        "baseline_type": "Business Heuristic",
        "baseline_name": "Dominant Genre (Top Thể Loại)",
        "k": len(set(dominant_genre)),
        "silhouette_cosine": round(sil_dg, 4),
        "davies_bouldin": round(db_dg, 4),
        "calinski_harabasz": round(ch_dg, 4),
        "min_cluster_size": min_dg,
        "max_cluster_size": max_dg,
        "cluster_sizes": str(sizes_dg),
        "balance_ratio": bal_dg,
        "description": "Gán khách theo thể loại chiếm nhiều track mua nhất (Rock: 47, Latin: 7, Metal: 5)",
    })

    # ---------------------------------------------------------
    # Baseline 2: Phân nhóm theo Quốc gia (Country Grouping)
    # ---------------------------------------------------------
    country_series = dims["dim_customer"].set_index("CustomerId")["Country"].reindex(customer_ids)
    top_countries = country_series.value_counts()[lambda x: x >= 5].index.tolist()
    grouped_country = country_series.apply(lambda c: c if c in top_countries else "Other")
    country_labels = pd.Categorical(grouped_country).codes
    sil_ct, db_ct, ch_ct, min_ct, max_ct, sizes_ct, bal_ct = compute_clustering_metrics(
        X_ref, country_labels, metric="cosine"
    )
    baseline_records.append({
        "baseline_type": "Business Heuristic",
        "baseline_name": "Country (Quốc gia gộp nhóm nhỏ)",
        "k": len(set(country_labels)),
        "silhouette_cosine": round(sil_ct, 4),
        "davies_bouldin": round(db_ct, 4),
        "calinski_harabasz": round(ch_ct, 4),
        "min_cluster_size": min_ct,
        "max_cluster_size": max_ct,
        "cluster_sizes": str(sizes_ct),
        "balance_ratio": bal_ct,
        "description": "Gán khách theo quốc gia (USA: 13, Canada: 8, Brazil: 5, France: 5, Other: 28)",
    })

    # ---------------------------------------------------------
    # Baseline 3: Gán cụm ngẫu nhiên giữ nguyên phân phối kích thước (Random Baseline)
    # ---------------------------------------------------------
    rng = np.random.default_rng(random_state)
    for k_test in [2, 3, 4, 5, 6]:
        # Phân phối kích thước cụm đều hoặc lấy từ KMeans k
        dummy_labels = np.array([i % k_test for i in range(len(customer_ids))])
        sils_rand, dbs_rand, chs_rand = [], [], []
        for _ in range(n_random_runs):
            perm_labels = rng.permutation(dummy_labels)
            s_r, d_r, c_r, _, _, _, _ = compute_clustering_metrics(
                X_ref, perm_labels, metric="cosine"
            )
            sils_rand.append(s_r)
            dbs_rand.append(d_r)
            chs_rand.append(c_r)

        baseline_records.append({
            "baseline_type": "Statistical Random",
            "baseline_name": f"Random Permutation (k={k_test})",
            "k": k_test,
            "silhouette_cosine": round(float(np.mean(sils_rand)), 4),
            "davies_bouldin": round(float(np.mean(dbs_rand)), 4),
            "calinski_harabasz": round(float(np.mean(chs_rand)), 4),
            "min_cluster_size": int(len(customer_ids) // k_test),
            "max_cluster_size": int(len(customer_ids) // k_test + 1),
            "cluster_sizes": f"Mean over {n_random_runs} runs (P95 Sil={np.percentile(sils_rand, 95):.4f})",
            "balance_ratio": 1.0,
            "description": f"Gán nhãn ngẫu nhiên độc lập k={k_test} qua {n_random_runs} lần lặp",
        })

    # ---------------------------------------------------------
    # Baseline 4: KMeans trên Ma trận Nhị phân Thô (Raw Binary KMeans)
    # ---------------------------------------------------------
    raw_album_df = build_matrix(fact_purchase, dims, level="album", metric="binary")
    X_raw_bin = raw_album_df.values
    for k_test in [2, 3, 4, 5, 6]:
        km_raw = KMeans(n_clusters=k_test, n_init=30, random_state=random_state).fit(X_raw_bin)
        s_km, d_km, c_km, min_km, max_km, sizes_km, bal_km = compute_clustering_metrics(
            X_raw_bin, km_raw.labels_, metric="cosine"
        )
        baseline_records.append({
            "baseline_type": "Raw Feature Benchmark",
            "baseline_name": f"Raw Binary KMeans (k={k_test})",
            "k": k_test,
            "silhouette_cosine": round(s_km, 4),
            "davies_bouldin": round(d_km, 4),
            "calinski_harabasz": round(c_km, 4),
            "min_cluster_size": min_km,
            "max_cluster_size": max_km,
            "cluster_sizes": str(sizes_km),
            "balance_ratio": bal_km,
            "description": "KMeans trực tiếp trên ma trận nhị phân 59x347 (không IDF, không SVD, không L2-norm)",
        })

    # ---------------------------------------------------------
    # Baseline 5: Null Model P95 Threshold (từ Pha B)
    # ---------------------------------------------------------
    baseline_records.append({
        "baseline_type": "Null Model Benchmark",
        "baseline_name": "Null Model P95 (Khách x Album Agglomerative)",
        "k": 6,
        "silhouette_cosine": 0.0301,
        "davies_bouldin": 999.0,
        "calinski_harabasz": 0.0,
        "min_cluster_size": 0,
        "max_cluster_size": 59,
        "cluster_sizes": "P95 of 300 curveball permutations",
        "balance_ratio": 0.0,
        "description": "Ngưỡng phân vị 95 của mô hình vô hiệu hoán vị bảo toàn bậc hàng và cột",
    })

    return pd.DataFrame(baseline_records)


def run_model_tuning_grid(
    features_config_path: str = "config/features.yaml",
    decision_rules_path: str = "config/decision_rules.yaml",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Chạy toàn bộ lưới huấn luyện và tinh chỉnh mô hình theo decision_rules.yaml."""
    with open(features_config_path, "r", encoding="utf-8") as f:
        feat_cfg = yaml.safe_load(f)

    with open(decision_rules_path, "r", encoding="utf-8") as f:
        rules_cfg = yaml.safe_load(f)

    output_dir = Path("outputs")
    figures_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fact_purchase, dims = load_processed_data("data/processed")
    k_range = list(range(rules_cfg["k_range"]["min_k"], rules_cfg["k_range"]["max_k"] + 1))
    min_size_threshold = rules_cfg["cluster_size_constraints"]["min_cluster_size"]
    null_p95_map = rules_cfg["null_model_thresholds"]["reference_null_p95"]

    # 1. Nạp và chuẩn bị các ma trận đặc trưng
    feature_matrices = {}
    for v_key in feat_cfg["variants"].keys():
        parquet_path = output_dir / f"feature_matrix_{v_key}.parquet"
        if parquet_path.exists():
            feature_matrices[v_key] = pd.read_parquet(parquet_path)

    # 2. Đánh giá Baselines
    print("\n>>> Đang tính toán các mốc Baseline so sánh...")
    df_baselines = evaluate_baselines(
        (fact_purchase, dims),
        feature_matrices,
        reference_variant="V1_album_binary",
        random_state=random_state,
    )
    df_baselines.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {output_dir / 'baseline_comparison.csv'}")

    # 3. Lưới thực nghiệm mô hình
    records = []
    print("\n>>> Bắt đầu quét lưới siêu tham số huấn luyện mô hình (Model Tuning Grid)...")

    # Danh sách biến thể kiểm tra
    variants_to_test = list(feat_cfg["variants"].keys())

    # Các cấu hình SVD / drop_first mở rộng để đánh giá
    grid_configurations = [
        {"svd_dim": 10, "drop_first": False},
        {"svd_dim": 10, "drop_first": True},
        {"svd_dim": 5, "drop_first": False},
    ]

    total_runs = 0
    t0_all = time.time()

    for v_key in variants_to_test:
        v_meta = feat_cfg["variants"][v_key]
        raw_df = pd.read_parquet(output_dir / f"raw_matrix_{v_key}.parquet")

        # Xác định null model threshold tương ứng với loại thực thể
        if "album" in v_key:
            null_p95_threshold = null_p95_map.get("album_agglomerative", 0.0301)
        elif "artist" in v_key:
            null_p95_threshold = null_p95_map.get("artist_agglomerative", 0.0945)
        elif "genre" in v_key:
            null_p95_threshold = null_p95_map.get("genre_agglomerative", 0.3318)
        else:
            null_p95_threshold = null_p95_map.get("album_agglomerative", 0.0301)

        for grid_cfg in grid_configurations:
            svd_dim = grid_cfg["svd_dim"]
            drop_first = grid_cfg["drop_first"]

            # Bỏ qua svd_dim=10 nếu là genre (chỉ có 24 cột)
            if v_key == "V4_genre_share" and svd_dim > 8:
                continue

            # Xây dựng pipeline với tham số tương ứng
            custom_v_cfg = dict(v_meta)
            custom_v_cfg["n_components"] = svd_dim
            custom_v_cfg["drop_first_component"] = drop_first

            pipe = build_variant_pipeline(v_key, custom_v_cfg, random_state=random_state)
            X = pipe.fit_transform(raw_df.values)

            # A. HỌ 1: KMEANS
            for k in k_range:
                t0 = time.time()
                km = KMeans(n_clusters=k, n_init=30, random_state=random_state).fit(X)
                rt = time.time() - t0
                sil, db, ch, min_s, max_s, sizes, bal = compute_clustering_metrics(
                    X, km.labels_, metric="cosine"
                )
                pass_k = (rules_cfg["k_range"]["min_k"] <= k <= rules_cfg["k_range"]["max_k"])
                pass_size = (min_s >= min_size_threshold)
                pass_null = (sil > null_p95_threshold)
                pass_all = (pass_k and pass_size and pass_null)

                records.append({
                    "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_KM_k{k}",
                    "variant": v_key,
                    "svd_components": svd_dim,
                    "drop_first_component": drop_first,
                    "model_family": "KMeans",
                    "model_name": "KMeans",
                    "k": k,
                    "model_params": f"n_init=30, random_state={random_state}",
                    "silhouette_cosine": round(sil, 4),
                    "davies_bouldin": round(db, 4),
                    "calinski_harabasz": round(ch, 4),
                    "inertia": round(float(km.inertia_), 4),
                    "bic": None,
                    "min_cluster_size": min_s,
                    "max_cluster_size": max_s,
                    "cluster_sizes": str(sizes),
                    "balance_ratio": bal,
                    "runtime_sec": round(rt, 4),
                    "null_p95_threshold": null_p95_threshold,
                    "passed_min_size": pass_size,
                    "passed_null_model": pass_null,
                    "passed_all_rules": pass_all,
                })
                total_runs += 1

            # B. HỌ 2: AGGLOMERATIVE CLUSTERING (Ward, Average, Complete)
            linkages = [("ward", "euclidean"), ("average", "cosine"), ("complete", "cosine")]
            for link, metric in linkages:
                for k in k_range:
                    t0 = time.time()
                    agg = AgglomerativeClustering(n_clusters=k, linkage=link, metric=metric).fit(X)
                    rt = time.time() - t0
                    sil, db, ch, min_s, max_s, sizes, bal = compute_clustering_metrics(
                        X, agg.labels_, metric="cosine"
                    )
                    pass_k = (rules_cfg["k_range"]["min_k"] <= k <= rules_cfg["k_range"]["max_k"])
                    pass_size = (min_s >= min_size_threshold)
                    pass_null = (sil > null_p95_threshold)
                    pass_all = (pass_k and pass_size and pass_null)

                    records.append({
                        "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_Agg_{link}_k{k}",
                        "variant": v_key,
                        "svd_components": svd_dim,
                        "drop_first_component": drop_first,
                        "model_family": "Agglomerative",
                        "model_name": f"Agglomerative_{link}",
                        "k": k,
                        "model_params": f"linkage={link}, metric={metric}",
                        "silhouette_cosine": round(sil, 4),
                        "davies_bouldin": round(db, 4),
                        "calinski_harabasz": round(ch, 4),
                        "inertia": None,
                        "bic": None,
                        "min_cluster_size": min_s,
                        "max_cluster_size": max_s,
                        "cluster_sizes": str(sizes),
                        "balance_ratio": bal,
                        "runtime_sec": round(rt, 4),
                        "null_p95_threshold": null_p95_threshold,
                        "passed_min_size": pass_size,
                        "passed_null_model": pass_null,
                        "passed_all_rules": pass_all,
                    })
                    total_runs += 1

            # C. HỌ 3: GAUSSIAN MIXTURE (diag, spherical)
            for cov_type in ["diag", "spherical"]:
                for k in k_range:
                    t0 = time.time()
                    try:
                        gmm = GaussianMixture(
                            n_components=k,
                            covariance_type=cov_type,
                            n_init=20,
                            random_state=random_state,
                        ).fit(X)
                        rt = time.time() - t0
                        labels_gmm = gmm.predict(X)
                        bic_val = round(float(gmm.bic(X)), 4)
                        sil, db, ch, min_s, max_s, sizes, bal = compute_clustering_metrics(
                            X, labels_gmm, metric="cosine"
                        )
                    except Exception:
                        rt = 0.0
                        bic_val = 9999.0
                        sil, db, ch, min_s, max_s, sizes, bal = -1.0, 999.0, 0.0, 0, 59, [59], 0.0

                    pass_k = (rules_cfg["k_range"]["min_k"] <= k <= rules_cfg["k_range"]["max_k"])
                    pass_size = (min_s >= min_size_threshold)
                    pass_null = (sil > null_p95_threshold)
                    pass_all = (pass_k and pass_size and pass_null)

                    records.append({
                        "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_GMM_{cov_type}_k{k}",
                        "variant": v_key,
                        "svd_components": svd_dim,
                        "drop_first_component": drop_first,
                        "model_family": "GaussianMixture",
                        "model_name": f"GMM_{cov_type}",
                        "k": k,
                        "model_params": f"covariance_type={cov_type}, n_init=20",
                        "silhouette_cosine": round(sil, 4),
                        "davies_bouldin": round(db, 4),
                        "calinski_harabasz": round(ch, 4),
                        "inertia": None,
                        "bic": bic_val,
                        "min_cluster_size": min_s,
                        "max_cluster_size": max_s,
                        "cluster_sizes": str(sizes),
                        "balance_ratio": bal,
                        "runtime_sec": round(rt, 4),
                        "null_p95_threshold": null_p95_threshold,
                        "passed_min_size": pass_size,
                        "passed_null_model": pass_null,
                        "passed_all_rules": pass_all,
                    })
                    total_runs += 1

            # D. HỌ 4: SPECTRAL CLUSTERING (nearest_neighbors)
            for k in k_range:
                t0 = time.time()
                try:
                    sc = SpectralClustering(
                        n_clusters=k,
                        affinity="nearest_neighbors",
                        n_neighbors=8,
                        n_init=15,
                        random_state=random_state,
                    ).fit(X)
                    rt = time.time() - t0
                    sil, db, ch, min_s, max_s, sizes, bal = compute_clustering_metrics(
                        X, sc.labels_, metric="cosine"
                    )
                except Exception:
                    rt = 0.0
                    sil, db, ch, min_s, max_s, sizes, bal = -1.0, 999.0, 0.0, 0, 59, [59], 0.0

                pass_k = (rules_cfg["k_range"]["min_k"] <= k <= rules_cfg["k_range"]["max_k"])
                pass_size = (min_s >= min_size_threshold)
                pass_null = (sil > null_p95_threshold)
                pass_all = (pass_k and pass_size and pass_null)

                records.append({
                    "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_Spectral_k{k}",
                    "variant": v_key,
                    "svd_components": svd_dim,
                    "drop_first_component": drop_first,
                    "model_family": "SpectralClustering",
                    "model_name": "Spectral_NN8",
                    "k": k,
                    "model_params": f"affinity=nearest_neighbors, n_neighbors=8, random_state={random_state}",
                    "silhouette_cosine": round(sil, 4),
                    "davies_bouldin": round(db, 4),
                    "calinski_harabasz": round(ch, 4),
                    "inertia": None,
                    "bic": None,
                    "min_cluster_size": min_s,
                    "max_cluster_size": max_s,
                    "cluster_sizes": str(sizes),
                    "balance_ratio": bal,
                    "runtime_sec": round(rt, 4),
                    "null_p95_threshold": null_p95_threshold,
                    "passed_min_size": pass_size,
                    "passed_null_model": pass_null,
                    "passed_all_rules": pass_all,
                })
                total_runs += 1

            # E. HỌ 5: DBSCAN (Density Benchmark / Rejection Rationale)
            for eps in [0.4, 0.6]:
                t0 = time.time()
                dbs = DBSCAN(eps=eps, min_samples=3, metric="cosine").fit(X)
                rt = time.time() - t0
                sil, db, ch, min_s, max_s, sizes, bal = compute_clustering_metrics(
                    X, dbs.labels_, metric="cosine"
                )
                noise_pct = round(float((dbs.labels_ == -1).sum() / len(X)), 4)
                n_clusters_dbs = len(set(dbs.labels_)) - (1 if -1 in dbs.labels_ else 0)

                pass_k = (rules_cfg["k_range"]["min_k"] <= n_clusters_dbs <= rules_cfg["k_range"]["max_k"])
                pass_size = (min_s >= min_size_threshold)
                pass_null = (sil > null_p95_threshold)
                pass_all = (pass_k and pass_size and pass_null)

                records.append({
                    "config_id": f"{v_key}_svd{svd_dim}_df{drop_first}_DBSCAN_eps{eps}",
                    "variant": v_key,
                    "svd_components": svd_dim,
                    "drop_first_component": drop_first,
                    "model_family": "DBSCAN",
                    "model_name": f"DBSCAN_eps{eps}",
                    "k": n_clusters_dbs,
                    "model_params": f"eps={eps}, min_samples=3, metric=cosine (Noise: {noise_pct*100:.1f}%)",
                    "silhouette_cosine": round(sil, 4),
                    "davies_bouldin": round(db, 4),
                    "calinski_harabasz": round(ch, 4),
                    "inertia": None,
                    "bic": None,
                    "min_cluster_size": min_s,
                    "max_cluster_size": max_s,
                    "cluster_sizes": str(sizes),
                    "balance_ratio": bal,
                    "runtime_sec": round(rt, 4),
                    "null_p95_threshold": null_p95_threshold,
                    "passed_min_size": pass_size,
                    "passed_null_model": pass_null,
                    "passed_all_rules": pass_all,
                })
                total_runs += 1

    df_results = pd.DataFrame(records)

    # 4. Lưu toàn bộ kết quả vào model_comparison.csv
    comparison_csv_path = output_dir / "model_comparison.csv"
    df_results.to_csv(comparison_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã ghi nhận toàn bộ {len(df_results)} cấu hình vào: {comparison_csv_path}")

    # 5. Xếp hạng và chọn Top 3 - 5 ứng viên theo Decision Rules
    df_valid = df_results[df_results["passed_all_rules"]].copy()
    print(f"Số cấu hình vượt qua toàn bộ tiêu chí (Passed All Rules): {len(df_valid)} / {len(df_results)}")

    if len(df_valid) == 0:
        print("\n[CẢNH BÁO] KHÔNG CÓ CẤU HÌNH NÀO VƯỢT QUA TOÀN BỘ TIÊU CHÍ.")
        top_candidates = []
    else:
        # Xếp hạng tổng hợp (Composite Ranking)
        # 1. Silhouette rank (cao -> tốt, rank giảm dần)
        df_valid["rank_sil"] = df_valid["silhouette_cosine"].rank(ascending=False)
        # 2. Davies-Bouldin rank (thấp -> tốt, rank tăng dần)
        df_valid["rank_db"] = df_valid["davies_bouldin"].rank(ascending=True)
        # 3. Calinski-Harabasz rank (cao -> tốt, rank giảm dần)
        df_valid["rank_ch"] = df_valid["calinski_harabasz"].rank(ascending=False)
        # 4. Balance ratio rank (cao -> tốt, rank giảm dần)
        df_valid["rank_bal"] = df_valid["balance_ratio"].rank(ascending=False)

        # Tổng điểm rank (càng nhỏ càng tốt)
        df_valid["composite_rank_score"] = (
            df_valid["rank_sil"] * 0.4
            + df_valid["rank_db"] * 0.2
            + df_valid["rank_ch"] * 0.2
            + df_valid["rank_bal"] * 0.2
        )

        df_sorted = df_valid.sort_values(
            by=["composite_rank_score", "silhouette_cosine", "k"],
            ascending=[True, False, True],
        )

        # Chọn top 3-5 đại diện đa dạng theo họ mô hình và biến thể
        top_candidates = []
        seen_keys = set()
        for _, row in df_sorted.iterrows():
            key = (row["model_family"], row["k"])
            if key not in seen_keys:
                top_candidates.append(row.to_dict())
                seen_keys.add(key)
            if len(top_candidates) >= 5:
                break

    # Lưu danh sách top ứng viên ra JSON
    top_candidates_path = output_dir / "top_candidates.json"
    with open(top_candidates_path, "w", encoding="utf-8") as f:
        json.dump(top_candidates, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu top ứng viên vào: {top_candidates_path}")

    # 6. Vẽ các biểu đồ trực quan hóa
    generate_modeling_plots(df_results, df_valid, figures_dir)

    print(f"\nToàn bộ quá trình quét {total_runs} cấu hình hoàn tất trong {time.time() - t0_all:.2f}s.")
    return df_results, {"top_candidates": top_candidates, "total_runs": total_runs}


def generate_modeling_plots(
    df_results: pd.DataFrame,
    df_valid: pd.DataFrame,
    figures_dir: Path,
):
    """Vẽ các đồ thị đánh giá mô hình tuân thủ quy tắc không che khuất dữ liệu."""
    # -------------------------------------------------------------
    # Hình 1: Đánh giá chỉ số (Silhouette, DB, CH) theo k cho từng họ mô hình trên V1
    # -------------------------------------------------------------
    df_v1 = df_results[
        (df_results["variant"] == "V1_album_binary")
        & (df_results["svd_components"] == 10)
        & (~df_results["drop_first_component"])
        & (df_results["model_family"].isin(["KMeans", "Agglomerative", "GaussianMixture", "SpectralClustering"]))
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Đồ thị 1.1: Silhouette Cosine theo k
    ax = axes[0]
    for model_name, grp in df_v1.groupby("model_name"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted["silhouette_cosine"], marker="o", label=model_name)
    ax.axhline(0.0301, color="red", linestyle="--", linewidth=1.2, label="Null Model P95 (0.030)")
    ax.set_title("Silhouette Score (Cosine) theo k\n(Biến thể: V1 Album Binary)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Số cụm k", fontsize=10)
    ax.set_ylabel("Silhouette Score (Cao hơn là tốt hơn)", fontsize=10)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")

    # Đồ thị 1.2: Davies-Bouldin theo k
    ax = axes[1]
    for model_name, grp in df_v1.groupby("model_name"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted["davies_bouldin"], marker="s", label=model_name)
    ax.set_title("Davies-Bouldin Index theo k\n(Biến thể: V1 Album Binary)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Số cụm k", fontsize=10)
    ax.set_ylabel("Davies-Bouldin (Thấp hơn là tốt hơn)", fontsize=10)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    # Đồ thị 1.3: Calinski-Harabasz theo k
    ax = axes[2]
    for model_name, grp in df_v1.groupby("model_name"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted["calinski_harabasz"], marker="^", label=model_name)
    ax.set_title("Calinski-Harabasz Index theo k\n(Biến thể: V1 Album Binary)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Số cụm k", fontsize=10)
    ax.set_ylabel("Calinski-Harabasz (Cao hơn là tốt hơn)", fontsize=10)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig_path1 = figures_dir / "model_selection_metrics_by_k.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ: {fig_path1}")

    # -------------------------------------------------------------
    # Hình 2: Đường cong Elbow (Inertia) của KMeans trên các biến thể chính
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    df_km = df_results[
        (df_results["model_family"] == "KMeans")
        & (df_results["svd_components"] == 10)
        & (~df_results["drop_first_component"])
    ]

    for v_key, grp in df_km.groupby("variant"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted["inertia"], marker="o", linewidth=1.8, label=v_key)

    ax.set_title("Đường cong Elbow (KMeans Inertia) theo k", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số cụm k", fontsize=11)
    ax.set_ylabel("Inertia (Tổng bình phương khoảng cách tới tâm cụm)", fontsize=11)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path2 = figures_dir / "model_selection_kmeans_elbow.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ: {fig_path2}")

    # -------------------------------------------------------------
    # Hình 3: Đường cong BIC của GaussianMixture
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    df_gmm = df_results[
        (df_results["model_family"] == "GaussianMixture")
        & (df_results["variant"] == "V1_album_binary")
        & (df_results["svd_components"] == 10)
        & (~df_results["drop_first_component"])
    ]

    for model_name, grp in df_gmm.groupby("model_name"):
        grp_sorted = grp.sort_values("k")
        ax.plot(grp_sorted["k"], grp_sorted["bic"], marker="d", linewidth=1.8, label=model_name)

    ax.set_title("Đường cong BIC của GaussianMixture theo k (V1 Album Binary)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Số thành phần k", fontsize=11)
    ax.set_ylabel("Bayesian Information Criterion (BIC - Thấp hơn là tốt hơn)", fontsize=11)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path3 = figures_dir / "model_selection_gmm_bic.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ: {fig_path3}")


if __name__ == "__main__":
    run_model_tuning_grid()
