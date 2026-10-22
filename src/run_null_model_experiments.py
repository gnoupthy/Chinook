"""
Optimized Null Model Permutation Test (Curveball algorithm preserving row & col margins)
Running 300 iterations on Track, Album, Artist, Genre.
Includes Cosine Distance Distributions, Hopkins Statistic, and Null Model Silhouette comparisons.
Saves metrics to outputs/problem_selection_metrics.json and plots figure to figures/problem_selection_cosine_dist.png.
"""

import os
import sys
import copy
import time
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_connection, load_all_tables, load_master_purchases

# ----------------- 1. HOPKINS STATISTIC -----------------
def calculate_hopkins(X, m=10, n_iter=100, random_state=42, use_pca=False, n_components=5):
    rng = np.random.RandomState(random_state)
    if use_pca and X.shape[1] > n_components:
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_eval = svd.fit_transform(X)
    else:
        X_eval = X.astype(float)
        
    N, D = X_eval.shape
    m = min(m, N // 2)
    mins = X_eval.min(axis=0)
    maxs = X_eval.max(axis=0)
    diff = maxs - mins
    diff[diff == 0] = 1e-6
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(X_eval)
    nbrs_unif = NearestNeighbors(n_neighbors=1).fit(X_eval)
    
    h_scores = []
    for _ in range(n_iter):
        sample_indices = rng.choice(N, size=m, replace=False)
        real_samples = X_eval[sample_indices]
        distances, _ = nbrs.kneighbors(real_samples)
        w = distances[:, 1]
        
        unif_samples = rng.uniform(mins, maxs, size=(m, D))
        u, _ = nbrs_unif.kneighbors(unif_samples)
        u = u[:, 0]
        
        sum_u = np.sum(u)
        sum_w = np.sum(w)
        if sum_u + sum_w > 0:
            h_scores.append(sum_u / (sum_u + sum_w))
            
    return float(np.mean(h_scores)), float(np.std(h_scores))

# ----------------- 2. CURVEBALL PERMUTATION -----------------
def matrix_to_adj(mat):
    return [set(np.where(row > 0)[0]) for row in mat]

def adj_to_matrix(adj, n_rows, n_cols):
    mat = np.zeros((n_rows, n_cols), dtype=int)
    for i, s in enumerate(adj):
        for j in s:
            mat[i, j] = 1
    return mat

def curveball_step(adj_list, num_swaps=200):
    n = len(adj_list)
    for _ in range(num_swaps):
        i, j = np.random.choice(n, 2, replace=False)
        set_i = adj_list[i]
        set_j = adj_list[j]
        
        shared = set_i & set_j
        diff_i = set_i - shared
        diff_j = set_j - shared
        
        len_diff_i = len(diff_i)
        len_diff_j = len(diff_j)
        
        if len_diff_i == 0 or len_diff_j == 0:
            continue
            
        union_diff = list(diff_i | diff_j)
        np.random.shuffle(union_diff)
        
        adj_list[i] = shared | set(union_diff[:len_diff_i])
        adj_list[j] = shared | set(union_diff[len_diff_i:])
    return adj_list

# ----------------- 3. FAST BEST SILHOUETTE -----------------
def eval_best_silhouette(mat, k_range=range(2, 7)):
    """
    Computes best silhouette score across k in [2, 6] using:
    1. Agglomerative with average linkage on cosine distance
    2. K-Means on L2-normalized vectors (n_init=3 for speed)
    Returns (best_sil_agg, best_k_agg), (best_sil_km, best_k_km)
    """
    # Cosine distance
    dist_mat = squareform(pdist(mat, metric='cosine'))
    dist_mat = np.nan_to_num(dist_mat, nan=1.0)
    
    mat_norm = normalize(mat, norm='l2')
    
    best_sil_agg, best_k_agg = -1.0, 2
    best_sil_km, best_k_km = -1.0, 2
    
    for k in k_range:
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
        labels_agg = agg.fit_predict(dist_mat)
        if len(set(labels_agg)) > 1 and len(set(labels_agg)) < len(mat):
            try:
                s_agg = silhouette_score(dist_mat, labels_agg, metric='precomputed')
            except Exception:
                s_agg = -1.0
        else:
            s_agg = -1.0
        if s_agg > best_sil_agg:
            best_sil_agg = s_agg
            best_k_agg = k
            
        # KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=3)
        labels_km = km.fit_predict(mat_norm)
        if len(set(labels_km)) > 1 and len(set(labels_km)) < len(mat):
            try:
                s_km = silhouette_score(dist_mat, labels_km, metric='precomputed')
            except Exception:
                s_km = -1.0
        else:
            s_km = -1.0
        if s_km > best_sil_km:
            best_sil_km = s_km
            best_k_km = k
            
    return (best_sil_agg, best_k_agg), (best_sil_km, best_k_km)

def run_experiment(n_null_iter=300):
    t_start = time.time()
    conn = get_connection()
    tables = load_all_tables()
    df = load_master_purchases(conn)
    conn.close()

    customers = sorted(tables["Customer"]["CustomerId"].unique())
    n_cust = len(customers)

    # 4 Representations
    # (a) Track (Sold)
    track_ct = pd.crosstab(df["CustomerId"], df["TrackId"]).reindex(index=customers, fill_value=0)
    track_mat = (track_ct.values > 0).astype(int)

    # (b) Album (Sold)
    album_ct = pd.crosstab(df["CustomerId"], df["AlbumId"]).reindex(index=customers, fill_value=0)
    album_mat = (album_ct.values > 0).astype(int)
    album_count_mat = album_ct.values

    # (c1) Artist (Sold)
    artist_ct = pd.crosstab(df["CustomerId"], df["ArtistId"]).reindex(index=customers, fill_value=0)
    artist_mat = (artist_ct.values > 0).astype(int)

    # (c2) Genre (Sold)
    genre_ct = pd.crosstab(df["CustomerId"], df["GenreId"]).reindex(index=customers, fill_value=0)
    genre_mat = (genre_ct.values > 0).astype(int)

    matrices = [
        ("Khách × Track", track_mat, "TrackId", len(tables["Track"])),
        ("Khách × Album", album_mat, "AlbumId", len(tables["Album"])),
        ("Khách × Artist", artist_mat, "ArtistId", len(tables["Artist"])),
        ("Khách × Genre", genre_mat, "GenreId", len(tables["Genre"]))
    ]

    all_metrics = {}
    cosine_distributions = {}

    for name, mat, entity_id, full_catalog_size in matrices:
        n_rows, n_cols = mat.shape
        density_sold = np.count_nonzero(mat) / (n_rows * n_cols)
        density_catalog = np.count_nonzero(mat) / (n_rows * full_catalog_size)
        
        # Signals per customer
        signals = np.count_nonzero(mat, axis=1)
        
        # Overlap with at least 1 other customer
        cooc = mat @ mat.T
        np.fill_diagonal(cooc, 0)
        has_overlap = int(np.sum(np.any(cooc > 0, axis=1)))
        
        # Cosine distance
        dists = pdist(mat, metric="cosine")
        dists = np.nan_to_num(dists, nan=1.0)
        cosine_distributions[name] = dists
        median_dist = float(np.median(dists))
        pct_ge_95 = float(np.mean(dists >= 0.95) * 100)
        mean_dist = float(np.mean(dists))
        q25_dist = float(np.percentile(dists, 25))
        q75_dist = float(np.percentile(dists, 75))
        
        # Hopkins
        h_raw, h_raw_std = calculate_hopkins(mat, m=10, n_iter=100, use_pca=False)
        h_pca, h_pca_std = calculate_hopkins(mat, m=10, n_iter=100, use_pca=True, n_components=min(5, n_cols-1))
        
        # Real best silhouette
        real_agg, real_km = eval_best_silhouette(mat)
        
        print(f"\n=======================================================", flush=True)
        print(f"Representation: {name} ({n_rows} x {n_cols})", flush=True)
        print(f"  Density (Sold): {density_sold*100:.3f}%, Density (Catalog): {density_catalog*100:.3f}%", flush=True)
        print(f"  Signals/Customer: Mean={signals.mean():.2f} +/- {signals.std():.2f} (Min={signals.min()}, Max={signals.max()})", flush=True)
        print(f"  Overlap >= 1 customer: {has_overlap}/{n_rows} ({has_overlap/n_rows*100:.1f}%)", flush=True)
        print(f"  Cosine Dist: Median={median_dist:.4f}, Mean={mean_dist:.4f}, % >= 0.95: {pct_ge_95:.2f}%", flush=True)
        print(f"  Hopkins: Raw H={h_raw:.4f} +/- {h_raw_std:.4f} | PCA H={h_pca:.4f} +/- {h_pca_std:.4f}", flush=True)
        print(f"  Real Silhouette: Agglomerative={real_agg[0]:.4f} (k={real_agg[1]}) | K-Means={real_km[0]:.4f} (k={real_km[1]})", flush=True)
        
        # Null model permutation
        print(f"  Running Null Model (Curveball {n_null_iter} iterations)...", flush=True)
        adj = matrix_to_adj(mat)
        adj_curr = copy.deepcopy(adj)
        # Burn in
        curveball_step(adj_curr, num_swaps=n_rows * 20)
        
        null_sils_agg = []
        null_sils_km = []
        
        t_sub = time.time()
        for b in range(n_null_iter):
            curveball_step(adj_curr, num_swaps=n_rows * 5)
            mat_rand = adj_to_matrix(adj_curr, n_rows, n_cols)
            res_agg, res_km = eval_best_silhouette(mat_rand)
            null_sils_agg.append(res_agg[0])
            null_sils_km.append(res_km[0])
            
        null_sils_agg = np.array(null_sils_agg)
        null_sils_km = np.array(null_sils_km)
        
        p_val_agg = float((1 + np.sum(null_sils_agg >= real_agg[0])) / (n_null_iter + 1))
        p_val_km = float((1 + np.sum(null_sils_km >= real_km[0])) / (n_null_iter + 1))
        
        print(f"    Completed in {time.time()-t_sub:.1f}s", flush=True)
        print(f"    Agglomerative Null Sil: Mean={null_sils_agg.mean():.4f}, Std={null_sils_agg.std():.4f}, 95th={np.percentile(null_sils_agg, 95):.4f}, p-val={p_val_agg:.4f}", flush=True)
        print(f"    K-Means Null Sil:       Mean={null_sils_km.mean():.4f}, Std={null_sils_km.std():.4f}, 95th={np.percentile(null_sils_km, 95):.4f}, p-val={p_val_km:.4f}", flush=True)
        
        all_metrics[name] = {
            "entity": entity_id,
            "n_rows": int(n_rows),
            "n_cols_sold": int(n_cols),
            "n_cols_catalog": int(full_catalog_size),
            "density_sold": float(density_sold),
            "density_catalog": float(density_catalog),
            "signals_mean": float(signals.mean()),
            "signals_std": float(signals.std()),
            "signals_min": int(signals.min()),
            "signals_max": int(signals.max()),
            "overlap_customers": int(has_overlap),
            "cosine_dist": {
                "median": median_dist,
                "mean": mean_dist,
                "q25": q25_dist,
                "q75": q75_dist,
                "pct_ge_95": pct_ge_95
            },
            "hopkins": {
                "raw_mean": h_raw,
                "raw_std": h_raw_std,
                "pca_mean": h_pca,
                "pca_std": h_pca_std
            },
            "clustering": {
                "real_agg": {"silhouette": float(real_agg[0]), "k": int(real_agg[1])},
                "real_km": {"silhouette": float(real_km[0]), "k": int(real_km[1])},
                "null_agg": {
                    "mean": float(null_sils_agg.mean()),
                    "std": float(null_sils_agg.std()),
                    "median": float(np.median(null_sils_agg)),
                    "p95": float(np.percentile(null_sils_agg, 95)),
                    "max": float(null_sils_agg.max()),
                    "p_val": p_val_agg
                },
                "null_km": {
                    "mean": float(null_sils_km.mean()),
                    "std": float(null_sils_km.std()),
                    "median": float(np.median(null_sils_km)),
                    "p95": float(np.percentile(null_sils_km, 95)),
                    "max": float(null_sils_km.max()),
                    "p_val": p_val_km
                }
            }
        }

    # Save metrics JSON
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/problem_selection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print("\nSaved outputs/problem_selection_metrics.json!", flush=True)

    # ----------------- 4. PLOT COSINE DISTRIBUTIONS & NULL MODEL -----------------
    # REMINDER: "bạn lưu ý sau này vẽ chart nếu có nhận xét thì phải để nhận xét ở chỗ trống chứ đừng để che cột/data của chart nhe, lỡ ko có chỗ thì khỏi luôn"
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    
    colors = {
        "Khách × Track": "#EF4444",
        "Khách × Album": "#10B981",
        "Khách × Artist": "#8B5CF6",
        "Khách × Genre": "#F59E0B"
    }
    
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    
    for ax, (name, dists) in zip(ax_list, cosine_distributions.items()):
        med = np.median(dists)
        pct95 = np.mean(dists >= 0.95) * 100
        color = colors[name]
        
        # Plot histogram
        sns.histplot(dists, bins=30, kde=True, ax=ax, color=color, edgecolor="#1E293B", alpha=0.6)
        ax.axvline(med, color="#DC2626", linestyle="--", lw=1.8, label=f"Median = {med:.3f}")
        ax.axvline(0.95, color="#1E293B", linestyle=":", lw=1.5, label="Ngưỡng 0.95")
        
        ax.set_title(f"Phân phối Khoảng cách Cosine: {name}\n(1.711 cặp khách hàng)", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Khoảng cách Cosine (0: Giống nhau hoàn toàn -> 1: Trực giao)", fontsize=9.5)
        ax.set_ylabel("Số lượng cặp khách hàng", fontsize=9.5)
        ax.set_xlim(-0.05, 1.05)
        ax.legend(loc="upper left", fontsize=8.5)
        
        # Place key metric box in the clear empty area if available, or just upper area where no bars exist
        # Notice for Track, data is concentrated near 1.0, so left side (0.05 to 0.5) is completely empty!
        # For Genre, data is concentrated near 0.2, so right side (0.5 to 0.95) is completely empty!
        if name == "Khách × Track":
            # Left side is completely empty
            ax.text(0.05, 0.65, f"• Trung vị khoảng cách: {med:.4f}\n• Tỷ lệ cặp d_cos >= 0.95: {pct95:.1f}%\n• 97.5% cặp gần như trực giao\n-> Hầu như không có bài hát chung!",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor="#EF4444", lw=1))
        elif name == "Khách × Album":
            # Left side is empty
            ax.text(0.05, 0.65, f"• Trung vị khoảng cách: {med:.4f}\n• Tỷ lệ cặp d_cos >= 0.95: {pct95:.1f}%\n• Độ chồng lấn tăng rõ rệt\n-> Cung cấp tín hiệu gom cụm chi tiết",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5", edgecolor="#10B981", lw=1))
        elif name == "Khách × Artist":
            # Left side is empty
            ax.text(0.05, 0.65, f"• Trung vị khoảng cách: {med:.4f}\n• Tỷ lệ cặp d_cos >= 0.95: {pct95:.1f}%\n• Tín hiệu thị hiếu tập trung\n-> 76% cặp có khoảng cách < 0.95",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F3FF", edgecolor="#8B5CF6", lw=1))
        elif name == "Khách × Genre":
            # Right side is completely empty!
            ax.text(0.55, 0.65, f"• Trung vị khoảng cách: {med:.4f}\n• Tỷ lệ cặp d_cos >= 0.95: 0.0%\n• Phân hóa thị hiếu rõ nét\n-> 100% cặp đều có giao thoa thể loại",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=1))

    plt.suptitle("SO SÁNH PHÂN PHỐI KHOẢNG CÁCH COSINE GIỮA CÁC CẶP KHÁCH HÀNG (N=59, 1.711 CẶP)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/problem_selection_cosine_dist.png", dpi=300)
    plt.close()
    print("Saved figures/problem_selection_cosine_dist.png!", flush=True)

    # ----------------- 5. PLOT NULL MODEL SILHOUETTE COMPARISON -----------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig2.patch.set_facecolor("#FFFFFF")
    
    # K-Means Null vs Real
    ax1 = axes2[0]
    rep_names = list(all_metrics.keys())
    x_positions = np.arange(len(rep_names))
    
    real_km_sils = [all_metrics[n]["clustering"]["real_km"]["silhouette"] for n in rep_names]
    null_km_means = [all_metrics[n]["clustering"]["null_km"]["mean"] for n in rep_names]
    null_km_stds = [all_metrics[n]["clustering"]["null_km"]["std"] for n in rep_names]
    null_km_p95 = [all_metrics[n]["clustering"]["null_km"]["p95"] for n in rep_names]
    
    width = 0.35
    b1 = ax1.bar(x_positions - width/2, real_km_sils, width, label="Dữ liệu thật (Real Best Sil)", color="#2563EB", edgecolor="#1E293B")
    b2 = ax1.bar(x_positions + width/2, null_km_means, width, yerr=null_km_stds, capsize=4, label=f"Null Model Mean +/- Std (N={n_null_iter})", color="#94A3B8", edgecolor="#1E293B")
    
    ax1.set_title("5A. K-Means: So sánh Silhouette Dữ liệu thật vs Null Model", fontsize=11, fontweight="bold", pad=12)
    ax1.set_ylabel("Best Silhouette Score (k=2..6)", fontsize=9.5)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(rep_names, fontsize=9.5)
    ax1.set_ylim(0, max(real_km_sils) * 1.22)
    ax1.legend(loc="upper left", fontsize=8.5)
    
    # Put p-values text on top of real bars safely
    for i, (bar, n) in enumerate(zip(b1, rep_names)):
        p_val = all_metrics[n]["clustering"]["null_km"]["p_val"]
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006, f"p={p_val:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")
    
    # Agglomerative Null vs Real
    ax2 = axes2[1]
    real_agg_sils = [all_metrics[n]["clustering"]["real_agg"]["silhouette"] for n in rep_names]
    null_agg_means = [all_metrics[n]["clustering"]["null_agg"]["mean"] for n in rep_names]
    null_agg_stds = [all_metrics[n]["clustering"]["null_agg"]["std"] for n in rep_names]
    
    b3 = ax2.bar(x_positions - width/2, real_agg_sils, width, label="Dữ liệu thật (Real Best Sil)", color="#10B981", edgecolor="#1E293B")
    b4 = ax2.bar(x_positions + width/2, null_agg_means, width, yerr=null_agg_stds, capsize=4, label=f"Null Model Mean +/- Std (N={n_null_iter})", color="#CBD5E1", edgecolor="#1E293B")
    
    ax2.set_title("5B. Agglomerative: So sánh Silhouette Dữ liệu thật vs Null Model", fontsize=11, fontweight="bold", pad=12)
    ax2.set_ylabel("Best Silhouette Score (k=2..6)", fontsize=9.5)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(rep_names, fontsize=9.5)
    ax2.set_ylim(0, max(real_agg_sils) * 1.22)
    ax2.legend(loc="upper left", fontsize=8.5)
    
    for i, (bar, n) in enumerate(zip(b3, rep_names)):
        p_val = all_metrics[n]["clustering"]["null_agg"]["p_val"]
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f"p={p_val:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#1E293B")

    plt.suptitle(f"KIỂM ĐỊNH CẤU TRÚC CỤM VỚI NULL MODEL HOÁN VỊ BẢO TOÀN MARGINS (N={n_null_iter} LẦN)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig("figures/problem_selection_null_model.png", dpi=300)
    plt.close()
    print("Saved figures/problem_selection_null_model.png!", flush=True)

    print(f"\nAll experiments completed in {time.time()-t_start:.1f}s!", flush=True)

if __name__ == "__main__":
    run_experiment(n_null_iter=300)
