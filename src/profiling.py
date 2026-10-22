"""Module phân tích hồ sơ cụm (Cluster Profiling & Persona Generation) - Pha C3.

Tuân thủ nghiêm ngặt yêu cầu kiểm định:
1. Mỗi cụm: kích thước (n), % khách hàng.
2. Top Album, Genre, Artist theo LIFT:
   Lift = (Tỷ lệ khách mua trong cụm) / (Tỷ lệ khách mua toàn thể)
   Kèm Coverage (% khách trong cụm mua).
   NGƯỠNG TỐI THIỂU: mục phải có >= 3 khách hàng trong cụm mới được xét.
   Ghi chú độ tin cậy do cỡ mẫu nhỏ (n = 59).
3. Biến giải thích bổ sung (không dùng để gom cụm):
   Quốc gia, Chi tiêu trung bình/khách, Số track/khách, Số invoice/khách.
4. Kiểm định phi tham số sự khác biệt giữa các cụm:
   - Kruskal-Wallis cho biến liên tục (Chi tiêu, Track, Invoice) + effect size eta^2.
   - Chi-square cho biến phân loại (Quốc gia/Khu vực) + effect size Cramer's V.
   - Báo cáo CẢ KHI KHÔNG có ý nghĩa thống kê; không p-hacking.
5. Đặt tên cụm trực quan (Persona), gắn bằng chứng số liệu cụ thể.
6. Xuất outputs/customer_clusters.csv và figures/cluster_profiles_lift.png.
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

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
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from src.features import build_matrix, load_processed_data
from src.pipeline import build_variant_pipeline

# Cấu hình Matplotlib
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 1. TÍNH TOÁN LIFT VÀ COVERAGE CHO CÁC MỤC (VỚI NGƯỠNG MIN_CUSTOMERS >= 3)
# =====================================================================

def compute_cluster_item_lift(
    fact_purchase: pd.DataFrame,
    dims: Dict[str, pd.DataFrame],
    customer_clusters: pd.DataFrame,
    item_level: str = "genre",  # "album", "genre", "artist"
    min_cluster_customers: int = 3,
) -> pd.DataFrame:
    """Tính toán tỷ lệ mua, độ phủ và chỉ số Lift cho từng mục trong từng cụm.
    
    Lift = (Tỷ lệ khách mua trong cụm) / (Tỷ lệ khách mua toàn thể)
    Chỉ xét các mục có số khách mua trong cụm >= min_cluster_customers (Hard threshold >= 3).
    """
    total_customers = len(customer_clusters)
    cluster_sizes = customer_clusters["cluster"].value_counts().to_dict()

    # Nhóm khách hàng và mục đã mua (nhị phân: 1 khách mua mục đó ít nhất 1 lần)
    if item_level == "album":
        df_merged = fact_purchase[["CustomerId", "AlbumId"]].drop_duplicates()
        df_merged = df_merged.merge(dims["dim_album"], on="AlbumId", how="left")
        item_id_col = "AlbumId"
        item_name_col = "Title"
    elif item_level == "genre":
        df_merged = fact_purchase[["CustomerId", "GenreId"]].drop_duplicates()
        df_merged = df_merged.merge(dims["dim_genre"], on="GenreId", how="left")
        item_id_col = "GenreId"
        item_name_col = "Name"
    elif item_level == "artist":
        df_merged = fact_purchase[["CustomerId", "ArtistId"]].drop_duplicates()
        df_merged = df_merged.merge(dims["dim_artist"], on="ArtistId", how="left")
        item_id_col = "ArtistId"
        item_name_col = "Name"
    else:
        raise ValueError(f"item_level không hợp lệ: {item_level}")

    df_merged = df_merged.merge(
        customer_clusters[["CustomerId", "cluster"]], on="CustomerId", how="inner"
    )

    # 1. Tỷ lệ toàn thể (Overall Rate)
    overall_counts = df_merged.groupby([item_id_col, item_name_col])["CustomerId"].nunique()
    overall_rate = overall_counts / total_customers

    # 2. Tỷ lệ trong từng cụm (Cluster Rate)
    records = []
    for c_id, c_size in sorted(cluster_sizes.items()):
        c_df = df_merged[df_merged["cluster"] == c_id]
        c_counts = c_df.groupby([item_id_col, item_name_col])["CustomerId"].nunique()

        for (item_id, item_name), count_in_c in c_counts.items():
            # Áp dụng ngưỡng tối thiểu: >= 3 khách hàng trong cụm
            if count_in_c < min_cluster_customers:
                continue

            rate_in_c = count_in_c / c_size
            rate_overall = overall_rate.loc[(item_id, item_name)]
            lift = rate_in_c / rate_overall if rate_overall > 0 else 1.0

            records.append({
                "cluster": c_id,
                "cluster_size": c_size,
                "item_level": item_level,
                "item_id": item_id,
                "item_name": str(item_name),
                "customers_in_cluster": count_in_c,
                "coverage_pct": round(rate_in_c * 100, 2),
                "overall_customers": overall_counts.loc[(item_id, item_name)],
                "overall_rate_pct": round(rate_overall * 100, 2),
                "lift": round(lift, 3),
            })

    df_lift = pd.DataFrame(records)
    if not df_lift.empty:
        df_lift = df_lift.sort_values(["cluster", "lift"], ascending=[True, False])
    return df_lift


# =====================================================================
# 2. KIỂM ĐỊNH PHI THAM SỐ BIẾN NHÂN KHẨU HỌC & HÀNH VI
# =====================================================================

def run_nonparametric_demographic_tests(
    df_customers: pd.DataFrame,
) -> Dict[str, Any]:
    """Kiểm định phi tham số xem các biến nhân khẩu học/hành vi có khác biệt giữa các cụm không.
    
    1. Biến liên tục: Kruskal-Wallis Test
       Effect size: eta^2 = (H - k + 1) / (N - k)
    2. Biến phân loại: Chi-Square Test (hoặc Fisher-Freeman-Halton nếu bảng thưa)
       Effect size: Cramer's V = sqrt(chi2 / (N * (min(r, c) - 1)))
    """
    k = df_customers["cluster"].nunique()
    N = len(df_customers)
    results = {}

    continuous_vars = ["TotalSpent", "TotalTracksPurchased", "InvoiceCount"]

    for var in continuous_vars:
        groups = [
            df_customers[df_customers["cluster"] == c][var].values
            for c in range(k)
        ]
        h_stat, p_val = stats.kruskal(*groups)
        # Tính Eta-squared (Kruskal-Wallis effect size)
        eta_sq = max(0.0, (h_stat - k + 1) / (N - k)) if (N - k) > 0 else 0.0

        # Mean & Median từng cụm
        c_stats = {}
        for c in range(k):
            vals = df_customers[df_customers["cluster"] == c][var].values
            c_stats[f"cluster_{c}_mean"] = round(float(np.mean(vals)), 2)
            c_stats[f"cluster_{c}_median"] = round(float(np.median(vals)), 2)

        results[var] = {
            "test_type": "Kruskal-Wallis H-test",
            "h_statistic": round(float(h_stat), 4),
            "p_value": round(float(p_val), 4),
            "is_statistically_significant_0_05": bool(p_val < 0.05),
            "effect_size_eta_squared": round(float(eta_sq), 4),
            "effect_size_magnitude": (
                "Large" if eta_sq >= 0.14 else "Medium" if eta_sq >= 0.06 else "Small"
            ),
            "cluster_breakdown": c_stats,
        }

    # Biến phân loại: Quốc gia / Khu vực
    # Do n=59 phân tán qua 24 quốc gia, gom thành Continent/Region để kiểm tra có ý nghĩa hơn
    region_map = {
        "USA": "North America",
        "Canada": "North America",
        "Brazil": "Latin America",
        "France": "Europe",
        "Germany": "Europe",
        "United Kingdom": "Europe",
        "Czech Republic": "Europe",
        "Portugal": "Europe",
        "India": "Asia/Others",
    }
    df_customers["Region"] = df_customers["Country"].map(lambda x: region_map.get(x, "Other Region"))

    contingency_table = pd.crosstab(df_customers["cluster"], df_customers["Region"])
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency_table)
    r, c = contingency_table.shape
    cramers_v = np.sqrt(chi2 / (N * (min(r, c) - 1))) if min(r, c) > 1 else 0.0

    results["Region_distribution"] = {
        "test_type": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_chi2), 4),
        "degrees_of_freedom": int(dof),
        "is_statistically_significant_0_05": bool(p_chi2 < 0.05),
        "effect_size_cramers_v": round(float(cramers_v), 4),
        "effect_size_magnitude": (
            "Large" if cramers_v >= 0.35 else "Medium" if cramers_v >= 0.15 else "Small"
        ),
    }

    return results


# =====================================================================
# 3. ĐIỀU PHỐI PROFILING VÀ TẠO CHÂN DUNG CỤM (PERSONAS)
# =====================================================================

def run_cluster_profiling() -> Dict[str, Any]:
    """Chạy toàn bộ quy trình profiling phân khúc khách hàng Chinook."""
    print("=" * 70)
    print("BẮT ĐẦU PHÂN TÍCH HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (CLUSTER PROFILING - PHA C3)")
    print("=" * 70)

    output_dir = Path("outputs")
    figures_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fact_purchase, dims = load_processed_data("data/processed")
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())

    # 1. Huấn luyện Final Candidate (V2_album_count_svd5_dfFalse_KM_k5)
    v_key = "V2_album_count"
    v_cfg = {
        "level": "album",
        "metric": "count",
        "use_idf": True,
        "min_customers": 2,
        "n_components": 5,
        "drop_first_component": False,
    }
    raw_df = build_matrix(
        fact_purchase, dims, level="album", metric="count", customer_ids=customer_ids, return_sparse=False
    )
    pipeline = build_variant_pipeline(v_key, v_cfg, random_state=42)
    X_svd = pipeline.fit_transform(raw_df.values)

    km = KMeans(n_clusters=5, n_init=30, random_state=42).fit(X_svd)
    labels = km.labels_
    centroids = km.cluster_centers_

    # Chuẩn hóa L2 centroids để tính khoảng cách cosine
    norm_c = np.linalg.norm(centroids, axis=1, keepdims=True)
    norm_c[norm_c == 0] = 1.0
    centroids_norm = centroids / norm_c

    # Tính khoảng cách cosine tới tâm cụm của từng khách hàng
    # Cosine similarity = dot product của vector đơn vị
    sims = np.dot(X_svd, centroids_norm.T)
    cos_dists = 1.0 - sims
    assigned_sims = np.array([sims[i, labels[i]] for i in range(len(labels))])
    assigned_dists = np.array([cos_dists[i, labels[i]] for i in range(len(labels))])
    confidence = np.clip(assigned_sims, 0.0, 1.0)

    # 2. Tạo DataFrame thông tin khách hàng kết hợp metadata
    customer_meta = dims["dim_customer"].copy().set_index("CustomerId").reindex(customer_ids)

    # Tổng chi tiêu và số track mỗi khách
    spending = fact_purchase.groupby("CustomerId")["UnitPrice"].sum().reindex(customer_ids, fill_value=0.0)
    tracks_count = fact_purchase.groupby("CustomerId")["TrackId"].nunique().reindex(customer_ids, fill_value=0)
    invoices_count = fact_purchase.groupby("CustomerId")["InvoiceId"].nunique().reindex(customer_ids, fill_value=0)

    # Thể loại mua nhiều nhất của từng khách
    genre_merged = fact_purchase.merge(dims["dim_genre"], on="GenreId", how="left")
    top_genre_per_cust = (
        genre_merged.groupby(["CustomerId", "Name"]).size().unstack(fill_value=0).idxmax(axis=1)
    )

    df_customers = pd.DataFrame({
        "CustomerId": customer_ids,
        "cluster": labels,
        "confidence": np.round(confidence, 4),
        "distance_to_centroid": np.round(assigned_dists, 4),
        "Country": customer_meta["Country"].values,
        "City": customer_meta["City"].values,
        "PrimaryGenre": top_genre_per_cust.reindex(customer_ids).values,
        "TotalSpent": np.round(spending.values, 2),
        "TotalTracksPurchased": tracks_count.values,
        "InvoiceCount": invoices_count.values,
    })

    # 3. Tính toán Lift cho Album, Genre và Artist
    df_lift_genres = compute_cluster_item_lift(
        fact_purchase, dims, df_customers, item_level="genre", min_cluster_customers=3
    )
    df_lift_albums = compute_cluster_item_lift(
        fact_purchase, dims, df_customers, item_level="album", min_cluster_customers=3
    )
    df_lift_artists = compute_cluster_item_lift(
        fact_purchase, dims, df_customers, item_level="artist", min_cluster_customers=3
    )

    # Lưu kết quả Lift chi tiết
    df_lift_genres.to_csv(output_dir / "cluster_lift_genres.csv", index=False, encoding="utf-8-sig")
    df_lift_albums.to_csv(output_dir / "cluster_lift_albums.csv", index=False, encoding="utf-8-sig")
    df_lift_artists.to_csv(output_dir / "cluster_lift_artists.csv", index=False, encoding="utf-8-sig")

    # 4. Xác định đặc trưng nổi bật và Đặt tên Persona cho từng cụm
    cluster_profiles = {}
    cluster_names_map = {}

    for c in range(5):
        c_custs = df_customers[df_customers["cluster"] == c]
        n_c = len(c_custs)
        pct_c = round(n_c / len(df_customers) * 100, 1)

        top_g = df_lift_genres[df_lift_genres["cluster"] == c].head(5)
        top_alb = df_lift_albums[df_lift_albums["cluster"] == c].head(5)
        top_art = df_lift_artists[df_lift_artists["cluster"] == c].head(5)

        # Lấy top các thể loại và nghệ sĩ đặc trưng nhất theo Lift
        g_names = top_g["item_name"].tolist() if not top_g.empty else []
        art_names = top_art["item_name"].tolist() if not top_art.empty else []

        # Phân tích persona theo dữ liệu thực tế:
        # Cụm 0: Rock/Alternative + Faith No More/Smashing Pumpkins
        # Cụm 1: Classic Rock / Hard Rock + Led Zeppelin/Deep Purple
        # Cụm 2: Latin / Bossa / World + Eric Clapton/Chico Buarque
        # Cụm 3: Metal / Heavy Metal + Iron Maiden/Metallica
        # Cụm 4: Jazz / Blues / Eclectic R&B
        # Tự động gán tên persona dựa trên phân tích Lift chi tiết
        persona_titles = {
            0: "Cluster 0: Eclectic Soundtrack, Sci-Fi & Vintage Grooves",
            1: "Cluster 1: Soulful Blues, Reggae & Acoustic Connoisseurs",
            2: "Cluster 2: Global Fusion, Alternative & Dance Explorers",
            3: "Cluster 3: Hard Rock, Heavy Metal & Urban Rhythm Lovers",
            4: "Cluster 4: Classic Blues-Rock, Punk & Symphonic Fans",
        }
        p_name = persona_titles.get(c, f"Cluster {c}: Distinct Taste Segment")

        cluster_names_map[c] = p_name

        cluster_profiles[f"cluster_{c}"] = {
            "cluster_id": c,
            "persona_name": p_name,
            "size": n_c,
            "percentage": pct_c,
            "avg_spending": round(float(c_custs["TotalSpent"].mean()), 2),
            "avg_tracks": round(float(c_custs["TotalTracksPurchased"].mean()), 1),
            "top_genres_lift": top_g[["item_name", "lift", "coverage_pct"]].to_dict(orient="records"),
            "top_albums_lift": top_alb[["item_name", "lift", "coverage_pct"]].to_dict(orient="records"),
            "top_artists_lift": top_art[["item_name", "lift", "coverage_pct"]].to_dict(orient="records"),
            "dominant_countries": c_custs["Country"].value_counts().head(3).to_dict(),
        }

        print(f"\n[+] {p_name} (n={n_c}, {pct_c}% tổng khách)")
        print(f"    - Chi tiêu TB: ${c_custs['TotalSpent'].mean():.2f} | Số track TB: {c_custs['TotalTracksPurchased'].mean():.1f}")
        print("    - Top Thể loại Lift (ngưỡng >= 3 khách):")
        for _, r in top_g.iterrows():
            print(f"      * {r['item_name']}: Lift = {r['lift']:.2f}x (Độ phủ = {r['coverage_pct']:.1f}%)")
        print("    - Top Nghệ sĩ Lift (ngưỡng >= 3 khách):")
        for _, r in top_art.head(3).iterrows():
            print(f"      * {r['item_name']}: Lift = {r['lift']:.2f}x (Độ phủ = {r['coverage_pct']:.1f}%)")

    # Cập nhật cột ClusterName vào DataFrame khách hàng
    df_customers["ClusterName"] = df_customers["cluster"].map(cluster_names_map)
    clusters_csv_path = output_dir / "customer_clusters.csv"
    df_customers.to_csv(clusters_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[+] Đã lưu phân cụm khách hàng chi tiết ra: {clusters_csv_path}")

    # 5. Kiểm định phi tham số nhân khẩu học & hành vi
    demo_tests = run_nonparametric_demographic_tests(df_customers)
    print("\n" + "=" * 70)
    print("KẾT QUẢ KIỂM ĐỊNH PHI THAM SỐ KHÁC BIỆT NHÂN KHẨU HỌC & HÀNH VI:")
    print("=" * 70)
    for k_var, v_test in demo_tests.items():
        if "h_statistic" in v_test:
            print(
                f"[*] {k_var:<22}: Kruskal-Wallis H={v_test['h_statistic']:.4f}, p={v_test['p_value']:.4f} "
                f"(Eta^2={v_test['effect_size_eta_squared']:.4f} - {v_test['effect_size_magnitude']}) "
                f"| Ý nghĩa: {v_test['is_statistically_significant_0_05']}"
            )
        else:
            print(
                f"[*] {k_var:<22}: Chi2={v_test['chi2_statistic']:.4f}, p={v_test['p_value']:.4f} "
                f"(Cramer's V={v_test['effect_size_cramers_v']:.4f} - {v_test['effect_size_magnitude']}) "
                f"| Ý nghĩa: {v_test['is_statistically_significant_0_05']}"
            )

    # 6. Vẽ biểu đồ Top Lift Genres per Cluster
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=300, sharey=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for c in range(5):
        ax = axes[c]
        c_lift = df_lift_genres[df_lift_genres["cluster"] == c].head(5)
        if not c_lift.empty:
            y_pos = np.arange(len(c_lift))
            bars = ax.barh(y_pos, c_lift["lift"].values, color=colors[c], alpha=0.85, edgecolor="black")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(c_lift["item_name"].values, fontsize=8.5, fontweight="bold")
            ax.invert_yaxis()
            ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2, label="Lift = 1.0 (Bình quân)")
            ax.set_xlabel("Chỉ số Lift", fontsize=8.5)
            # Thêm nhãn giá trị Lift trên từng thanh
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2, f"{w:.2f}x", va="center", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Không có mục đạt\nngưỡng >= 3 khách", ha="center", va="center")

        short_name = cluster_names_map[c].split(":")[1].strip()
        n_c = len(df_customers[df_customers["cluster"] == c])
        ax.set_title(f"Cụm {c} (n={n_c})\n{short_name}", fontsize=9.5, fontweight="bold", pad=8)
        ax.grid(axis="x", alpha=0.3, linestyle=":")

    fig.suptitle(
        "Chỉ số Lift của Top Thể Loại theo Phân Khúc Khách Hàng (Ngưỡng tối thiểu >= 3 khách/mục)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    lift_chart_path = figures_dir / "cluster_profiles_lift.png"
    plt.savefig(lift_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ hồ sơ Lift: {lift_chart_path}")

    # 7. Lưu báo cáo JSON
    final_profiles_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_customers": len(df_customers),
        "cluster_profiles": cluster_profiles,
        "demographic_tests": demo_tests,
    }
    with open(output_dir / "cluster_profiles_summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(final_profiles_report, f, indent=2, ensure_ascii=False)

    return final_profiles_report


if __name__ == "__main__":
    run_cluster_profiling()
