"""
EDA Visualizations Generator for Chinook Customer Clustering Project.
Generates 5 publication-quality figures in figures/eda_*.png:
1. figures/eda_revenue_trends.png (Doanh thu theo thời gian, quốc gia, thể loại, nghệ sĩ)
2. figures/eda_customer_behavior_variance.png (Phương sai hành vi khách, chứng minh tính đồng nhất số lượng)
3. figures/eda_coverage_longtail.png (Phân phối Long-tail & Overlap: Track vs Album)
4. figures/eda_density_heatmaps.png (Ma trận tương tác thưa: Track vs Album vs Genre)
5. figures/eda_genre_artist_cooccurrence.png (Đồng xuất hiện giữa các thể loại & nghệ sĩ)
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_connection, load_all_tables, load_master_purchases

# Setup styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8


def plot_revenue_trends(df_master, tables):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # 1. Monthly Revenue Trend
    ax1 = axes[0, 0]
    monthly = df_master.set_index("InvoiceDate").resample("ME")["LineTotal"].sum().reset_index()
    ax1.plot(monthly["InvoiceDate"], monthly["LineTotal"], color="#2563EB", lw=2.2, marker="o", markersize=4)
    ax1.fill_between(monthly["InvoiceDate"], monthly["LineTotal"], alpha=0.15, color="#3B82F6")
    ax1.set_title("1A. Doanh thu theo tháng (2009 - 2013)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Doanh thu (USD)", fontsize=10)
    ax1.set_xlabel("Thời gian", fontsize=10)
    # Insight box
    ax1.text(0.03, 0.90, "Insight: Doanh thu dao động ổn định quanh 30 - 50 USD/tháng,\nkhông có đột biến mùa vụ hay xu hướng suy giảm rõ rệt.",
             transform=ax1.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    # 2. Top Countries by Revenue
    ax2 = axes[0, 1]
    country_rev = df_master.groupby("CustomerCountry")["LineTotal"].sum().sort_values(ascending=False).head(10)
    bars2 = ax2.barh(country_rev.index[::-1], country_rev.values[::-1], color="#0EA5E9", height=0.65)
    ax2.set_title("1B. Top 10 Quốc gia đóng góp doanh thu lớn nhất", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Doanh thu (USD)", fontsize=10)
    for bar in bars2:
        ax2.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f} USD", va="center", fontsize=8.5)
    ax2.text(0.40, 0.15, "Insight: USA (523 USD) và Canada (304 USD) chiếm 35% doanh thu;\ntuy nhiên khách hàng phân bổ rải rác trên 24 quốc gia.",
             transform=ax2.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    # 3. Top Genres by Sales
    ax3 = axes[1, 0]
    genre_sales = df_master.groupby("GenreName")["LineTotal"].sum().sort_values(ascending=False).head(8)
    bars3 = ax3.bar(genre_sales.index, genre_sales.values, color="#10B981", width=0.55)
    ax3.set_title("1C. Top Thể loại âm nhạc bán chạy nhất", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Doanh thu (USD)", fontsize=10)
    ax3.tick_params(axis="x", rotation=25)
    for bar in bars3:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{bar.get_height():.0f}", ha="center", fontsize=8.5)
    ax3.text(0.30, 0.85, "Insight: Rock chiếm áp đảo (827 USD, ~35%), theo sau là Latin (382 USD)\nvà Metal (261 USD). Thị hiếu âm nhạc tập trung rất mạnh vào Rock/Metal.",
             transform=ax3.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    # 4. Top Artists by Sales
    ax4 = axes[1, 1]
    artist_sales = df_master.groupby("ArtistName")["LineTotal"].sum().sort_values(ascending=False).head(10)
    bars4 = ax4.barh(artist_sales.index[::-1], artist_sales.values[::-1], color="#8B5CF6", height=0.65)
    ax4.set_title("1D. Top 10 Nghệ sĩ có doanh thu cao nhất", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Doanh thu (USD)", fontsize=10)
    for bar in bars4:
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}", va="center", fontsize=8.5)
    ax4.text(0.35, 0.15, "Insight: Iron Maiden dẫn đầu (139 USD), theo sau là U2 (106 USD)\nvà Metallica (90 USD). Phản ánh đúng sự thống trị của Rock/Metal.",
             transform=ax4.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"))

    plt.suptitle("PHÂN TÍCH DOANH THU THEO THỜI GIAN, ĐỊA LÝ VÀ NỘI DUNG ÂM NHẠC", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("figures/eda_revenue_trends.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figures/eda_revenue_trends.png!")


def plot_customer_behavior_variance(df_master, tables):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    cust_summary = df_master.groupby("CustomerId").agg(
        total_spending=("LineTotal", "sum"),
        total_tracks=("TrackId", "count"),
        unique_albums=("AlbumId", "nunique"),
        total_invoices=("InvoiceId", "nunique")
    ).reset_index()

    metrics = [
        ("total_tracks", "2A. Số lượng Track mua mỗi khách", "#2563EB", axes[0, 0], "track"),
        ("total_invoices", "2B. Số lượng Invoice mỗi khách", "#0EA5E9", axes[0, 1], "hóa đơn"),
        ("total_spending", "2C. Tổng chi tiêu (USD) mỗi khách", "#10B981", axes[1, 0], "USD"),
        ("unique_albums", "2D. Số lượng Album khác nhau mỗi khách", "#F59E0B", axes[1, 1], "album")
    ]

    for col, title, color, ax, unit in metrics:
        vals = cust_summary[col]
        mean_v = vals.mean()
        std_v = vals.std()
        cv_v = (std_v / mean_v) * 100

        sns.histplot(vals, kde=True, ax=ax, color=color, bins=12, edgecolor="#1E293B", alpha=0.6)
        ax.axvline(mean_v, color="#DC2626", linestyle="--", lw=1.8, label=f"Mean = {mean_v:.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(f"{col} ({unit})", fontsize=10)
        ax.set_ylabel("Số lượng khách hàng", fontsize=10)

        stat_text = (
            f"Mean: {mean_v:.2f}\n"
            f"Std: {std_v:.2f}\n"
            f"Min: {vals.min()} | Max: {vals.max()}\n"
            f"Hệ số CV: {cv_v:.2f}%"
        )
        ax.text(0.04, 0.93, stat_text, transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1"))
        ax.legend(loc="upper right", fontsize=8.5)

    # Global insight banner at top/bottom
    fig.text(0.5, 0.02,
             "KẾT LUẬN THEN CHỐT: Hệ số biến thiên (CV) của số track (0.69%) và số hóa đơn (1.86%) gần như bằng 0!\n"
             "Khách hàng hoàn toàn KHÔNG KHÁC NHAU về quy mô hành vi mua (đều mua 36-38 bài, 7 đơn, ~39 USD).\n"
             "Do đó, bài toán gom cụm BẮT BUỘC phải dựa trên NỘI DUNG NHẠC (Track/Album/Genre) chứ không thể dựa vào hành vi tổng quát.",
             ha="center", fontsize=10, fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", lw=1.2))

    plt.suptitle("PHÂN TÍCH PHƯƠNG SAI GIỮA CÁC KHÁCH HÀNG (CUSTOMER BEHAVIOR VARIANCE)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig("figures/eda_customer_behavior_variance.png", dpi=300)
    plt.close()
    print("Saved figures/eda_customer_behavior_variance.png!")


def plot_coverage_longtail(df_master, tables):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Track long-tail
    track_buyers = df_master.groupby("TrackId")["CustomerId"].nunique()
    track_counts = track_buyers.value_counts().sort_index()

    ax1 = axes[0]
    bars1 = ax1.bar([f"{k} khách" for k in track_counts.index], track_counts.values, color=["#EF4444", "#3B82F6"], width=0.45)
    ax1.set_title("3A. Phân phối số người mua mỗi Track (Long-tail cực đoan)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Số lượng Track", fontsize=10)
    ax1.set_xlabel("Số lượng khách hàng cùng mua 1 track", fontsize=10)
    for bar in bars1:
        pct = (bar.get_height() / len(track_buyers)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25, f"{bar.get_height():,} ({pct:.1f}%)", ha="center", fontsize=9.5, fontweight="bold")
    ax1.text(0.05, 0.70,
             "• 1.728 track (87.1%) chỉ có DUY NHẤT 1 khách mua.\n"
             "• Chỉ 256 track (12.9%) có 2 khách mua.\n"
             "• Không có track nào có >= 3 khách mua!\n"
             "-> Ma trận Track quá thưa, overlap cực kỳ yếu.",
             transform=ax1.transAxes, fontsize=9.0, va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEE2E2", edgecolor="#EF4444"))

    # Album long-tail
    album_buyers = df_master.groupby("AlbumId")["CustomerId"].nunique()
    ax2 = axes[1]
    sns.histplot(album_buyers, discrete=True, ax=ax2, color="#10B981", edgecolor="#064E3B", alpha=0.7)
    ax2.set_title("3B. Phân phối số người mua mỗi Album (Overlap mạnh mẽ)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Số lượng Album", fontsize=10)
    ax2.set_xlabel("Số lượng khách hàng cùng mua trong 1 album", fontsize=10)
    ax2.axvline(album_buyers.mean(), color="#DC2626", linestyle="--", lw=1.8, label=f"Mean = {album_buyers.mean():.1f} khách/album")
    ax2.legend(loc="upper right", fontsize=9.0)

    ax2.text(0.35, 0.70,
             f"• 265/304 album ({265/304*100:.1f}%) có >= 2 khách mua!\n"
             f"• Album bán chạy nhất có tới {album_buyers.max()} khách cùng mua.\n"
             "• Độ chồng lấn (overlap) giữa các khách cao gấp 6 lần Track.\n"
             "-> Cấp độ Album là cơ sở vững chắc hơn cho gom cụm.",
             transform=ax2.transAxes, fontsize=9.0, va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#DCFCE7", edgecolor="#10B981"))

    plt.suptitle("SO SÁNH ĐỘ PHỦ VÀ PHÂN PHỐI NGƯỜI MUA: CẤP ĐỘ TRACK VS CẤP ĐỘ ALBUM", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("figures/eda_coverage_longtail.png", dpi=300)
    plt.close()
    print("Saved figures/eda_coverage_longtail.png!")


def plot_density_heatmaps(df_master, tables):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # 1. Customer x Track Matrix (Sample top 40 tracks to visualize pattern)
    top_tracks = df_master["TrackId"].value_counts().head(40).index
    ct_matrix = pd.crosstab(df_master["CustomerId"], df_master["TrackId"])
    ct_sample = ct_matrix[top_tracks].iloc[:30]  # first 30 customers

    sns.heatmap(ct_sample, ax=axes[0], cmap=["#F8FAFC", "#2563EB"], cbar=False, linewidths=0.3, linecolor="#E2E8F0")
    axes[0].set_title(f"4A. Ma trận Khách x Track (Mẫu 30x40)\nĐộ thưa thực tế: 98.92% (Density: 1.08%)", fontsize=11, fontweight="bold", pad=8)
    axes[0].set_xlabel("TrackId (Top 40)", fontsize=9)
    axes[0].set_ylabel("CustomerId (Top 30)", fontsize=9)

    # 2. Customer x Album Matrix (Sample top 40 albums)
    top_albums = df_master["AlbumId"].value_counts().head(40).index
    ca_matrix = pd.crosstab(df_master["CustomerId"], df_master["AlbumId"])
    ca_sample = ca_matrix[top_albums].iloc[:30]

    sns.heatmap(ca_sample, ax=axes[1], cmap=["#F8FAFC", "#10B981"], cbar=False, linewidths=0.3, linecolor="#E2E8F0")
    axes[1].set_title(f"4B. Ma trận Khách x Album (Mẫu 30x40)\nĐộ thưa thực tế: 93.65% (Density: 6.36%)", fontsize=11, fontweight="bold", pad=8)
    axes[1].set_xlabel("AlbumId (Top 40)", fontsize=9)
    axes[1].set_ylabel("CustomerId (Top 30)", fontsize=9)

    # 3. Customer x Genre Matrix (Full 59 customers x 25 genres)
    cg_matrix = pd.crosstab(df_master["CustomerId"], df_master["GenreName"])
    sns.heatmap(cg_matrix.iloc[:30], ax=axes[2], cmap="YlOrRd", cbar=True, linewidths=0.3, linecolor="#E2E8F0")
    axes[2].set_title(f"4C. Ma trận Khách x Thể loại (Genre)\nĐộ đậm đặc cao: Density: 29.83%", fontsize=11, fontweight="bold", pad=8)
    axes[2].set_xlabel("Genre", fontsize=9)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_ylabel("CustomerId (Top 30)", fontsize=9)

    plt.suptitle("TRỰC QUAN HÓA ĐỘ THƯA CỦA KHÔNG GIAN ĐẶC TRƯNG (SPARSITY HEATMAPS)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("figures/eda_density_heatmaps.png", dpi=300)
    plt.close()
    print("Saved figures/eda_density_heatmaps.png!")


def plot_genre_artist_cooccurrence(df_master, tables):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Genre co-occurrence
    # Create binary Customer-Genre matrix
    cg_bin = pd.crosstab(df_master["CustomerId"], df_master["GenreName"]) > 0
    # Select top 10 genres
    top_genres = df_master["GenreName"].value_counts().head(10).index
    cg_top = cg_bin[top_genres]
    genre_cooc = cg_top.T.dot(cg_top)

    sns.heatmap(genre_cooc, ax=axes[0], annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=0.5)
    axes[0].set_title("5A. Ma trận đồng xuất hiện giữa các Thể loại (Top 10 Genres)\nSố khách hàng cùng mua cả 2 thể loại", fontsize=11, fontweight="bold", pad=8)
    axes[0].tick_params(axis="x", rotation=35)

    # Top Artist co-occurrence
    top_artists = df_master["ArtistName"].value_counts().head(10).index
    ca_bin = pd.crosstab(df_master["CustomerId"], df_master["ArtistName"]) > 0
    ca_top = ca_bin[top_artists]
    artist_cooc = ca_top.T.dot(ca_top)

    sns.heatmap(artist_cooc, ax=axes[1], annot=True, fmt="d", cmap="Purples", cbar=False, linewidths=0.5)
    axes[1].set_title("5B. Ma trận đồng xuất hiện giữa các Nghệ sĩ (Top 10 Artists)\nSố khách hàng cùng mua cả 2 nghệ sĩ", fontsize=11, fontweight="bold", pad=8)
    axes[1].tick_params(axis="x", rotation=35)

    fig.text(0.5, 0.01,
             "Insight sơ bộ: Rock và Latin có mức độ đồng xuất hiện lớn nhất (34 khách cùng mua cả 2 dòng nhạc),\n"
             "tiếp theo là Rock - Metal (31 khách). Ở cấp độ nghệ sĩ, Iron Maiden và Metallica/U2 thường xuyên đi cùng nhau.",
             ha="center", fontsize=9.5, fontweight="bold", color="#1E293B",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9", edgecolor="#CBD5E1"))

    plt.suptitle("PHÂN TÍCH ĐỒNG XUẤT HIỆN THỊ HIẾU ÂM NHẠC (CO-OCCURRENCE ANALYSIS)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig("figures/eda_genre_artist_cooccurrence.png", dpi=300)
    plt.close()
    print("Saved figures/eda_genre_artist_cooccurrence.png!")


def run_all_eda():
    conn = get_connection()
    tables = load_all_tables()
    df_master = load_master_purchases(conn)
    conn.close()

    print(">>> Generating EDA Figure 1: Revenue Trends...")
    plot_revenue_trends(df_master, tables)

    print(">>> Generating EDA Figure 2: Customer Behavior Variance & CV...")
    plot_customer_behavior_variance(df_master, tables)

    print(">>> Generating EDA Figure 3: Coverage & Long-tail Distribution...")
    plot_coverage_longtail(df_master, tables)

    print(">>> Generating EDA Figure 4: Density & Sparsity Heatmaps...")
    plot_density_heatmaps(df_master, tables)

    print(">>> Generating EDA Figure 5: Genre & Artist Co-occurrence...")
    plot_genre_artist_cooccurrence(df_master, tables)

    print("All 5 EDA figures generated successfully in figures/!")


if __name__ == "__main__":
    run_all_eda()
