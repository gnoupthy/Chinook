# BẢN BÀN GIAO TOÀN DIỆN DỰ ÁN (PROJECT HANDOFF REPORT)
**Dự án:** Gom cụm khách hàng Chinook theo thị hiếu âm nhạc (Chinook Customer Music Taste Clustering)  
**Mục đích tài liệu:** Bản tóm lược cô đọng toàn bộ cốt truyện khoa học, số liệu kiểm chứng, hình ảnh và gợi ý dàn ý chi tiết để tác giả bắt đầu viết báo cáo / làm slide thuyết trình.  
**Kho lưu trữ mã nguồn:** [https://github.com/gnoupthy/Chinook.git](https://github.com/gnoupthy/Chinook.git) (Nhánh `main`)  
**Tập tin mô hình:** [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib)  

---

## 1. CỐT TRUYỆN BÁO CÁO (THE CORE PROJECT NARRATIVE)
*Khi viết báo cáo, hãy dẫn dắt theo mạch tư duy logic khoa học sau:*

1. **Khởi đầu (Bối cảnh & Thách thức ban đầu):**
   - Nền tảng bán lẻ âm nhạc số Chinook muốn phân khúc 59 khách hàng để cá nhân hóa trang chủ và đóng gói album (bundles).
   - Tuy nhiên, dữ liệu Chinook có đặc thù: **100% khách hàng đồng nhất tuyệt đối về lượng chi tiêu và số đơn** (~39.5 USD, ~7 đơn, ~38 bài, hệ số biến thiên $\text{CV} < 8\%$). Mô hình RFM truyền thống hoàn toàn mất tác dụng.
   - Hơn nữa, ma trận tương tác Khách hàng $\times$ Bài hát cực kỳ thưa (**98.92%** độ thưa; **87.1%** bài hát chỉ có đúng 1 người mua; không có bài nào có $\ge 3$ người mua). 97.49% các cặp khách hàng có khoảng cách trực giao ($d_{\cos} = 1.0$).

2. **Nút thắt & Nghiên cứu định lượng (Phase B - Null Model):**
   - Không thể phân cụm trực tiếp trên bài hát vì thuật toán sẽ gom cụm trên nhiễu ngẫu nhiên ($p = 0.0797 > 0.05$ trước Null Model).
   - Chiếu lên cấp **Album** giúp tăng tín hiệu chồng lấn gấp 6 lần. Kiểm định hoán vị vô hiệu **Curveball Null Model ($N=300$ lần)** chứng minh không gian Khách hàng $\times$ Album có cấu trúc cụm thực sự vượt trội so với ngẫu nhiên ($p = 0.0033 \ll 0.05$).

3. **Giải pháp Kỹ thuật (Phase C1 & C2):**
   - Xây dựng Pipeline chuẩn: $\text{Raw Count Matrix} \to \text{ItemFilter}(\ge 2) \to \text{IDFTransformer} \to \text{L2 Norm} \to \text{TruncatedSVD}(5D) \to \text{L2 Norm} \to \text{KMeans}(k=5)$.
   - Khóa quy tắc quyết định trước tại `config/decision_rules.yaml`. Thử nghiệm lưới 1.184 cấu hình.
   - Mô hình chiến thắng: `V2_album_count_svd5_dfFalse_KM_k5`, đạt Silhouette SVD = **0.5941** (vượt Null P95 = **0.4441**, khoảng cách dôi dư **+0.1500**).

4. **Đánh giá Đa chiều & Sự trung thực khoa học (Phase C3):**
   - **Độ nhạy ngẫu nhiên:** Hội tụ hoàn hảo qua 25 seeds ($\text{ARI} = 0.9945$).
   - **Độ ổn định mẫu con (Hold-out CV 100 lần):** Median $\text{ARI} = \mathbf{0.4038}$. Thuộc ngưỡng ổn định vừa phải ($0.40 - 0.60$), phản ánh tính biến động của tập dữ liệu nhỏ ($n=59$).
   - **Kiểm định theo thời gian (Quá khứ vs Tương lai):** Khách hàng không mua lại cùng album đã sở hữu ($p=0.9640$), nhưng thị hiếu ở cấp thể loại (Genre) lại cực kỳ nhất quán ($\text{Cosine} = 0.5065, p < 0.001$).
   - **Công suất thống kê:** Khẳng định trung thực rằng với $n=59$, **A/B testing tiêu chuẩn là bất khả thi**; đề xuất thử nghiệm pilot nhỏ (10-12 khách/cụm) và chuyển dịch lâu dài sang Item-Item Collaborative Filtering theo thể loại.

---

## 2. BẢNG TRA CỨU SỐ LIỆU "VÀNG" KHI VIẾT BÁO CÁO (KEY METRICS CHEAT-SHEET)

| Chỉ số / Thông tin | Con số chính xác | Nguồn dẫn chứng trong repo |
| :--- | :---: | :--- |
| **Quy mô CSDL Chinook** | 59 khách, 412 hóa đơn, 2.240 dòng giao dịch | [`reports/data_dictionary.md`](file:///e:/ML/Chinook/reports/data_dictionary.md) |
| **Tổng doanh thu hệ thống** | **2.328,60 USD** (2.129 dòng $0.99, 111 dòng $1.99) | [`reports/data_quality.md`](file:///e:/ML/Chinook/reports/data_quality.md) |
| **Độ thưa ma trận Khách x Track** | **98.916%** (Độ đậm đặc 1.084%, đúng 2.240 ô) | [`reports/eda_findings.md`](file:///e:/ML/Chinook/reports/eda_findings.md): Phát hiện 3 |
| **Phân phối Long-tail Track** | **87.10%** bài (1.728/1.984) chỉ 1 người mua; max = 2 khách | [`figures/eda_coverage_longtail.png`](file:///e:/ML/Chinook/figures/eda_coverage_longtail.png) |
| **Chi tiêu trung bình mỗi khách** | **39.47 USD** (Std = 2.91, Min = 36.64, Max = 49.62, CV = 7.38%) | [`figures/eda_customer_behavior_variance.png`](file:///e:/ML/Chinook/figures/eda_customer_behavior_variance.png) |
| **Số track trung bình mỗi khách** | **37.97 bài** (Std = 0.26, Min = 36, Max = 38, CV = 0.69%) | [`reports/eda_findings.md`](file:///e:/ML/Chinook/reports/eda_findings.md): Phát hiện 1 |
| **Số hóa đơn trung bình mỗi khách**| **6.98 hóa đơn** (Std = 0.13, Min = 6, Max = 7, CV = 1.86%) | [`reports/eda_findings.md`](file:///e:/ML/Chinook/reports/eda_findings.md): Phát hiện 1 |
| **Doanh thu 3 thể loại dẫn đầu** | Rock: 826.65 USD (35.5%); Latin: 382.14 USD (16.4%); Metal: 261.36 USD (11.2%) | [`figures/eda_revenue_trends.png`](file:///e:/ML/Chinook/figures/eda_revenue_trends.png) |
| **Tỷ lệ cặp khách trực giao (Track)**| **97.49%** (1.668 / 1.711 cặp có $d_{\cos} \ge 0.95$) | [`reports/problem_selection.md`](file:///e:/ML/Chinook/reports/problem_selection.md): Bảng 1 |
| **Tỷ lệ cặp khách trực giao (Album)**| **60.84%** (1.041 cặp; overlap tăng gấp nhiều lần) | [`figures/problem_selection_cosine_dist.png`](file:///e:/ML/Chinook/figures/problem_selection_cosine_dist.png) |
| **Null Model p-value (Album)** | **$p = 0.0033 \ll 0.05$** (Đạt ý nghĩa thống kê vượt trội) | [`figures/problem_selection_null_model.png`](file:///e:/ML/Chinook/figures/problem_selection_null_model.png) |
| **Mô hình tối ưu lựa chọn** | `V2_album_count_svd5_dfFalse_KM_k5` (KMeans $k=5$) | [`reports/model_card.md`](file:///e:/ML/Chinook/reports/model_card.md) |
| **Silhouette SVD (Mô hình chọn)** | **0.5941** (vs Null P95 = **0.4441**, Gap dôi dư = **+0.1500**) | [`outputs/internal_quality_comparison.csv`](file:///e:/ML/Chinook/outputs/internal_quality_comparison.csv) |
| **Silhouette ma trận thô (347D)** | **0.0665** (Phản ánh đúng độ thưa thực tế $>97\%$) | [`reports/c2_audit.md`](file:///e:/ML/Chinook/reports/c2_audit.md): Mục 2 |
| **Davies-Bouldin / Calinski-Harabasz**| **0.8234** (Rất nén) / **32.54** (Phân tách tốt) | [`outputs/internal_quality_comparison.csv`](file:///e:/ML/Chinook/outputs/internal_quality_comparison.csv) |
| **Phân bổ kích thước 5 cụm** | `[15, 12, 12, 10, 10]` (Tỷ lệ cân đối = **0.6667**, min = 10 $\ge$ 5) | [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv) |
| **Độ nhạy Random Seed (25 seeds)** | Mean Pairwise $\text{ARI} = \mathbf{0.9945}$, Median $\text{ARI} = \mathbf{1.0000}$ | [`reports/evaluation.md`](file:///e:/ML/Chinook/reports/evaluation.md): Mục 3.3 |
| **Hold-out 80/20 CV (100 lần lặp)**| Median $\text{ARI} = \mathbf{0.4038}$ (P5 = 0.0782, P95 = 0.7966) | [`outputs/stability_summary.csv`](file:///e:/ML/Chinook/outputs/stability_summary.csv) |
| **Độ chính xác dự báo tương lai** | Album-level: **11.86%** ($p = 0.9640$); Genre-level: Cosine = **0.5065** ($p < 0.001$) | [`figures/cluster_temporal_prediction.png`](file:///e:/ML/Chinook/figures/cluster_temporal_prediction.png) |
| **Kiểm định khác biệt chi tiêu** | Kruskal-Wallis $H = 4.8250, p = 0.3057$ (Không có ý nghĩa thống kê) | [`outputs/cluster_profiles_summary.json`](file:///e:/ML/Chinook/outputs/cluster_profiles_summary.json) |

---

## 3. THẺ TÓM TẮT 5 PHÂN KHÚC KHÁCH HÀNG (THE 5 PERSONAS CARD)

| Phân khúc | Quy mô | Chi tiêu | Gu âm nhạc đặc trưng (Chỉ số Lift cao nhất, $\ge 3$ khách) | Gợi ý Gói Sản Phẩm (Bundle) làm Marketing |
| :---: | :---: | :---: | :--- | :--- |
| **Cụm 0** | 12 khách (20.3%) | $41.29 (Cao nhất) | - **Easy Listening (Lift 4.92x, 33.3% khách)**<br>- **Sci-Fi & TV Shows (Lift 4.92x & 1.81x)**<br>- Frank Sinatra (4.92x), Milton Nascimento (4.92x) | **Gói "Retro Cinema & Space Grooves":** Battlestar Galactica OST + Frank Sinatra Best + Vintage Funk/Latin. Giảm 25% khi mua trọn gói. |
| **Cụm 1** | 10 khách (16.9%) | $39.12 | - **Soundtrack (Lift 3.28x, 50% khách)**<br>- **Reggae (Lift 2.95x, 60% khách)**<br>- **Blues (Lift 1.54x, 60% khách)**<br>- Frank Zappa (5.90x), Incognito (5.90x), Clapton | **Gói "Unplugged Roots & Blues Chill":** Eric Clapton Unplugged + Frank Zappa + Axé Bahia / Chill: Brazil. Mua 2 tặng 1 bài độc quyền. |
| **Cụm 2** | 15 khách (25.4% - Đông nhất) | $38.89 | - **Alternative (Lift 2.95x, 20% khách)**<br>- **Electronica/Dance (Lift 2.81x, 33.3% khách)**<br>- **Bossa Nova & World (Lift 2.25x & 2.19x)**<br>- Joe Satriani (3.93x), Jorge Ben (3.93x), Cafezinho | **Gói "Global Beats & Instrumental Virtuoso":** Surfing with the Alien (Joe Satriani) + Cássia Eller + Cafezinho World Bossa. |
| **Cụm 3** | 10 khách (16.9%) | $39.72 | - **Hip Hop/Rap (Lift 2.62x, 40% khách)**<br>- **Reggae (Lift 2.46x, 50% khách)**<br>- **Metal (Lift 1.07x, 100% khách mua)**<br>- Velvet Revolver (5.90x), Van Halen (5.06x) | **Gói "Heavy Rockers & Urban Power":** Contraband (Velvet Revolver) + Van Halen III + Hip Hop Mix. Chương trình $0.69/track hoàn thiện album. |
| **Cụm 4** | 12 khách (20.3%) | $38.45 | - **Classical (Lift 1.76x, 41.7% khách)**<br>- **Blues (Lift 1.50x, 58.3% khách)**<br>- **Alternative & Punk (Lift 1.08x, 91.7% khách)**<br>- Aerosmith (4.92x), The Clash (4.92x), Marillion (4.92x) | **Gói "From Punk to Beethoven: The Boxset":** Let There Be Rock (AC/DC) + The Clash (Punk) + Symphonic Classical Masterpieces. |

---

## 4. DÀN Ý CHI TIẾT ĐỀ XUẤT CHO BÁO CÁO CỦA BẠN (SUGGESTED REPORT OUTLINE)

Bạn có thể cấu trúc báo cáo của mình theo 5 chương chuẩn học thuật / doanh nghiệp:

### Chương 1: Bối cảnh nghiệp vụ và Khám phá dữ liệu (EDA)
- **1.1. Giới thiệu:** Nền tảng Chinook và mục tiêu phân khúc khách hàng.
- **1.2. Thách thức dữ liệu cốt lõi:**
  - Khách hàng nhân tạo đồng nhất: Nêu bảng CV về chi tiêu và bài hát $\to$ Loại bỏ RFM.
  - Ma trận thưa 98.92% và hiện tượng Long-tail (87% track bán 1 lần) $\to$ Curse of Dimensionality.
- **1.3. Cơ cấu doanh thu:** Chi phối bởi Rock, Latin, Metal (63%).
- *Hình minh họa chèn vào:* `figures/erd.png`, `figures/eda_revenue_trends.png`, `figures/eda_customer_behavior_variance.png`, `figures/eda_coverage_longtail.png`.

### Chương 2: Đặt bài toán kỹ thuật & So sánh không gian biểu diễn
- **2.1. Định nghĩa "Sự quan tâm":** Không có streaming logs $\to$ Dựa vào hành vi mua thực tế (nhị phân, số lượng, độ phủ album).
- **2.2. So sánh thực nghiệm 4 không gian (Track vs Album vs Artist vs Genre):**
  - Trực giao tại Track ($d_{\cos} = 1.0$).
  - Thuật toán Curveball Null Model ($N=300$ hoán vị bảo toàn bậc hàng/cột).
  - Phân tích $p$-value: Album ($p=0.0033$) vượt trội so với Track ($p=0.0797$) và Genre ($p=0.0831$).
- **2.3. Lựa chọn:** Không gian Album làm phân cụm chính + Genre/Artist làm không gian diễn giải.
- *Hình minh họa chèn vào:* `figures/problem_selection_cosine_dist.png`, `figures/problem_selection_null_model.png`.

### Chương 3: Phương pháp luận Pipeline & Huấn luyện mô hình
- **3.1. Thiết kế Pipeline khép kín:** $\text{ItemFilter} \to \text{IDF} \to \text{L2 Norm} \to \text{TruncatedSVD} \to \text{L2 Norm} \to \text{Clustering}$.
- **3.2. Luật chọn mô hình đặt trước (`config/decision_rules.yaml`):** Giữ tính liêm chính khoa học, tránh data dredging.
- **3.3. So sánh với Baseline:** Dominant Genre (Sil = -0.013), Country (Sil = -0.118), Random (Sil = -0.090) đều thất bại.
- **3.4. Kết quả duyệt lưới 1.184 cấu hình:** Tuyển chọn mô hình KMeans $k=5$ trên SVD 5D (Silhouette SVD = 0.5941, vượt Null P95 = 0.4441).
- *Hình minh họa chèn vào:* `figures/model_selection_kmeans_elbow.png`, `figures/model_selection_metrics_by_k.png`, `figures/cluster_silhouette_plot.png`.

### Chương 4: Đánh giá độ ổn định & Kiểm định tổng quát hóa
- **4.1. Chiếu trực quan 2D PCA & t-SNE:** 5 cụm tách biệt rõ rệt.
- **4.2. Kiểm tra độ nhạy Seed:** 25 hạt ngẫu nhiên cho Mean ARI = 0.9945 $\to$ Không phụ thuộc khởi tạo.
- **4.3. Kiểm định Hold-out CV (100 lần lặp):** Median ARI = 0.4038 $\to$ Độ ổn định trung bình do cỡ mẫu nhỏ $n=59$.
- **4.4. Kiểm định theo thời gian (Temporal Split):** Khách không mua lại album cũ ($p=0.9640$) nhưng gu thể loại giữ nguyên ($d_{\cos} = 0.5065, p < 0.001$).
- *Hình minh họa chèn vào:* `figures/cluster_pca_projection.png`, `figures/cluster_stability_boxplots.png`, `figures/cluster_temporal_prediction.png`.

### Chương 5: Đề xuất hành động kinh doanh & Khuyến cáo triển khai
- **5.1. Chân dung 5 Persona:** Cụm 0 (Sci-Fi/Soundtrack), Cụm 1 (Acoustic/Reggae), Cụm 2 (Fusion/Dance), Cụm 3 (Metal/Urban), Cụm 4 (Punk/Symphonic).
- **5.2. Thiết kế 5 gói Bundle cụ thể:** Bằng chứng Lift $\ge 3$ khách, cơ chế giá, KPI theo dõi.
- **5.3. Cảnh báo về A/B Testing:** Với $n=59$, công suất thống kê quá thấp $\to$ Chỉ làm Pilot định tính.
- **5.4. Lộ trình dài hạn:** Thu thập streaming logs và triển khai Item-Item Collaborative Filtering theo thể loại.
- *Hình minh họa chèn vào:* `figures/cluster_profiles_lift.png`.

---

## 5. DANH MỤC HÌNH ẢNH SẴN CÓ ĐỂ CHÈN VÀO BÀI (FIGURE CATALOG)

Tất cả 16 hình ảnh đã được lưu trữ sẵn với độ phân giải cao tại thư mục [`figures/`](file:///e:/ML/Chinook/figures/):

1. [`figures/erd.png`](file:///e:/ML/Chinook/figures/erd.png): Sơ đồ thực thể liên kết chuẩn CSDL Chinook.
2. [`figures/eda_revenue_trends.png`](file:///e:/ML/Chinook/figures/eda_revenue_trends.png): Doanh thu theo thời gian, quốc gia, thể loại và nghệ sĩ.
3. [`figures/eda_customer_behavior_variance.png`](file:///e:/ML/Chinook/figures/eda_customer_behavior_variance.png): Biểu đồ phân bổ chi tiêu, số track, số hóa đơn chứng minh $\text{CV} < 8\%$.
4. [`figures/eda_coverage_longtail.png`](file:///e:/ML/Chinook/figures/eda_coverage_longtail.png): So sánh phân phối Long-tail giữa Track (87% bán 1 lần) và Album (87% có $\ge 2$ khách).
5. [`figures/eda_density_heatmaps.png`](file:///e:/ML/Chinook/figures/eda_density_heatmaps.png): Trực quan hóa độ thưa ma trận tương tác.
6. [`figures/eda_genre_artist_cooccurrence.png`](file:///e:/ML/Chinook/figures/eda_genre_artist_cooccurrence.png): Ma trận đồng xuất hiện giữa các thể loại và nghệ sĩ.
7. [`figures/problem_selection_cosine_dist.png`](file:///e:/ML/Chinook/figures/problem_selection_cosine_dist.png): Phân phối khoảng cách Cosine chứng minh hiện tượng trực giao ở Track.
8. [`figures/problem_selection_null_model.png`](file:///e:/ML/Chinook/figures/problem_selection_null_model.png): Kiểm định Null Model so sánh 4 không gian ở Pha B.
9. [`figures/model_selection_kmeans_elbow.png`](file:///e:/ML/Chinook/figures/model_selection_kmeans_elbow.png): Biểu đồ Elbow mở rộng từ $k=2$ đến $k=10$.
10. [`figures/model_selection_gmm_bic.png`](file:///e:/ML/Chinook/figures/model_selection_gmm_bic.png): Biểu đồ tiêu chí thông tin Bayes (BIC) của GMM.
11. [`figures/model_selection_metrics_by_k.png`](file:///e:/ML/Chinook/figures/model_selection_metrics_by_k.png): 4 panel Silhouette, DB, CH và Min Size so với Null Model.
12. [`figures/cluster_silhouette_plot.png`](file:///e:/ML/Chinook/figures/cluster_silhouette_plot.png): Biểu đồ Silhouette từng khách hàng (100% dương).
13. [`figures/cluster_pca_projection.png`](file:///e:/ML/Chinook/figures/cluster_pca_projection.png): Chiếu không gian 2D PCA & t-SNE của 5 cụm và 5 tâm cụm.
14. [`figures/cluster_stability_boxplots.png`](file:///e:/ML/Chinook/figures/cluster_stability_boxplots.png): Hộp phân bổ Hold-out ARI & Silhouette qua 100 lần lặp.
15. [`figures/cluster_temporal_prediction.png`](file:///e:/ML/Chinook/figures/cluster_temporal_prediction.png): Biểu đồ phân phối kiểm định dự báo tương lai và độ tự nhất quán thể loại.
16. [`figures/cluster_profiles_lift.png`](file:///e:/ML/Chinook/figures/cluster_profiles_lift.png): Biểu đồ cột thể hiện Top thể loại Lift theo từng cụm.

---
*Tài liệu bàn giao hoàn tất — Bạn đã có đầy đủ vũ khí, số liệu và hình ảnh để tự tin viết một báo cáo xuất sắc!*
