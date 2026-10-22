# BÁO CÁO TỔNG KẾT DỰ ÁN: PHÂN KHÚC KHÁCH HÀNG CHINOOK THEO THỊ HIẾU ÂM NHẠC
**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album (Chinook Customer Music Taste Clustering)  
**Phiên bản hoàn thiện:** 1.0.0 (Phase D Final Report)  
**Tập dữ liệu:** `data/Chinook_Sqlite.sqlite` (2009 - 2013)  
**Mã cấu hình mô hình tối ưu:** `V2_album_count_svd5_dfFalse_KM_k5`  
**Mô hình lưu trữ:** [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib)  

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)
*(Giới hạn $\le 10$ dòng theo quy định nghiêm ngặt của đề bài)*

1. **KẾT LUẬN ĐỘ TIN CẬY**: Mô hình phân cụm nhận diện được 5 nhóm thị hiếu ngách rõ nét, nhưng **ĐỘ ỔN ĐỊNH CHỈ Ở MỨC VỪA PHẢI (Hold-out ARI = 0.4038)** và **KHÔNG ĐẠT MỨC VỮNG CHẮC ($\ge 0.60$)** theo luật đặt trước [`config/decision_rules.yaml`]; không khuyến nghị đóng băng phân cụm cứng cho toàn bộ hoạt động kinh doanh [`reports/evaluation.md`].
2. **BÀI TOÁN**: Phân khúc 59 khách hàng Chinook dựa trên lịch sử mua 2.240 bài hát nhằm thiết kế gói sản phẩm (album bundles) và cá nhân hóa danh mục [`reports/business_problem.md`].
3. **KẾT QUẢ ML**: Cấu hình tối ưu là KMeans $k=5$ trên đặc trưng số lượng bài hát theo album nén SVD 5 chiều (`V2_album_count_svd5_dfFalse_KM_k5`), đạt Silhouette SVD = **0.5941** (vượt Null Model P95 = **0.4441**, chênh lệch **+0.1500**) [`reports/model_comparison.md`, `outputs/internal_quality_comparison.csv`].
4. **HIỆN TRẠNG KHÁCH HÀNG**: 100% khách hàng đồng nhất tuyệt đối về lượng chi tiêu (trung bình **39.47 USD**, hệ số biến thiên $\text{CV} = 7.38\%$) và số lượng bài hát (trung bình **37.97 bài**, $\text{CV} = 0.69\%$); sự khác biệt duy nhất nằm ở gu âm nhạc [`reports/eda_findings.md`, `figures/eda_customer_behavior_variance.png`].
5. **ĐẶC TRƯNG 5 PHÂN KHÚC**: Cụm 0 (Sci-Fi/Soundtrack, Lift=**4.92x**, $n=12$); Cụm 1 (Reggae/Blues/Pop, Lift=**2.95x**, $n=10$); Cụm 2 (Alternative/Dance/World, Lift=**2.95x**, $n=15$); Cụm 3 (Metal/Urban/Jazz, Lift=**2.62x**, $n=10$); Cụm 4 (Classical/Punk/Blues-Rock, Lift=**1.76x**, $n=12$) [`outputs/cluster_profiles_summary.json`, `figures/cluster_profiles_lift.png`].
6. **KIỂM ĐỊNH THỜI GIAN**: Khách hàng không mua lại cùng một album trong tương lai (độ chính xác dự báo tương lai ở cấp album chỉ đạt **11.86%**, $p=0.9640$), nhưng thị hiếu ở cấp thể loại (Genre) lại cực kỳ nhất quán (Cosine tương đồng = **0.5065**, $p < 0.001$) [`reports/evaluation.md`, `figures/cluster_temporal_prediction.png`].
7. **GIỚI HẠN THỐNG KÊ**: Với quy mô chỉ 59 khách hàng, phương pháp A/B testing tiêu chuẩn là **HOÀN TOÀN KHÔNG KHẢ THI VỀ MẶT CÔNG SUẤT THỐNG KÊ (Statistical Power)** [`reports/cluster_profiles.md`].
8. **KHUYẾN NGHỊ HÀNH ĐỘNG**: Chỉ triển khai thử nghiệm Pilot nhỏ có nhóm đối chứng định tính cho các gói bundle album ngách (10-15 khách/nhóm), đồng thời chuyển dịch sang hệ thống gợi ý Item-Item Collaborative Filtering theo thể loại [`reports/cluster_profiles.md`].

---

## 2. DỮ LIỆU VÀ CHẤT LƯỢNG DỮ LIỆU (DATA & DATA QUALITY)

### 2.1. Số liệu tổng quan CSDL Chinook
Dữ liệu được trích xuất trực tiếp từ file gốc [`data/Chinook_Sqlite.sqlite`](file:///e:/ML/Chinook/data/Chinook_Sqlite.sqlite) giai đoạn giao dịch từ **2009-01-01** đến **2013-12-22** (tròn 5 năm tài chính) [`reports/data_dictionary.md`, `reports/data_quality.md`]:
- **Khách hàng (`Customer`):** 59 khách hàng phân bố trên 24 quốc gia. 100% khách hàng đều phát sinh ít nhất 1 hóa đơn mua hàng (0 khách hàng ảo) [`reports/data_quality.md`: Mục 36].
- **Hóa đơn (`Invoice`):** 412 hóa đơn. Tổng doanh thu hệ thống ghi nhận là **2.328,60 USD** [`reports/data_quality.md`: Mục 32, `reports/eda_findings.md`].
- **Chi tiết hóa đơn (`InvoiceLine`):** 2.240 dòng giao dịch. 100% dòng có `Quantity = 1` (không có giá trị khác 1) với đơn giá chuẩn **0.99 USD** (2.129 dòng) hoặc **1.99 USD** (111 dòng video/chương trình đặc biệt) [`reports/data_quality.md`: Mục 33-34].
- **Sản phẩm (`Track`, `Album`, `Artist`, `Genre`):** Catalog gồm 3.503 bài hát, 347 album, 275 nghệ sĩ và 25 thể loại âm nhạc [`reports/data_dictionary.md`].

### 2.2. Đánh giá chất lượng và Quyết định làm sạch ETL
Dựa trên báo cáo kiểm định 38 chỉ mục chất lượng dữ liệu [`reports/data_quality.md`] và nhật ký làm sạch [`outputs/cleaning_log.csv`]:
1. **Toàn vẹn tham chiếu (Referential Integrity):** Đạt **100% hoàn hảo**. Không có bất kỳ khóa ngoại mồ côi (Orphan FK) nào giữa tất cả 11 bảng quan hệ [`reports/data_quality.md`: Mục 22-31, `tests/test_etl.py`].
2. **Khớp số liệu tài chính:** 412/412 hóa đơn có `Invoice.Total` khớp chính xác tuyệt đối với $\sum (\text{UnitPrice} \times \text{Quantity})$ với sai số $0.0000$ [`reports/data_quality.md`: Mục 32].
3. **Phân biệt Missing Value:**
   - Cột `Company` (49/59 NULL, 83.05%), `Fax` (47/59 NULL, 79.66%), `State` (29/59 NULL, 49.15%) là **NULL hợp lệ về nghiệp vụ** do đối tượng là khách hàng cá nhân quốc tế. Giữ nguyên, không xóa dòng [`reports/data_quality.md`: Mục 1-5].
   - Cột `Composer` (978/3.503 NULL, 27.92%) được điền giá trị chuỗi `'Unknown'` tại tầng làm sạch `dim_track` [`outputs/cleaning_log.csv`: Rule RULE_IMPUTE_TRACK_COMPOSER].
4. **Xử lý 6 cặp bài hát trùng logic `(Name, AlbumId)`:**
   - Quét được đúng 6 cặp (12 bài hát) trùng tên và album (ví dụ `Imagine`, `Branch Closing`, `Banditismo Por Uma Questa`) [`reports/data_quality.md`: Mục 21, bảng chi tiết 2.2].
   - Bản chất: Khác biệt về `MediaTypeId`, thời lượng (`Milliseconds`) và kích thước tệp (`Bytes`) do là các phiên bản định dạng số khác nhau. Không xóa bản ghi gốc trong SQLite; hệ thống ETL gắn cờ `is_duplicate_name_in_album = True` tại bảng `dim_track` để theo dõi [`outputs/cleaning_log.csv`: Rule RULE_FLAG_DUPLICATE_TRACKS].
5. **Xây dựng Data Warehouse phân tích:**
   - Tầng ETL lưu trữ tại [`data/processed/`](file:///e:/ML/Chinook/data/processed/) gồm bảng sự kiện `fact_purchase.parquet` (2.240 dòng hạt nhân Khách hàng $\times$ Track) và các bảng chiều `dim_customer`, `dim_track`, `dim_album`, `dim_artist`, `dim_genre` [`reports/etl_design.md`, `reports/phase_ETL_summary.md`].
   - Thuộc tính nhân khẩu học (Country, City, SupportRepId) được cách ly hoàn toàn ở bảng chiều phục vụ diễn giải kinh doanh sau gom cụm, triệt tiêu rủi ro rò rỉ dữ liệu (Data Leakage) vào thuật toán [`tests/test_data_validation.py`].

---

## 3. HIỆN TRẠNG KINH DOANH TỪ EDA (BUSINESS CONTEXT & EDA)

Toàn bộ các phát hiện định lượng dưới đây được trích xuất từ tài liệu khám phá dữ liệu [`reports/eda_findings.md`] và hệ thống biểu đồ trực quan hóa:

### 3.1. Phân bổ Doanh thu, Quốc gia và Thể loại
- **Cơ cấu dòng nhạc thống trị:** Doanh thu toàn cửa hàng bị chi phối áp đảo bởi 3 thể loại: **Rock** đạt **826,65 USD** (35,50% tổng doanh thu), **Latin** đạt **382,14 USD** (16,41%) và **Metal** đạt **261,36 USD** (11,22%). Ba thể loại này chiếm tới **63,13%** tổng doanh thu toàn hệ thống [`reports/eda_findings.md`: Phát hiện 7, `figures/eda_revenue_trends.png`: Hình 1C].
- **Nghệ sĩ bán chạy nhất:** Iron Maiden (138,60 USD, 140 bài), U2 (105,93 USD, 107 bài), Metallica (90,09 USD, 91 bài) [`figures/eda_revenue_trends.png`: Hình 1D].
- **Thị trường địa lý:** USA dẫn đầu doanh thu với **523,06 USD** (13 khách hàng), theo sau là Canada với **303,96 USD** (8 khách hàng), Brazil và France đều có 5 khách hàng [`figures/eda_revenue_trends.png`: Hình 1B].

### 3.2. Tính đồng đều nhân tạo giữa các khách hàng
- Số lượng bài hát mỗi khách hàng đã mua: Trung bình = **37,97 bài**, Độ lệch chuẩn = **0,26**, Min = **36**, Max = **38** bài. **Hệ số biến thiên $\text{CV} = 0,69\%$** [`reports/eda_findings.md`: Phát hiện 1, `figures/eda_customer_behavior_variance.png`: Hình 2A].
- Số lượng hóa đơn mỗi khách hàng: Trung bình = **6,98 hóa đơn**, Độ lệch chuẩn = **0,13**, Min = **6**, Max = **7**. **Hệ số biến thiên $\text{CV} = 1,86\%$** [`figures/eda_customer_behavior_variance.png`: Hình 2B].
- Tổng số tiền chi trả mỗi khách hàng: Trung bình = **39,47 USD**, Độ lệch chuẩn = **2,91 USD**, Min = **36,64 USD**, Max = **49,62 USD**. **Hệ số biến thiên $\text{CV} = 7,38\%$** [`figures/eda_customer_behavior_variance.png`: Hình 2C].
- **Hệ quả kinh doanh cốt lõi:** Khách hàng Chinook hoàn toàn đồng nhất về quy mô số lượng và số tiền. Các mô hình truyền thống như RFM (Recency, Frequency, Monetary) hoàn toàn mất tác dụng. Điểm khác biệt duy nhất của khách hàng là **Gu âm nhạc / Sở thích nội dung (Content Taste)** [`reports/eda_findings.md`: Phát hiện 2].

### 3.3. Độ thưa và Hiện tượng Long-tail cực đoan
- **Ma trận Khách hàng $\times$ Track ($59 \times 3.503$):** Độ thưa lên tới **98,916%** (Độ đậm đặc chỉ **1,084%** với đúng 2.240 ô có giao dịch) [`reports/eda_findings.md`: Phát hiện 3, `figures/eda_density_heatmaps.png`: Hình 4A].
- **Hiện tượng Long-tail phân mảnh ở cấp độ Track:** Trong 1.984 bài hát đã bán, có tới **1.728 bài (87,10%) chỉ có đúng 1 khách hàng mua**. Chỉ có 256 bài (12,90%) có 2 khách hàng mua. **Không có bất kỳ bài hát nào trong catalog có từ 3 khách hàng cùng mua trở lên** [`reports/eda_findings.md`: Phát hiện 4, `figures/eda_coverage_longtail.png`: Hình 3A].
- **Khách hàng mua bài lẻ (Cherry-picking):** Tỷ lệ mua trọn vẹn 100% một album chỉ chiếm **3,77%** (49 / 1.301 cặp tương tác Khách hàng - Album). Trung bình mỗi khách hàng mua rải rác ở **22,05 album khác nhau** với độ phủ trung bình chỉ đạt **16,79%** số bài của mỗi album [`reports/eda_findings.md`: Phát hiện 6].

---

## 4. BÀI TOÁN KINH DOANH VÀ BÀI TOÁN MACHINE LEARNING (PROBLEM FORMULATION)

### 4.1. Bài toán kinh doanh (Business Objective)
Từ hiện trạng khách hàng mua bài lẻ và danh mục bị chiếm lĩnh bởi Rock, bài toán kinh doanh được xác định là: **Phân khúc thị hiếu khách hàng để thiết kế các gói sản phẩm đóng gói (Album Bundles: "Complete My Album" / "Cross-Artist Bundles") và cá nhân hóa trang chủ, nhằm tăng giá trị giỏ hàng và thúc đẩy tiêu thụ danh mục ngách** [`reports/business_problem.md`: Mục 2].

### 4.2. So sánh thực nghiệm 4 không gian biểu diễn (Pha B)
Trước khi chọn bài toán kỹ thuật, một kiểm định thực nghiệm định lượng giữa 4 không gian biểu diễn đã được thực thi với **$N = 300$ lần hoán vị Null Model (Thuật toán Curveball bảo toàn bậc hàng và cột)** [`reports/problem_selection.md`, `outputs/problem_selection_metrics.json`]:

| Tiêu chuẩn đo lường định lượng | (a) Khách $\times$ Track | (b) Khách $\times$ Album (ĐƯỢC CHỌN) | (c1) Khách $\times$ Artist | (c2) Khách $\times$ Genre | Nguồn dữ liệu kiểm chứng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Kích thước ma trận (Số chiều mục đã bán)** | $59 \times 1.984$ | **$59 \times 304$** | $59 \times 165$ | $59 \times 24$ | [`reports/problem_selection.md`: Bảng 1] |
| **Độ thưa trên tập bán (Sparsity)** | 98,09% | **92,75%** | 90,52% | 68,93% | [`outputs/problem_selection_metrics.json`] |
| **Trung vị khoảng cách Cosine ($d_{\cos}$)** | **1,0000** | **0,9544** | 0,8655 | 0,3828 | [`figures/problem_selection_cosine_dist.png`] |
| **Tỷ lệ cặp khách hàng trực giao ($d_{\cos} \ge 0,95$)**| **97,49%** (1.668 cặp) | **60,84%** (1.041 cặp) | 8,88% (152 cặp) | 0,00% (0 cặp) | [`reports/problem_selection.md`: Mục 2.1] |
| **Hopkins Statistic (Không gian SVD)** | $0,6035 \pm 0,035$ | **$0,6435 \pm 0,036$** | $0,6513 \pm 0,032$ | $0,5971 \pm 0,040$ | [`outputs/problem_selection_metrics.json`] |
| **Silhouette thật (KMeans tốt nhất $k \in [2,6]$)** | 0,0027 ($k=3$) | **0,0679 ($k=5$)** | 0,0845 ($k=6$) | 0,1790 ($k=2$) | [`reports/problem_selection.md`: Bảng 1] |
| **Silhouette Null Model (Mean $\pm$ Std)** | $0,0020 \pm 0,0005$ | **$0,0126 \pm 0,0022$** | $0,0327 \pm 0,0059$ | $0,1565 \pm 0,0145$ | [`figures/problem_selection_null_model.png`] |
| **Empirical $p$-value so với Null Model** | **$p = 0,0797$ (FAIL)** | **$p = 0,0033$ ($p < 0,005$)** | $p = 0,0033$ | **$p = 0,0831$ (FAIL)** | [`outputs/problem_selection_metrics.json`] |

### 4.3. Lý do lựa chọn bài toán kỹ thuật
1. **Loại bỏ không gian Track:** 97,49% số cặp khách hàng trực giao hoàn toàn ($d_{\cos} = 1.0$). $p$-value so với Null Model đạt $0,0797 > 0,05$, chứng minh thuật toán phân cụm trên ma trận Track chỉ gom cụm trên nhiễu ngẫu nhiên [`reports/problem_selection.md`: Mục 3.3].
2. **Loại bỏ không gian Genre thuần túy:** Mặc dù Silhouette đạt 0,1790, Null Model ngẫu nhiên cũng tạo ra Silhouette lên tới $0,1565$ ($p = 0,0831 > 0,05$). Do Rock/Latin có mặt ở mọi khách hàng, việc gom cụm trên 24 thể loại bị chia cắt giả tạo, không tạo ra hành động cụ thể cho sản phẩm [`reports/problem_selection.md`: Mục 2.2].
3. **Quyết định chính thức:** Lựa chọn **Không gian Khách hàng $\times$ Album** làm ma trận phân cụm cốt lõi ($p = 0,0033 \ll 0,05$), kết hợp với không gian **Artist và Genre làm tầng giải thích và đặt tên Persona (Profiling Space)** [`reports/problem_selection.md`: Mục 3.1].

---

## 5. PHƯƠNG PHÁP NGHIÊN CỨU & KHUNG QUYẾT ĐỊNH (METHODOLOGY & PRE-REGISTERED RULES)

### 5.1. Luật chọn mô hình đặt trước (Pre-registered Decision Rules)
Để ngăn ngừa thiên lệch lựa chọn sau khi xem kết quả (HARK-ing / data dredging), bộ quy tắc quyết định đã được thiết lập và đóng băng trước tại [`config/decision_rules.yaml`]:
1. **Dải số cụm kiểm tra:** $k \in [2, 6]$ (với cỡ mẫu $n=59$, $k > 6$ sẽ tạo ra các cụm vi mô $< 5$ khách hàng, không khả thi về mặt kinh doanh).
2. **Ngưỡng kích thước cụm tối thiểu:** Loại bỏ mọi cấu hình có $\text{min\_cluster\_size} < 5$ khách hàng.
3. **Ngưỡng kiểm định Null Model:** Chỉ xem xét các cấu hình có Silhouette trên không gian SVD **vượt phân vị 95 ($\text{P95}$) của phân phối Null Model** tương ứng.
4. **Tiêu chuẩn ổn định Hold-out:**
   - $\text{ARI} \ge 0,60$: Cụm ổn định vững chắc (Robust clusters).
   - $0,40 \le \text{ARI} < 0,60$: Cụm có độ ổn định vừa phải (Moderate stability).
   - $\text{ARI} < 0,40$: Cụm không ổn định (Unstable) $\to$ Bắt buộc đưa ra phương án thay thế.

### 5.2. Pipeline đặc trưng (Feature Engineering Pipeline)
Sáu biến thể đặc trưng được định nghĩa tại [`config/features.yaml`] và kiểm thử tại [`tests/test_pipeline.py`]:
- **Biến thể V1:** `album_binary` (0/1).
- **Biến thể V2:** `album_count` (Số bài hát đã mua trong từng album).
- **Biến thể V3:** `album_coverage` (Tỷ lệ bài hát đã mua / tổng số bài trong album).
- **Biến thể V4:** `genre_share` (Tỷ trọng chi tiêu theo 25 thể loại).
- **Biến thể V5:** `artist_binary` (0/1 theo nghệ sĩ).
- **Biến thể V6:** `composite_album_genre` (Ghép nối đặc trưng Album và Genre).

Pipeline Scikit-Learn tuần tự khép kín không rò rỉ thông tin [`src/pipeline.py`]:
$$\mathbf{X}_{\text{raw}} \xrightarrow{\text{ItemFilter}(\text{min\_cust}=2)} \mathbf{X}_{\text{filtered}} \xrightarrow{\text{IDFTransformer}(\text{smooth})} \mathbf{X}_{\text{idf}} \xrightarrow{\text{Normalizer}(L_2)} \mathbf{X}_{\text{norm}} \xrightarrow{\text{TruncatedSVD}(d)} \mathbf{X}_{\text{svd}} \xrightarrow{\text{Normalizer}(L_2)} \mathbf{X}_{\text{final}}$$

---

## 6. KẾT QUẢ THỰC NGHIỆM VÀ ĐÁNH GIÁ (EMPIRICAL RESULTS & EVALUATION)

### 6.1. So sánh với các mô hình Baseline
So sánh giữa cấu hình tối ưu `V2_album_count_svd5_dfFalse_KM_k5` và các baseline định lượng chuẩn [`outputs/baseline_comparison.csv`]:

| Mô hình / Cấu hình đánh giá | Loại hình Baseline | Số cụm $k$ | Silhouette (SVD) | Silhouette (Raw L2) | Davies-Bouldin | Calinski-Harabasz | Kích thước cụm | Nguồn dữ liệu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mô hình đề xuất (V2 KMeans k=5)**| **Mô hình tối ưu** | **5** | **0,5941** | **0,0665** | **0,8234** | **32,54** | **[15, 12, 12, 10, 10]**| [`outputs/internal_quality_comparison.csv`] |
| **Dominant Genre Heuristic** | Business Rule | 3 | N/A | -0,0130 | 2,7992 | 2,42 | [47, 7, 5] (Lệch cực đoan) | [`outputs/baseline_comparison.csv`: Dòng 2] |
| **Country Grouping Heuristic** | Business Rule | 5 | N/A | -0,1183 | 5,2858 | 1,04 | [28, 13, 8, 5, 5] | [`outputs/baseline_comparison.csv`: Dòng 3] |
| **Random Permutation ($k=5$)** | Statistical Random | 5 | N/A | -0,0902 | 6,3390 | 1,00 | [11, 12] (Đều ngẫu nhiên) | [`outputs/baseline_comparison.csv`: Dòng 7] |
| **Raw Binary KMeans ($k=5$)** | Raw Feature ML | 5 | N/A | 0,0768 | 2,9776 | 2,47 | [33, 11, 6, 5, 4] | [`outputs/baseline_comparison.csv`: Dòng 12] |

*Nhận xét:* Các baseline theo quy tắc nghiệp vụ đơn giản (gán theo thể loại nhiều nhất hoặc theo quốc gia) đều có Silhouette âm ($-0,013$ và $-0,118$), Davies-Bouldin rất cao ($> 2,79$), chứng minh các quy tắc phân cụm trực giác hoàn toàn thất bại trong việc nhóm các khách hàng đồng điệu.

### 6.2. So sánh Top 5 ứng viên mô hình (Chỉ số nội tại & Null Model)
Từ 1.184 cấu hình tại Pha C2 [`outputs/model_comparison.csv`, `reports/c2_audit.md`], 5 ứng viên xuất sắc nhất đã được đánh giá đối đầu [`outputs/internal_quality_comparison.csv`]:

| Mã cấu hình ứng viên | Thuật toán & Biến thể | $k$ | Silhouette SVD | Null P95 | Khoảng cách vượt Null | Davies-Bouldin | Calinski-Harabasz | Cân đối cụm |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CANDIDATE 1 (CHỌN)** | **KMeans V2 Count SVD 5D** | **5** | **0,5941** | **0,4441** | **+0,1500** | **0,8234** | **32,54** | **0,6667** |
| CANDIDATE 2 | KMeans V2 Count SVD 5D | 6 | 0,5976 | 0,4411 | +0,1565 | 0,7831 | 34,57 | 0,5833 |
| CANDIDATE 3 | Agglomerative Complete SVD 5D | 5 | 0,5728 | 0,4010 | +0,1718 | 0,8835 | 31,03 | 0,8462 |
| CANDIDATE 4 | Spectral NN=8 SVD 5D | 5 | 0,5902 | 0,4347 | +0,1555 | 0,8100 | 32,18 | 0,7143 |
| CANDIDATE 5 | KMeans V1 Binary SVD 5D | 5 | 0,5608 | 0,4441 | +0,1167 | 0,8643 | 30,23 | 0,7857 |

- **Silhouette từng khách hàng:** 100% khách hàng có hệ số Silhouette dương ($s(i) > 0$), không có khách hàng nào bị gán sai cụm ($s(i) < 0$). Giá trị trung vị $s(i) = 0,6078$, giá trị tối thiểu là $+0,2073$ [`reports/evaluation.md`: Bảng 2.1, `figures/cluster_silhouette_plot.png`].
- **Trực quan hóa 2D PCA & t-SNE:** Biểu đồ chiếu không gian xác nhận 5 cụm tách biệt rõ ràng trên mặt phẳng tọa độ chính, các tâm cụm phân bổ đều khắp không gian đa tạp thị hiếu [`figures/cluster_pca_projection.png`].

### 6.3. Đánh giá tính ổn định và Khả năng tổng quát hóa (Pha C3)
Dựa trên kết quả thực nghiệm 100 lượt Cross-Validation Hold-out (80% train / 20% test) và 25 lượt kiểm tra Random Seed [`outputs/stability_summary.csv`, `reports/evaluation.md`]:
1. **Độ ổn định Random Seed (25 seeds độc lập):**
   - Mean Pairwise $\text{ARI} = \mathbf{0,9945}$, Median $\text{ARI} = \mathbf{1,0000}$, Min $\text{ARI} = \mathbf{0,9641}$ [`reports/evaluation.md`: Mục 3.3]. Thuật toán KMeans với $n\_init = 30$ hội tụ gần như tuyệt đối, không phụ thuộc vào điểm khởi tạo ngẫu nhiên.
2. **Độ ổn định Hold-out CV (80/20 train/test qua 100 lần lặp):**
   - **Candidate 1 (KMeans $k=5$):** Median Hold-out $\text{ARI} = \mathbf{0,4038}$ (P5 = 0,0782; P95 = 0,7966; Median Silhouette = 0,3760) [`outputs/stability_summary.csv`: Dòng 2, `figures/cluster_stability_boxplots.png`].
   - **So sánh với các ứng viên khác:** Candidate 2 ($k=6$, ARI = 0,2928); Candidate 3 (Agglomerative, ARI = 0,3210); Candidate 4 (Spectral, ARI = 0,3131); Candidate 5 (V1 Binary, ARI = 0,2632). Candidate 1 đạt độ ổn định mẫu con cao nhất trong toàn bộ các ứng viên.
   - **Đánh giá khách quan theo luật đặt trước:** Giá trị $\text{ARI} = 0,4038$ nằm trong khoảng $[0,40 - 0,60]$ - tức **Độ ổn định ở mức vừa phải (Moderate Stability)**, không đạt ngưỡng vững chắc $\ge 0,60$ [`config/decision_rules.yaml`]. Điều này giải thích tại sao không được áp dụng cụm này làm phân khúc cứng cho toàn bộ vận hành.
3. **Độ ổn định theo số cụm ($k$-sensitivity):**
   - So sánh phân hoạch $k=5$ với $k=4$: $\text{ARI} = \mathbf{0,7300}$, $\text{NMI} = \mathbf{0,7933}$ [`reports/evaluation.md`: Mục 3.4].
   - So sánh phân hoạch $k=5$ với $k=6$: $\text{ARI} = \mathbf{0,8522}$, $\text{NMI} = \mathbf{0,9158}$. Khi tăng/giảm $k$, các cụm cốt lõi vẫn giữ nguyên cấu trúc và chỉ phân rã hoặc gộp cụm ở biên.
4. **Kiểm định theo dòng thời gian (Temporal Future Validation):**
   - Chia dữ liệu theo thời gian: Giai đoạn 1 (3 hóa đơn đầu, 1.001 dòng) vs Giai đoạn 2 (các hóa đơn sau, 1.239 dòng) [`reports/evaluation.md`: Mục 4.1].
   - Đo độ chính xác gán tâm cụm tương lai ở cấp độ Album: Đạt **11,86%** (7/59 khách hàng), thấp hơn phân phối hoán vị ngẫu nhiên (Null Mean = **19,75%**, Null P95 = **28,81%**, $p = 0,9640$) [`figures/cluster_temporal_prediction.png`].
   - **Phát hiện bản chất ngành âm nhạc số:** Khách hàng **không bao giờ mua lại cùng một album đã sở hữu**. Sau 3 hóa đơn đầu, giỏ hàng của họ chuyển sang các album mới hoàn toàn (độ giao thoa Jaccard ở cấp album giữa 2 giai đoạn xấp xỉ bằng 0).
   - **Tuy nhiên, ở cấp độ Thể loại (Genre Level):** Độ tương đồng Cosine sở hữu thể loại giữa Giai đoạn 1 và Giai đoạn 2 đạt **Trung vị = 0,5065** (Mean = 0,4901), vượt trội tuyệt đối so với mức ngẫu nhiên 0,0500 ($p < 0,001$) [`reports/evaluation.md`: Mục 4.3]. Khách hàng có thể đổi album nhưng **gu thể loại âm nhạc được duy trì bền vững theo thời gian**.

### 6.4. Hồ sơ chi tiết 5 phân khúc khách hàng (Cluster Personas & Lift)
Toàn bộ chỉ số Lift dưới đây được lọc với ngưỡng tối thiểu $\ge 3$ khách hàng trong cụm cùng sở hữu để triệt tiêu nhiễu đơn lẻ [`reports/cluster_profiles.md`, `outputs/cluster_profiles_summary.json`, `figures/cluster_profiles_lift.png`]:

```
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 0: ECLECTIC SOUNDTRACK, SCI-FI & VINTAGE GROOVES (n=12, 20.3% khách hàng)                                      |
| - Chi tiêu TB: $41.29 (Cao nhất) | Track TB: 38.0 | Quốc gia chính: USA (3), Canada (2), Czech Republic (1)       |
| - Top Thể loại Lift: Easy Listening (4.92x, 33.3%), Science Fiction (4.92x, 33.3%), Soundtrack (2.19x, 33.3%),    |
|                      TV Shows (1.81x, 58.3%), Rock And Roll (3.69x, 25.0%)                                         |
| - Top Nghệ sĩ Lift:  Milton Nascimento (4.92x, 41.7%), Frank Sinatra (4.92x, 33.3%), Motörhead (3.93x, 33.3%)     |
| - Top Album Lift:    The Colour And The Shape (4.92x), My Way: Best Of Frank Sinatra (4.92x), Roda De Funk (4.92x)  |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 1: SOULFUL BLUES, REGGAE & ACOUSTIC CONNOISSEURS (n=10, 16.9% khách hàng)                                     |
| - Chi tiêu TB: $39.12 | Track TB: 38.0 | Quốc gia chính: Brazil (2), Germany (1), Austria (1)                      |
| - Top Thể loại Lift: Soundtrack (3.28x, 50.0%), Reggae (2.95x, 60.0%), Pop (2.27x, 50.0%), Blues (1.54x, 60.0%)    |
| - Top Nghệ sĩ Lift:  Frank Zappa (5.90x, 40.0%), Incognito (5.90x, 40.0%), Page & Plant (5.90x, 40.0%)             |
| - Top Album Lift:    Axé Bahia 2001 (5.90x), Bongo Fury (5.90x), Chill: Brazil (5.90x), BBC Sessions Live (5.90x)   |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 2: GLOBAL FUSION, ALTERNATIVE & DANCE EXPLORERS (n=15, 25.4% khách hàng)                                       |
| - Chi tiêu TB: $38.89 | Track TB: 37.9 | Quốc gia chính: USA (3), Canada (3), France (2)                           |
| - Top Thể loại Lift: Alternative (2.95x, 20.0%), Electronica/Dance (2.81x, 33.3%), Bossa Nova (2.25x, 26.7%),       |
|                      World (2.19x, 33.3%), Hip Hop/Rap (1.75x, 26.7%)                                             |
| - Top Nghệ sĩ Lift:  Joe Satriani (3.93x, 26.7%), Jorge Ben (3.93x, 26.7%), Jota Quest (3.93x, 26.7%)              |
| - Top Album Lift:    Surfing with the Alien (3.93x), Cássia Eller Sem Limite (3.93x), Cafezinho (3.93x)            |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 3: HARD ROCK, HEAVY METAL & URBAN RHYTHM LOVERS (n=10, 16.9% khách hàng)                                       |
| - Chi tiêu TB: $39.72 | Track TB: 38.0 | Quốc gia chính: USA (2), Czech Republic (1), Brazil (1)                    |
| - Top Thể loại Lift: Hip Hop/Rap (2.62x, 40.0%), Reggae (2.46x, 50.0%), Drama (1.26x, 30.0%), Metal (1.07x, 100.0%)|
| - Top Nghệ sĩ Lift:  David Coverdale (5.90x, 40.0%), Velvet Revolver (5.90x, 50.0%), Van Halen (5.06x, 60.0%)      |
| - Top Album Lift:    Contraband (5.90x, 50.0%), Van Halen III (5.90x), Into The Light (5.90x)                       |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 4: CLASSIC BLUES-ROCK, PUNK & SYMPHONIC FANS (n=12, 20.3% khách hàng)                                          |
| - Chi tiêu TB: $38.45 | Track TB: 38.0 | Quốc gia chính: USA (4), United Kingdom (2), Brazil (1)                   |
| - Top Thể loại Lift: Classical (1.76x, 41.7%), Blues (1.50x, 58.3%), R&B/Soul (1.45x, 41.7%),                      |
|                      Alternative & Punk (1.08x, 91.7%), Metal (1.07x, 100.0%)                                      |
| - Top Nghệ sĩ Lift:  Aerosmith (4.92x, 33.3%), Marillion (4.92x, 33.3%), The Clash (4.92x, 41.7%)                  |
| - Top Album Lift:    Let There Be Rock (4.92x), Big Ones (4.92x), Misplaced Childhood (4.92x)                       |
+--------------------------------------------------------------------------------------------------------------------+
```

### 6.5. Kiểm định thống kê phi tham số về nhân khẩu học & hành vi
Kết quả kiểm định nghiêm ngặt xác nhận sự độc lập giữa các cụm và các yếu tố phi sở thích [`outputs/cluster_profiles_summary.json`: Dòng 476-546]:
1. **Tổng chi tiêu (TotalSpent):** Kiểm định Kruskal-Wallis $H = 4,8250, p = 0,3057 > 0,05$ ($\eta^2 = 0,0153$ - hiệu ứng rất nhỏ). Chi tiêu không có sự khác biệt có ý nghĩa thống kê giữa 5 cụm.
2. **Tổng số bài đã mua (TotalTracksPurchased):** $H = 2,9333, p = 0,5690 > 0,05$ ($\eta^2 = 0,0$).
3. **Số lượng hóa đơn (InvoiceCount):** $H = 2,9333, p = 0,5690 > 0,05$ ($\eta^2 = 0,0$).
4. **Phân bổ địa lý theo khu vực (Region):** Kiểm định Chi bình phương độc lập $\chi^2 = 7,0320, p = 0,9726 > 0,05$ ($df=16$, Cramér's $V = 0,1726$). Phân bổ quốc gia hoàn toàn đồng đều giữa các cụm.

---

## 7. ĐỀ XUẤT HÀNH ĐỘNG KINH DOANH THEO CỤM (ACTIONABLE BUSINESS RECOMMENDATIONS)

### 7.1. Cảnh báo trung thực về năng lực kiểm định (Statistical Power Disclaimer)
> [!WARNING]
> **A/B TESTING TIÊU CHUẨN LÀ HOÀN TOÀN KHÔNG KHẢ THI (NOT STATISTICALLY VIABLE):**  
> Tổng quy mô khách hàng toàn hệ thống hiện tại chỉ là **$n = 59$ khách hàng** (mỗi cụm chỉ có từ 10 đến 15 khách).  
> Theo công thức tính cỡ mẫu kiểm định giả thuyết thống kê hai phía ($\alpha = 0,05, \text{Power} = 0,80$) để phát hiện mức tăng chuyển đổi (Lift) thực tế $15\% - 20\%$, cần tối thiểu **vài trăm đến vài nghìn khách hàng trên mỗi nhánh thử nghiệm**.  
> Việc chia đôi 59 khách hàng thành nhóm A ($n \approx 30$) và nhóm B ($n \approx 29$) sẽ dẫn đến sai lầm loại II nghiêm trọng (False Negative do biên độ sai số cực lớn $\pm 18\%$).  
> **Giải pháp đề xuất:** Thay vì A/B testing quy mô lớn, doanh nghiệp bắt buộc phải áp dụng **Thử nghiệm Pilot có kiểm soát định tính (Qualitative Micro-Pilot)** trên nhóm nhỏ tình nguyện viên hoặc đo lường chỉ số phản hồi đơn vị trước/sau can thiệp (Pre/Post Intervention Tracking) [`reports/cluster_profiles.md`: Mục 3].

### 7.2. Chiến dịch chi tiết và Thiết kế Bundle theo từng cụm

```
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 0: CHIẾN DỊCH "VINTAGE CINEMA & GROOVY RETRO BUNDLE"                                                           |
| - Bằng chứng dữ liệu: Lift Easy Listening = 4.92x, Sci-Fi = 4.92x, TV Shows = 1.81x (58.3% khách mua).            |
| - Thiết kế gói Bundle: Gói 3 Album "Retro Cinema & Space Grooves" gồm:                                              |
|   1. Battlestar Galactica Season 3 Soundtrack (Sci-Fi TV)                                                          |
|   2. My Way: The Best Of Frank Sinatra (Easy Listening / Traditional Pop)                                         |
|   3. Roda De Funk / Milton Nascimento Collection (Latin Vintage Groove)                                            |
| - Cơ chế giá: Giảm 25% khi mua trọn gói bundle ($17.99 thay vì $23.76 mua lẻ).                                     |
| - Kênh thử nghiệm Pilot: Gửi email cá nhân hóa trực tiếp tới 12 khách hàng thuộc Cụm 0.                            |
| - KPI theo dõi: Tỷ lệ mở email (Mục tiêu $\ge 40\%$), Tỷ lệ Click-to-Purchase (Mục tiêu $\ge 15\%$, tức $\ge 2$ đơn)|
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 1: CHIẾN DỊCH "ACOUSTIC SESSIONS & ROOT SOUNDS"                                                                |
| - Bằng chứng dữ liệu: Lift Soundtrack = 3.28x, Reggae = 2.95x (60% khách), Blues = 1.54x (60% khách).              |
| - Thiết kế gói Bundle: Gói "Unplugged Roots & Blues Chill" gồm:                                                    |
|   1. Eric Clapton Unplugged / BBC Sessions Live                                                                    |
|   2. Bongo Fury (Frank Zappa & Captain Beefheart)                                                                  |
|   3. Axé Bahia 2001 / Chill: Brazil Disc 1 (Acoustic World)                                                        |
| - Cơ chế giá: Mua 2 tặng 1 bài hát độc quyền hoặc giảm 20% khi hoàn thiện album chưa đủ bài.                       |
| - Kênh thử nghiệm: Pop-up gợi ý cá nhân hóa khi khách hàng Cụm 1 đăng nhập vào hệ thống.                           |
| - KPI theo dõi: Tỷ lệ thêm vào giỏ hàng (Add-to-cart rate $\ge 20\%$).                                              |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 2: CHIẾN DỊCH "WORLD FUSION & INDIE GROOVES"                                                                   |
| - Bằng chứng dữ liệu: Lift Alternative = 2.95x, Electronica/Dance = 2.81x, Bossa Nova = 2.25x, World = 2.19x.      |
| - Thiết kế gói Bundle: Gói "Global Beats & Instrumental Virtuoso" gồm:                                             |
|   1. Surfing with the Alien (Joe Satriani - Instrumental Rock/Fusion)                                              |
|   2. Cássia Eller Coleção Sem Limite (Brazilian Alternative)                                                       |
|   3. Jorge Ben Jor 25 Anos / Cafezinho (World Bossa)                                                               |
| - Cơ chế: Gợi ý dạng danh sách phát tổng hợp theo chủ đề "Indie World Travelers".                                  |
| - KPI theo dõi: Số lượng bài hát mua ngoài danh mục Rock truyền thống (Non-Rock track uplift $\ge 25\%$).          |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 3: CHIẾN DỊCH "HEAVY RIFFS & STREET BEATS"                                                                     |
| - Bằng chứng dữ liệu: Lift Hip Hop/Rap = 2.62x, Reggae = 2.46x, Metal = 1.07x (100% khách Cụm 3 mua Metal).       |
| - Thiết kế gói Bundle: Gói "Heavy Rockers & Urban Power" gồm:                                                      |
|   1. Contraband (Velvet Revolver)                                                                                  |
|   2. Van Halen III / The Best Of Van Halen                                                                         |
|   3. Top Urban Beats Mix (Hip Hop / Cross-genre Rap Rock)                                                          |
| - Cơ chế giá: Chương trình "Complete My Metal Collection" - ưu đãi $0.69/bài cho các track còn thiếu trong album. |
| - KPI theo dõi: Tỷ lệ hoàn thiện album (Album Completion Rate tăng từ 3.77% lên 10%).                              |
+--------------------------------------------------------------------------------------------------------------------+
| CỤM 4: CHIẾN DỊCH "TIMELESS BLUES, PUNK & SYMPHONY"                                                                |
| - Bằng chứng dữ liệu: Lift Classical = 1.76x, Blues = 1.50x, Alternative & Punk = 1.08x (91.7% khách mua).        |
| - Thiết kế gói Bundle: Gói "From Punk to Beethoven: The Connoisseur Boxset" gồm:                                   |
|   1. Let There Be Rock (AC/DC) & Big Ones (Aerosmith)                                                              |
|   2. The Clash Anthology (Punk Rock)                                                                               |
|   3. Classic Masterpieces: Symphonic Selections (Classical)                                                        |
| - Cơ chế: Thư viện sưu tập cao cấp dành cho người sành nhạc cổ điển và rock nguyên bản.                            |
| - KPI theo dõi: Doanh thu trung bình trên mỗi khách hàng thử nghiệm tăng $\ge 10\%$ so với mức cơ sở $38.45.       |
+--------------------------------------------------------------------------------------------------------------------+
```

---

## 8. HẠN CHẾ VÀ BƯỚC ĐI TIẾP THEO (LIMITATIONS & NEXT STEPS)

### 8.1. Các hạn chế khoa học cốt lõi
1. **Quy mô mẫu cực nhỏ ($n = 59$):** Cỡ mẫu 59 điểm dữ liệu là giới hạn lớn nhất của dự án. Mặc dù các kiểm định hoán vị Null Model chứng minh cấu trúc cụm vượt trội so với ngẫu nhiên ($p < 0,005$), độ biến thiên mẫu con (Hold-out ARI đạt 0,4038) cho thấy các ranh giới phân tách cụm vẫn chịu ảnh hưởng lớn khi tập dữ liệu bị thay đổi [`reports/evaluation.md`: Mục 5].
2. **Khái niệm "Sự quan tâm" chỉ đo bằng hành vi mua:** Hệ thống không có logs lượt nghe (streams), thời gian nghe, tỷ lệ bỏ qua bài (skip rate), hay lượt thêm vào danh sách yêu thích. Khách hàng có thể mua một album làm quà tặng hoặc mua thử nhưng không thực sự yêu thích [`reports/eda_findings.md`: Định nghĩa cốt lõi].
3. **Dữ liệu có tính chất tổng hợp nhân tạo (Synthetic Nature):** Tính đồng nhất kỳ lạ về số lượng track (~38 bài) và số lượng hóa đơn (~7 đơn) phản ánh dữ liệu Chinook được tạo ra theo một khuôn mẫu mô phỏng, làm triệt tiêu các đặc trưng hành vi thực tế như khách hàng trung thành, khách hàng rời bỏ, hay độ nhạy cảm về giá [`reports/eda_findings.md`: Phát hiện 1].
4. **Hiện tượng "Rock Dominance":** Thể loại Rock chiếm hơn 70% số bài hát và 35,5% doanh thu catalog. Cả 5 cụm đều có 90% - 100% khách hàng mua Rock. Do đó, các cụm không phân hóa theo kiểu "Cụm nghe Rock vs Cụm không nghe Rock", mà phân hóa ở **sở thích thứ cấp đi kèm với Rock** [`reports/cluster_profiles.md`: Mục 2.3].
5. **Không gian thử nghiệm đã duyệt:** Dự án đã khảo nghiệm 16 pipeline đặc trưng và 1.184 cấu hình tham số. Việc tăng số cụm $k > 6$ chỉ làm phân rã cụm thành các nhóm nhỏ lẻ ($< 5$ khách), không mang lại giá trị gia tăng [`reports/c2_audit.md`: Mục 3].

### 8.2. Hướng mở rộng và Lộ trình tiếp theo
1. **Chuyển dịch sang Item-Item Collaborative Filtering (Gợi ý sản phẩm):**
   - Thay vì ép buộc mỗi khách hàng phải thuộc về một cụm duy nhất, việc áp dụng mô hình gợi ý theo cặp sản phẩm (ví dụ: Matrix Factorization / Implicit Feedback ALS) sẽ giải quyết triệt để vấn đề "khách không mua lại album cũ" bằng cách đề xuất các album tương tự mà khách chưa từng sở hữu [`reports/evaluation.md`: Mục 5.3].
2. **Thu thập dữ liệu Telemetry (Streaming & Listening Logs):**
   - Cần bổ sung bảng theo dõi lượt nghe (`TrackListenHistory`: timestamp, duration_played, skipped) để nắm bắt thị hiếu tiêu thụ thực tế thay vì chỉ có dữ liệu giao dịch tĩnh.
3. **Quy trình tái huấn luyện định kỳ (Periodic Re-clustering):**
   - Khi hệ thống thu hút thêm khách hàng mới đạt mốc $n \ge 200$, tự động chạy lại pipeline `src/pipeline.py` và `src/evaluation.py` để cập nhật lại các tâm cụm và kiểm định lại tính ổn định hold-out.

---

## 9. PHỤ LỤC (APPENDICES)

### 9.1. Bảng đối chiếu 10 bước đề bài và Tập tin đầu ra (10-Step Traceability Matrix)

| Bước | Tên nhiệm vụ theo đề bài | Mô tả chi tiết kỹ thuật | Tập tin báo cáo / Mã nguồn / Dữ liệu đầu ra | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Khám phá & Chất lượng dữ liệu** | Phân tích 11 bảng, kiểm tra 38 tiêu chí chất lượng, giải quyết duplicate | [`reports/data_quality.md`](file:///e:/ML/Chinook/reports/data_quality.md), [`reports/data_dictionary.md`](file:///e:/ML/Chinook/reports/data_dictionary.md) | **HOÀN THÀNH** |
| **2** | **Xây dựng tầng ETL** | Trích xuất read-only, làm sạch, tạo fact/dim, kiểm tra tính lũy kế | [`src/etl/extract.py`](file:///e:/ML/Chinook/src/etl/extract.py), [`src/etl/transform.py`](file:///e:/ML/Chinook/src/etl/transform.py), [`data/processed/`](file:///e:/ML/Chinook/data/processed/) | **HOÀN THÀNH** |
| **3** | **Khám phá kinh doanh (EDA)** | Phân tích doanh thu, hành vi, độ thưa 98.9%, phân phối long-tail | [`reports/eda_findings.md`](file:///e:/ML/Chinook/reports/eda_findings.md), [`figures/eda_revenue_trends.png`](file:///e:/ML/Chinook/figures/eda_revenue_trends.png) | **HOÀN THÀNH** |
| **4** | **Đặt bài toán kinh doanh** | Xác định mục tiêu bundle album, đối tượng phục vụ, tiêu chí thành công | [`reports/business_problem.md`](file:///e:/ML/Chinook/reports/business_problem.md) | **HOÀN THÀNH** |
| **5** | **Đặt bài toán kỹ thuật (Pha B)** | So sánh 4 không gian, Hopkins test, Curveball Null Model 300 lần | [`reports/problem_selection.md`](file:///e:/ML/Chinook/reports/problem_selection.md), [`figures/problem_selection_null_model.png`](file:///e:/ML/Chinook/figures/problem_selection_null_model.png) | **HOÀN THÀNH** |
| **6** | **Feature Engineering & Pipeline** | Xây dựng 6 biến thể, ItemFilter, IDF, L2, SVD, kiểm định rò rỉ | [`config/features.yaml`](file:///e:/ML/Chinook/config/features.yaml), [`src/features.py`](file:///e:/ML/Chinook/src/features.py), [`src/pipeline.py`](file:///e:/ML/Chinook/src/pipeline.py) | **HOÀN THÀNH** |
| **7** | **Huấn luyện & Chọn mô hình (C2)**| Thiết lập luật chọn, duyệt 1.184 cấu hình, sửa Null Model 200 lần | [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml), [`reports/model_comparison.md`](file:///e:/ML/Chinook/reports/model_comparison.md), [`reports/c2_audit.md`](file:///e:/ML/Chinook/reports/c2_audit.md) | **HOÀN THÀNH** |
| **8** | **Đánh giá & Tổng quát hóa (C3)** | Kiểm định 100 lần CV hold-out, nhạy cảm seed, kiểm định thời gian | [`src/evaluation.py`](file:///e:/ML/Chinook/src/evaluation.py), [`reports/evaluation.md`](file:///e:/ML/Chinook/reports/evaluation.md), [`figures/cluster_stability_boxplots.png`](file:///e:/ML/Chinook/figures/cluster_stability_boxplots.png) | **HOÀN THÀNH** |
| **9** | **Profile cụm & Đề xuất hành động**| Tính Lift, Coverage $\ge 3$ khách, non-parametric test, chiến dịch | [`src/profiling.py`](file:///e:/ML/Chinook/src/profiling.py), [`reports/cluster_profiles.md`](file:///e:/ML/Chinook/reports/cluster_profiles.md), [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv) | **HOÀN THÀNH** |
| **10**| **Đóng gói & Báo cáo cuối (Pha D)**| Lưu model joblib, kiểm tra tái lập 100%, viết predict.py & báo cáo | [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib), [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py), [`reports/final_report.md`](file:///e:/ML/Chinook/reports/final_report.md) | **HOÀN THÀNH** |

### 9.2. Danh sách cấu hình, Hạt ngẫu nhiên và Môi trường phần mềm
- **Mã cấu hình tối ưu:** `V2_album_count_svd5_dfFalse_KM_k5` [`reports/model_card.md`].
- **Hạt ngẫu nhiên cố định (Random Seeds):**
  - Khởi tạo K-Means & Pipeline: `random_state = 42`.
  - Kiểm tra độ nhạy (Seed Sensitivity): 25 seeds độc lập từ 0 đến 24.
  - Phân tách Hold-out Cross-Validation: 100 seeds ngẫu nhiên độc lập (`seed = 100 + run_idx`).
  - Kiểm định hoán vị Null Model: 200 lượt hoán vị Curveball độc lập.
- **Tính toán toàn vẹn dữ liệu (MD5 Data Hash):**
  - File `data/processed/fact_purchase.parquet`: Đã mã hóa và lưu trữ trực tiếp bên trong `models/final_model.joblib`.
  - Kiểm tra tính toàn vẹn tự động thông qua [`tests/test_reproducibility.py`].
- **Phiên bản thư viện phần mềm (Software Environment):**
  - Hệ điều hành: `Windows 11 Pro 64-bit`
  - Python: `3.13.1`
  - Scikit-Learn: `1.6.1`
  - NumPy: `2.2.3`
  - Pandas: `2.2.3`
  - SciPy: `1.15.2`
  - Joblib: `1.4.2`
  - Pytest: `9.1.1`
  - Matplotlib: `3.10.1`
  - Seaborn: `0.13.2`

---
*Báo cáo được hoàn thành và nghiệm thu toàn diện tại Phase D - Đảm bảo 100% số liệu truy nguyên về nguồn.*
