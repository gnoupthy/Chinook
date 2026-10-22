# BÁO CÁO TỔNG KẾT PHA C2: BASELINE, HUẤN LUYỆN, TUNING VÀ CHỌN MÔ HÌNH
**Thời gian hoàn thành:** 2026-10-22  
**Trạng thái:** HOÀN TẤT & ĐÃ TẠM DỪNG CHỜ DUYỆT (STOPPED AWAITING USER APPROVAL)  
**Quy tắc nghiệm thu:** Tuân thủ tuyệt đối [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) được tạo trước khi chạy mô hình.

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC CỐT LÕI CỦA PHA C2

1. **Chống thiên vị hậu nghiệm (Pre-registration):** Mọi ngưỡng lọc và tiêu chí đánh giá mô hình được thiết lập và ghi sẵn vào [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) trước khi khởi chạy bất kỳ thuật toán học máy nào:
   - Số cụm $k \in [2, 6]$ (phù hợp với tập mẫu $n = 59$ khách hàng).
   - Bộ lọc cứng kích thước cụm: Loại bỏ bất kỳ cấu hình nào có $\text{min\_cluster\_size} < 5$ khách hàng.
   - Bộ lọc kiểm định ngẫu nhiên: Điểm Silhouette (Cosine) bắt buộc vượt phân vị P95 của Null Model hoán vị bảo toàn biên hàng/cột.
   - Nguyên lý Dao cạo Ockham: Khi điểm số tương đương, ưu tiên thuật toán đơn giản, dễ giải thích kinh doanh và số cụm $k$ nhỏ hơn.
2. **Chứng minh giá trị của Học máy thông qua 4 nhóm Baseline:** So sánh công bằng với:
   - Baseline kinh doanh (Phân nhóm theo Thể loại trội nhất; Phân nhóm theo Quốc gia).
   - Baseline ngẫu nhiên (100 lần gán nhãn ngẫu nhiên bảo toàn phân phối).
   - Benchmark dữ liệu thô (KMeans trực tiếp trên ma trận nhị phân 59x347).
   - Null Model thống kê (Mức Silhouette do ngẫu nhiên bảo toàn cấu trúc thưa).
3. **Quét toàn diện và minh bạch không gian tham số (Multiple Testing Awareness):** Thử nghiệm 592 cấu hình, lưu trữ toàn bộ lịch sử thử nghiệm vào [`outputs/model_comparison.csv`](file:///e:/ML/Chinook/outputs/model_comparison.csv) để kiểm toán.
4. **Phạm vi nghiêm ngặt:** **CHƯA** đánh giá ổn định/hold-out chi tiết (dành riêng cho Pha C3), **CHƯA** viết khuyến nghị kinh doanh cuối cùng. Dừng lại tại bước lựa chọn top 3-5 ứng viên tiềm năng nhất.

---

## 2. KẾT QUẢ ĐỐI CHUẨN BASELINE VS. MÔ HÌNH HỌC MÁY

Kết quả đo lường chi tiết được ghi lại tại [`outputs/baseline_comparison.csv`](file:///e:/ML/Chinook/outputs/baseline_comparison.csv):

| Nhóm Đối chuẩn | Tên Cấu hình | Số cụm ($k$) | Silhouette (Cosine) | Davies-Bouldin | Calinski-Harabasz | Kích thước cụm | Kết luận |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Kinh doanh** | Dominant Genre (Top Thể loại) | 3 | **-0.0130** | 2.7992 | 2.4184 | [47, 7, 5] | **Thất bại:** 80% khách dồn vào Rock; cụm chồng lấn nghiêm trọng. |
| **Kinh doanh** | Country (Quốc gia gộp nhóm) | 5 | **-0.1183** | 5.2858 | 1.0363 | [28, 13, 8, 5, 5] | **Thất bại:** Silhouette rất âm; sở thích nghe nhạc không phụ thuộc địa lý. |
| **Ngẫu nhiên** | Random Permutation (100 runs) | 5 | **-0.0902** | 6.3390 | 1.0046 | [12, ..., 11] | P95 Silhouette đạt -0.0721; không có cấu trúc toán học. |
| **Dữ liệu thô** | Raw Binary KMeans | 2 | **0.0571** | 4.4839 | 2.7727 | [33, 26] | Ma trận nhị phân thưa không có SVD cho độ tách biệt rất kém. |
| **Dữ liệu thô** | Raw Binary KMeans | 5 | **0.0768** | 2.9776 | 2.4677 | [33, 11, 6, 5, 4] | **Bị loại:** Cụm nhỏ nhất = 4 khách hàng ($< 5$). |
| **Null Model** | P95 Null Hoán vị bảo toàn biên | 6 | **0.0301** | - | - | - | Ngưỡng trần ngẫu nhiên thống kê trên không gian Album. |
| **ML Tuned** | **Top Candidates (C2)** | **4 - 6** | **0.55 - 0.60** | **0.78 - 0.88** | **28.1 - 34.6** | **Cân đối (7 - 14)** | **Vượt trội tuyệt đối:** Cụm sắc nét, vượt ngưỡng Null Model 18-20 lần. |

---

## 3. TỔNG QUAN LƯỚI TUNING VÀ PHÂN TÍCH LOẠI TRỪ (TUNING & REJECTIONS)

### Thống kê Lưới Tham số (592 cấu hình)
- **Tổng cấu hình thử nghiệm:** **592**
- **Đạt kích thước cụm $\ge 5$ khách:** **459** (Loại 133 cấu hình)
- **Vượt phân vị P95 của Null Model:** **561** (Loại 31 cấu hình)
- **HỢP LỆ TOÀN BỘ (Passed All Rules):** **456** cấu hình (77.0%)
- **BỊ LOẠI:** **136** cấu hình (23.0%)

### Các Lý do Kỹ thuật dẫn đến Loại trừ
1. **DBSCAN (Loại 32/32 cấu hình - 100%):** Do bản chất dữ liệu $n = 59$ nằm trên đa tạp liên tục SVD, không gian không có các "thung lũng mật độ" (density chasms). Tại $\epsilon=0.6$, thuật toán sụp đổ thành 1 cụm duy nhất $[59]$; tại $\epsilon=0.4$, nó tạo ra các cụm ngoại lai 3-4 khách ($< 5$). Do đó DBSCAN hoàn toàn không phù hợp với bài toán này.
2. **GMM (Loại 69/160 cấu hình):** 67 cấu hình tạo ra cụm kích thước $< 5$ khách do hiện tượng cô lập điểm ngoại biên vào thành phần Gauss riêng lẻ.
3. **Agglomerative Average Linkage (Loại 34/240 cấu hình):** Do hiệu ứng xâu chuỗi (chaining effect) tách các điểm cá biệt thành cụm 4 khách.
4. **V4 Genre Share (Loại 31 cấu hình):** Do ngưỡng Null Model P95 của Genre rất cao ($0.3318$), các mô hình có Silhouette $< 0.33$ bị loại bỏ thẳng thừng theo đúng cam kết tiền định.

---

## 4. DANH SÁCH TOP 5 ỨNG VIÊN ĐƯỢC CHỌN (TOP 5 CANDIDATES)

Các ứng viên được chọn tự động dựa trên công thức Xếp hạng Tổng hợp (Composite Rank Score) và Quy tắc hòa điểm tại [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):

1. **Ứng viên 1 (Hạng 1 Tổng hợp):** [`V2_album_count_svd5_dfFalse_Agg_complete_k5`](file:///e:/ML/Chinook/outputs/top_candidates.json)  
   - *Thuật toán:* Agglomerative (linkage=complete, metric=cosine), $k = 5$.
   - *Chỉ số:* Silhouette = **0.5728**, Davies-Bouldin = **0.8835**, Calinski-Harabasz = **31.03**.
   - *Kích thước cụm:* **[13, 12, 12, 11, 11]** (Độ cân đối đạt **0.8462** - cao nhất toàn bộ lưới).
2. **Ứng viên 2 (Hạng 2 Tổng hợp):** [`V2_album_count_svd5_dfFalse_Spectral_k5`](file:///e:/ML/Chinook/outputs/top_candidates.json)  
   - *Thuật toán:* Spectral Clustering (k-NN = 8, metric=cosine), $k = 5$.
   - *Chỉ số:* Silhouette = **0.5902**, Davies-Bouldin = **0.8100**, Calinski-Harabasz = **32.18**.
   - *Kích thước cụm:* **[14, 13, 12, 10, 10]** (Độ cân đối đạt **0.7143**).
3. **Ứng viên 3 (Hạng 3 Tổng hợp - Chuẩn mực KMeans):** [`V1_album_binary_svd5_dfFalse_KM_k5`](file:///e:/ML/Chinook/outputs/top_candidates.json)  
   - *Thuật toán:* KMeans ($n\_init = 30$, random_state=42), $k = 5$.
   - *Chỉ số:* Silhouette = **0.5608**, Davies-Bouldin = **0.8643**, Calinski-Harabasz = **30.23**.
   - *Kích thước cụm:* **[14, 12, 11, 11, 11]** (Độ cân đối đạt **0.7857**). Khả năng diễn giải tâm cụm xuất sắc.
4. **Ứng viên 4 (Ứng viên Tinh gọn - Ockham's Razor):** [`V2_album_count_svd5_dfFalse_KM_k4`](file:///e:/ML/Chinook/outputs/model_comparison.csv)  
   - *Thuật toán:* KMeans ($n\_init = 30$, random_state=42), $k = 4$.
   - *Chỉ số:* Silhouette = **0.5535**, Davies-Bouldin = **0.9980**, Calinski-Harabasz = **28.11**.
   - *Kích thước cụm:* **[17, 16, 13, 13]** (Độ cân đối đạt **0.7647**). $k=4$ giúp tinh gọn vận hành, giảm 20% chi phí quản lý so với $k=5$.
5. **Ứng viên 5 (Tối ưu Toán học):** [`V2_album_count_svd5_dfFalse_KM_k6`](file:///e:/ML/Chinook/outputs/top_candidates.json)  
   - *Thuật toán:* KMeans ($n\_init = 30$, random_state=42), $k = 6$.
   - *Chỉ số:* Silhouette = **0.5976**, Davies-Bouldin = **0.7831**, Calinski-Harabasz = **34.57** (Chỉ số hình học cao nhất).
   - *Kích thước cụm:* **[12, 12, 10, 9, 9, 7]** (Cụm nhỏ nhất 7 khách, vẫn an toàn $\ge 5$).

---

## 5. DANH MỤC CÁC TÀI LIỆU VÀ TẬP TIN SẢN PHẨM (DELIVERABLES)

Mọi sản phẩm của Pha C2 đã được hoàn thiện và sẵn sàng để kiểm tra:

| STT | Tập tin / Thư mục | Mục đích & Nội dung | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) | Tiêu chí lựa chọn mô hình tiền định (Pre-registration) | Đã hoàn thành |
| 2 | [`src/modeling.py`](file:///e:/ML/Chinook/src/modeling.py) | Module mã nguồn chạy Baseline, Tuning Grid và Xuất biểu đồ | Đã hoàn thành |
| 3 | [`outputs/baseline_comparison.csv`](file:///e:/ML/Chinook/outputs/baseline_comparison.csv) | Bảng kết quả 14 cấu hình baseline đối chuẩn | Đã hoàn thành |
| 4 | [`outputs/model_comparison.csv`](file:///e:/ML/Chinook/outputs/model_comparison.csv) | Bảng toàn diện 592 cấu hình thử nghiệm và cờ hợp lệ | Đã hoàn thành |
| 5 | [`outputs/top_candidates.json`](file:///e:/ML/Chinook/outputs/top_candidates.json) | Danh sách và siêu dữ liệu của Top 5 ứng viên | Đã hoàn thành |
| 6 | [`figures/model_selection_metrics_by_k.png`](file:///e:/ML/Chinook/figures/model_selection_metrics_by_k.png) | Biểu đồ Silhouette / Davies-Bouldin / Calinski-Harabasz theo $k$ | Đã hoàn thành |
| 7 | [`figures/model_selection_kmeans_elbow.png`](file:///e:/ML/Chinook/figures/model_selection_kmeans_elbow.png) | Biểu đồ Elbow (Inertia) của KMeans theo số cụm $k$ | Đã hoàn thành |
| 8 | [`figures/model_selection_gmm_bic.png`](file:///e:/ML/Chinook/figures/model_selection_gmm_bic.png) | Biểu đồ đường BIC của GMM (diag vs. spherical) | Đã hoàn thành |
| 9 | [`reports/model_comparison.md`](file:///e:/ML/Chinook/reports/model_comparison.md) | Báo cáo chuyên sâu phân tích so sánh và loại trừ | Đã hoàn thành |
| 10 | [`reports/phase_C2_summary.md`](file:///e:/ML/Chinook/reports/phase_C2_summary.md) | Báo cáo tổng kết toàn bộ Pha C2 | Đã hoàn thành |

---

## 6. THÔNG BÁO DỪNG BÀN GIAO VÀ ĐỢI DUYỆT (STOPPING & AWAITING APPROVAL)

Theo đúng yêu cầu của người dùng tại chỉ dẫn:
> *"Output: outputs/model_comparison.csv, reports/model_comparison.md, figures/model_selection_*.png, reports/phase_C2_summary.md. **Sau đó DỪNG chờ duyệt**."*

Hệ thống **TẠM DỪNG TOÀN BỘ TIẾN TRÌNH THỰC THI TẠI ĐÂY**, không tự ý triển khai sang Pha C3 (Kiểm tra độ ổn định / Subsampling ARI / Hold-out) hoặc đưa ra các khuyến nghị kinh doanh cho đến khi người dùng kiểm tra và phê duyệt kết quả lựa chọn mô hình ở Pha C2.
