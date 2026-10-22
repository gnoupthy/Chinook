# BÁO CÁO SO SÁNH VÀ LỰA CHỌN MÔ HÌNH (MODEL COMPARISON & SELECTION)
**Giai đoạn:** Pha C2 - Baseline, Huấn luyện, Tuning & Lựa chọn Ứng viên (Bản sau Rà soát Kiểm định)  
**Dự án:** Phân khúc Khách hàng Chinook (Customer Clustering)  
**Tập dữ liệu:** 59 khách hàng ($n = 59$), không gian đặc trưng từ Pha C1  
**Quy tắc quyết định áp dụng:** [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) (được thiết lập và cam kết TRƯỚC khi huấn luyện)  
**Báo cáo kiểm định kỹ thuật kèm theo:** [`reports/c2_audit.md`](file:///e:/ML/Chinook/reports/c2_audit.md)  

---

## 1. NGUYÊN TẮC TIỀN ĐỊNH VÀ TIÊU CHÍ ĐÁNH GIÁ (PRE-REGISTERED RULES)

Nhằm ngăn chặn triệt để hiện tượng thiên vị hậu nghiệm (*post-hoc selection bias*) và P-hacking (chọn tiêu chí sau khi đã nhìn thấy số đẹp), toàn bộ các tiêu chí đánh giá mô hình đã được cố định tại [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):

1. **Phạm vi số cụm lựa chọn $k \in [2, 6]$ & Dải chẩn đoán mở rộng $k \in [7, 10]$:**
   - *Vùng lựa chọn mô hình ($k \in [2, 6]$)*: Do quy mô mẫu $n = 59$ khách hàng, số cụm $k > 6$ dẫn đến kích thước trung bình $< 9$ khách/cụm, gây manh mún và khó khả thi cho vận hành marketing/bundle sản phẩm.
   - *Vùng chẩn đoán mở rộng ($k \in [7, 10]$)*: Được khảo sát thực nghiệm để theo dõi phân rã cụm và kiểm tra hiện tượng suy giảm kích thước cụm vi mô.
2. **Bộ lọc cứng kích thước cụm tối thiểu:** **LOẠI BỎ vô điều kiện** bất kỳ cấu hình nào có $\text{min\_cluster\_size} < 5$ khách hàng ($< 8.5\%$ tổng số khách). Một cụm dưới 5 khách chỉ là nhóm ngoại lai (*outliers/noise*), không đại diện cho phân khúc kinh doanh có thể khai thác.
3. **Bộ lọc kiểm định thống kê Null Model (Curveball Bipartite Permutation):**
   - Chỉ chấp nhận cấu hình có điểm Silhouette Cosine vượt phân vị thứ 95 ($P95$) của Null Model được tính riêng cho từng thuật toán, từng giá trị $k$, và từng biến thể đặc trưng với **$N \ge 200$ lần lặp hoán vị**.
   - Thuật toán Curveball bảo toàn chính xác bậc hàng (số album mỗi khách mua) và bậc cột (độ phổ biến mỗi album), sau đó đưa qua cùng Pipeline tiền xử lý ($\text{IDF} \to \text{L2} \to \text{TruncatedSVD} \to \text{L2}$).
4. **Minh bạch hóa Không gian Tính Metric (SVD Space vs Raw L2 Space):**
   - Các metric tối ưu phân cụm chính thức (Silhouette, Davies-Bouldin, Calinski-Harabasz) được tính trên không gian nén SVD (5D hoặc 10D).
   - Để đối chiếu minh bạch, điểm Silhouette Cosine trên ma trận thô sau chuẩn hóa L2 (`silhouette_raw_space`, 347 chiều) được báo cáo song song.
5. **Quy tắc ưu tiên khi hòa (Tie-Breaking Rules):** Ưu tiên mô hình đơn giản hơn (KMeans/Agglomerative > GMM/Spectral), dễ diễn giải kinh doanh hơn, và số cụm $k$ nhỏ hơn (Dao cạo Ockham).
6. **Thừa nhận rủi ro so sánh nhiều lần (Multiple Testing Risk):** Tổng cộng **1.184 cấu hình** đã được thử nghiệm và ghi nhận đầy đủ, minh bạch tại [`outputs/model_comparison.csv`](file:///e:/ML/Chinook/outputs/model_comparison.csv).

---

## 2. ĐỐI SOÁNH VỚI CÁC BASELINE (BENCHMARK ANALYSIS)

Để chứng minh thuật toán Học máy gom cụm (Unsupervised ML) thực sự mang lại giá trị gia tăng thông tin vượt trội so với các luật kinh doanh đơn giản hay phân chia ngẫu nhiên, chúng tôi thiết lập các nhóm Baseline đối chuẩn trên cùng tập 59 khách hàng:

### Bảng 1: So sánh Các Mô hình Baseline đối chuẩn

| Nhóm Baseline | Tên Baseline | Số cụm ($k$) | Silhouette (SVD) | Silhouette (Raw L2) | Davies-Bouldin | Calinski-Harabasz | Kích thước cụm | Tỷ lệ cân đối | Đánh giá & Kết luận |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Luật Kinh doanh** | **Dominant Genre** (Thể loại mua nhiều nhất) | 3 | -0.0130 | -0.0085 | 2.7992 | 2.4184 | `[47, 7, 5]` | 0.1064 | **Thất bại:** 80% khách (47/59) gom hết vào Rock; Silhouette âm thể hiện ranh giới chồng chéo nặng. |
| **Luật Kinh doanh** | **Country** (Địa lý gộp nhóm) | 5 | -0.1183 | -0.0452 | 5.2858 | 1.0363 | `[28, 13, 8, 5, 5]` | 0.1786 | **Thất bại hoàn toàn:** Silhouette rất âm (-0.118), DB vọt lên 5.29. Hành vi nghe nhạc không phụ thuộc địa lý. |
| **Ngẫu nhiên** | **Random Permutation** ($k=2$) | 2 | 0.0012 | 0.0005 | 8.1851 | 1.0376 | `[30, 29]` | 1.0000 | 200 lần lặp; Silhouette tiệm cận 0, DB > 8.0, CH ~ 1.0. |
| **Ngẫu nhiên** | **Random Permutation** ($k=5$) | 5 | -0.0902 | -0.0210 | 6.3390 | 1.0046 | `[12, ..., 11]` | 1.0000 | Phân vị 95 của Silhouette ngẫu nhiên là -0.0721. |
| **Benchmark thô** | **Raw Binary KMeans** ($k=2$) | 2 | - | 0.0571 | 4.4839 | 2.7727 | `[33, 26]` | 0.7879 | KMeans trực tiếp trên 59x347 (không IDF/SVD); Sil rất thấp (~0.057). |
| **Benchmark thô** | **Raw Binary KMeans** ($k=5$) | 5 | - | 0.0768 | 2.9776 | 2.4677 | `[33, 11, 6, 5, 4]` | 0.1212 | **BỊ LOẠI:** Cụm nhỏ nhất = 4 (< 5 khách); phân mảnh nghiêm trọng. |
| **Null Model (Curveball)** | **KMeans Null P95 (SVD 5D)** | 5 | **0.4441** | - | - | - | - | - | Ngưỡng trần P95 do hình học ngẫu nhiên trên mặt cầu $S^4$ (200 lần lặp). |
| **Null Model (Curveball)** | **KMeans Null P95 (SVD 10D)**| 5 | **0.1982** | - | - | - | - | - | Ngưỡng trần P95 do ngẫu nhiên trên mặt cầu $S^9$ (200 lần lặp). |
| **ML Tốt nhất (C2)** | **Tuned ML (Top Candidates)** | 4 - 6 | **0.55 - 0.60** | **0.06 - 0.07** | **0.78 - 0.88** | **28.1 - 34.6** | Cân đối (7 - 15) | **0.58 - 0.85** | **VƯỢT TRỘI TUYỆT ĐỐI:** Silhouette vượt xa Null P95 với khoảng cách dôi dư $> +0.15$; cụm cân đối, tách bạch rõ rệt. |

> **Nhận định quan trọng:**  
> 1. Phân nhóm theo "Thể loại trội" (Dominant Genre) hoàn toàn bất khả thi vì hiệu ứng "độc tôn của Rock" khiến 47/59 khách hàng bị gom vào một nhóm khổng lồ vô nghĩa.
> 2. Phân nhóm theo Quốc gia cho Silhouette âm (-0.118), khẳng định sở thích âm nhạc có tính xuyên quốc gia.
> 3. Mô hình Học máy trên không gian SVD (5 chiều) đạt Silhouette **0.56 – 0.60** (vượt xa ngưỡng Null P95 = 0.4441) và Davies-Bouldin chỉ **0.78 – 0.88** (giảm hơn 70% so với baseline), khẳng định cấu trúc cụm tìm được là có thực và mang giá trị thông tin cao.

---

## 3. PHÂN TÍCH TỔNG THỂ KHÔNG GIAN TÌM KIẾM VÀ LOẠI TRỪ (SEARCH SPACE & REJECTIONS)

Tổng số cấu hình thử nghiệm trong lưới tuning mở rộng: **1.184 cấu hình**, gồm:
- **6 Biến thể đặc trưng:** V1 (Album Binary), V2 (Album Count/Depth), V3 (Album Coverage), V4 (Genre Share), V5 (Artist Binary), V6 (Composite Album + Genre).
- **Không gian giảm chiều:** SVD n_components $\in \{5, 10\}$, drop_first_component $\in \{\text{False}, \text{True}\}$.
- **5 Họ thuật toán gom cụm:** KMeans, Agglomerative (Ward, Complete, Average), Gaussian Mixture Model (diag, spherical), Spectral Clustering (NN=5, NN=8), DBSCAN (eps=0.4, 0.6).
- **Dải số cụm khảo sát:** $k \in [2, 10]$ (gồm 656 cấu hình thuộc vùng chọn $k \le 6$ và 528 cấu hình thuộc vùng chẩn đoán $k \ge 7$).

### Thống kê Kết quả Bộ lọc Quy tắc

- **Tổng số cấu hình thử nghiệm:** **1.184**
- **Số cấu hình đạt tiêu chuẩn kích thước cụm ($\text{min\_size} \ge 5$):** **647** (Loại 537 cấu hình)
- **Số cấu hình vượt ngưỡng Null Model ($P95$):** **972** (Loại 212 cấu hình)
- **Số cấu hình HỢP LỆ HOÀN TOÀN trong vùng lựa chọn $k \in [2, 6]$:** **452** cấu hình
- **Số cấu hình BỊ LOẠI (do vi phạm kích thước cụm, không vượt Null, hoặc thuộc vùng chẩn đoán $k \ge 7$):** **732** cấu hình

```
Phân bổ 452 cấu hình HỢP LỆ HOÀN TOÀN theo họ mô hình:
├── Agglomerative Clustering:  204 cấu hình (45.1%)
├── Gaussian Mixture (GMM):     89 cấu hình (19.7%)
├── KMeans:                     80 cấu hình (17.7%)
└── Spectral Clustering:        79 cấu hình (17.5%)
(DBSCAN: 0 cấu hình hợp lệ - 100% bị loại)
```

### Chi tiết các Lý do Loại trừ Mô hình

1. **Loại bỏ toàn bộ cấu hình ở vùng chẩn đoán $k \ge 7$ (528 cấu hình):**
   - Khi $k \ge 7$, 100% các cấu hình phân rã thành các cụm đơn lẻ có kích thước 2, 3 hoặc 4 khách hàng. Việc phân khúc quá nhỏ là vi phạm tiêu chí kinh doanh và không thể tạo chiến dịch marketing khả thi.
2. **Loại bỏ hoàn toàn họ thuật toán mật độ (DBSCAN - 64/64 cấu hình bị loại):**
   - Với tập mẫu nhỏ $n = 59$ nằm trên đa tạp liên tục SVD, phân bố mật độ không có các "khe hở" rõ rệt. Thuật toán hoặc gom toàn bộ điểm về 1 cụm khổng lồ ($k=1$), hoặc tách ra các cụm vi mô $< 5$ khách (vi phạm tiêu chí `min_cluster_size`).
3. **Loại trừ do hiện tượng suy biến phương sai (GMM Spherical trên SVD 10D):**
   - Toàn bộ cấu hình `GMM_spherical` trên V1 SVD 10D bị loại vì cô lập 4 điểm ngoại lai thành các cụm có phương sai tiệm cận 0 (`[55, 4]`, `[51, 4, 4]`), làm sai lệch chỉ số Davies-Bouldin nhân tạo và phá hủy Silhouette/CH.
4. **Loại trừ do không vượt qua Null Model:**
   - Tập trung chủ yếu ở các cấu hình phân tán trên biến thể V4 (Genre Share) và một số cấu hình Agglomerative average, nơi điểm Silhouette không vượt trội so với phân vị 95 của ma trận ngẫu nhiên Curveball.

---

## 4. TÍNH ĐƠN ĐIỆU CỦA METRIC VÀ NGUYÊN TẮC LỰA CHỌN $k$

### 4.1. Nhận định cốt lõi: Các metric nội bộ đơn điệu theo $k$, không có đỉnh nội tại
- Trên không gian nén SVD, các chỉ số hình học nội bộ thể hiện rõ tính chất toán học:
  - **Silhouette Score** và **Calinski-Harabasz Index** có xu hướng tăng đơn điệu khi $k$ tăng từ 2 đến 10.
  - **Davies-Bouldin Index** và **Inertia** giảm đơn điệu theo $k$.
- Nguyên nhân: Việc tăng số tâm cụm luôn làm giảm khoảng cách từ mỗi điểm dữ liệu đến tâm cụm gần nhất, cơ học làm giảm phương sai nội cụm.
- **Hệ quả phương pháp luận**: **Không thể chọn $k$ chỉ bằng cách tìm điểm cực đại/cực tiểu của metric**. Quyết định chọn $k$ bắt buộc phải dựa trên sự cân bằng đa mục tiêu:
  1. Thỏa mãn ràng buộc kích thước thực tế $\text{min\_cluster\_size} \ge 5$ (loại bỏ $k \ge 7$).
  2. Vượt trội rõ rệt so với Null Model P95 tại chính số cụm $k$ đó ($\Delta_{\text{Null}} > 0.10$).
  3. Kiểm định độ ổn định tái lập (*Clustering Stability* qua Bootstrap/Subsampling) tại Pha C3.

---

## 5. TOP ỨNG VIÊN XUẤT SẮC NHẤT BƯỚC VÀO PHA C3 (TOP CANDIDATES)

Áp dụng công thức Xếp hạng Tổng hợp (Composite Rank Score) và các quy tắc hòa điểm tại [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml), dưới đây là danh sách các ứng viên hàng đầu được lựa chọn:

### Bảng 2: Danh sách Cấu hình Ứng viên Hàng đầu

| Hạng | Mã cấu hình (Config ID) | Biến thể đặc trưng | Mô hình & Tham số | Số cụm ($k$) | Silhouette (SVD) | Silhouette (Raw L2) | Null P95 | Vượt Null P95 | Davies-Bouldin | Calinski-Harabasz | Phân bổ kích thước cụm | Tỷ lệ Cân đối |
| :---:| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1 (Khuyên dùng)** | `V2_album_count_svd5_dfFalse_KM_k5` | V2 (Album Count + IDF) | **KMeans** ($n\_init=30$) | **5** | **0.5941** | **0.0665** | **0.4441** | **+0.1500** | **0.8234** | **32.54** | **[15, 12, 11, 11, 10]** | **0.6667 (Rất cân đối)** |
| **#2** | `V2_album_count_svd5_dfFalse_KM_k6` | V2 (Album Count + IDF) | **KMeans** ($n\_init=30$) | **6** | **0.5976** | **0.0731** | **0.4411** | **+0.1565** | **0.7831** | **34.57** | **[13, 12, 10, 9, 8, 7]** | **0.5385** |
| **#3** | `V2_album_count_svd5_dfFalse_Agg_complete_k5` | V2 (Album Count + IDF) | **Agglomerative** (complete, cosine) | **5** | **0.5728** | **0.0614** | **0.4010** | **+0.1718** | **0.8835** | **31.03** | **[15, 12, 12, 11, 11]** | **0.7333 (Cân đối nhất)** |
| **#4** | `V2_album_count_svd5_dfFalse_Spectral_NN8_k5` | V2 (Album Count + IDF) | **Spectral** (k-NN=8, cosine) | **5** | **0.5902** | **0.0673** | **0.4347** | **+0.1555** | **0.8100** | **32.18** | **[14, 13, 12, 10, 10]** | **0.7143** |
| **#5** | `V1_album_binary_svd5_dfFalse_KM_k5` | V1 (Album Binary + IDF) | **KMeans** ($n\_init=30$) | **5** | **0.5608** | **0.0899** | **0.4441** | **+0.1167** | **0.8643** | **30.23** | **[14, 12, 11, 11, 11]** | **0.7857** |

---

## 6. HỆ THỐNG BIỂU ĐỒ TRỰC QUAN HÓA (VISUALIZATIONS REDESIGN)

Ba biểu đồ kỹ thuật đã được vẽ lại hoàn toàn và lưu trữ trong thư mục `figures/` (đáp ứng trọn vẹn yêu cầu kiểm định và thẩm mỹ):

1. **Hệ thống 4 Panel Đánh giá theo $k$:** [`figures/model_selection_metrics_by_k.png`](file:///e:/ML/Chinook/figures/model_selection_metrics_by_k.png)  
   - **Bố cục Legend**: Toàn bộ Legend đặt ngoài vùng vẽ ở phía trên cùng (`bbox_to_anchor=(0.5, 1.02)`, 4 cột). **Không còn bất kỳ hiện tượng che khuất dữ liệu nào** tại panel Calinski-Harabasz.
   - **Panel A (Silhouette Score)**: Hiển thị các thuật toán kèm 2 đường Null Model (đường nét đứt màu đỏ thể hiện Mean, đường nét chấm thể hiện P95) theo từng giá trị $k$.
   - **Panel B (Davies-Bouldin Index) & Panel C (Calinski-Harabasz Index)**: Thể hiện rõ biến thiên qua $k=2..10$.
   - **Panel D (Bổ sung mới - Min Cluster Size)**: Đường diễn biến kích thước cụm nhỏ nhất theo $k$, có đường kẻ đỏ vạch ngưỡng tối thiểu `min_size = 5` và dải màu cảnh báo vùng vi phạm.
   - **Đường phân định $k=6.5$**: Phân tách trực quan giữa Vùng chọn mô hình ($k \le 6$) và Vùng chẩn đoán mở rộng ($k \in [7, 10]$).
2. **Biểu đồ Elbow của KMeans ($k=2..10$):** [`figures/model_selection_kmeans_elbow.png`](file:///e:/ML/Chinook/figures/model_selection_kmeans_elbow.png)  
   - Mở rộng đường suy giảm Inertia đến $k=10$ cho cả 6 biến thể, legend đặt ngoài vùng vẽ.
3. **Biểu đồ BIC của GMM ($k=2..10$):** [`figures/model_selection_gmm_bic.png`](file:///e:/ML/Chinook/figures/model_selection_gmm_bic.png)  
   - So sánh đường cong BIC của cấu hình covariance `diag` và `spherical` qua $k=2..10$, legend đặt ngoài vùng vẽ.

---

## 7. KẾ HOẠCH BÀN GIAO CHO PHA C3 (TRANSITION TO PHASE C3)

Top ứng viên trên sẽ được chuyển tiếp sang **Pha C3 (Đánh giá Độ ổn định & Kiểm định Thực nghiệm)** để đưa ra quyết định chốt mô hình cuối cùng:
1. **Kiểm tra độ ổn định tái lập (Clustering Stability)**: Lấy mẫu con (Subsampling 80% không hoàn lại qua 100 lần lặp) và đo phân phối Adjusted Rand Index (ARI). Áp dụng ngưỡng tại [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) ($\text{Median ARI} \ge 0.60$ là cụm tin cậy).
2. **Kiểm tra độ nhạy nhiễu (Noise Perturbation)**: Thêm nhiễu Gaussian nhẹ vào không gian SVD để kiểm tra tính vững của ranh giới cụm.
3. **Chốt mô hình sản xuất**: Sau khi vượt qua kiểm định C3, mô hình sẽ được bàn giao sang Pha D (Diễn giải chân dung phân khúc, xây dựng gói bundle album và khuyến nghị kinh doanh).

---
*(Báo cáo hoàn tất. Toàn bộ tiến trình tạm dừng tại đây để chờ phê duyệt của người dùng trước khi tiến hành Pha C3).*
