# TÀI LIỆU ĐẶC TẢ CÁC BIẾN THỂ ĐẶC TRƯNG (FEATURE DESCRIPTION REPORT)

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Giai đoạn:** PHA C1 - FEATURE ENGINEERING & PIPELINE VALIDATION  
**Nguồn dữ liệu đầu vào:** `data/processed/*.parquet` (Tầng phân tích sạch từ Pha ETL)  
**Tài liệu tham chiếu:** [reports/problem_selection.md](file:///e:/ML/Chinook/reports/problem_selection.md) & [reports/data_quality.md](file:///e:/ML/Chinook/reports/data_quality.md)  
**File cấu hình:** [config/features.yaml](file:///e:/ML/Chinook/config/features.yaml)  
**Mã nguồn thực thi:** [src/features.py](file:///e:/ML/Chinook/src/features.py) & [src/pipeline.py](file:///e:/ML/Chinook/src/pipeline.py)  

---

## 1. Nguyên Tắc Thiết Kế Không Gian Đặc Trưng (Core Principles)

Dựa trên kết quả khám phá dữ liệu (EDA) và kiểm định mô hình vô hiệu (Null Model Permutation Test) tại Pha B, không gian đặc trưng của dự án được xây dựng dựa trên 4 nguyên tắc cốt lõi:

1. **Bảo toàn từ vựng đầy đủ từ Catalog (Catalog-level Vocabulary Enforcement):**
   - Từ vựng cột (item vocabulary) của các ma trận được thiết lập cố định từ bảng danh mục catalog chuẩn (`dim_album` 347 cột, `dim_genre` 25 cột, `dim_artist` 275 cột, `dim_track` 3.503 cột).
   - Tuyệt đối không lấy từ vựng từ tập train nhằm đảm bảo chiều không gian nhất quán 100% giữa các tập dữ liệu, trong cross-validation và khi suy luận cho khách hàng mới (Inference).
2. **Loại bỏ biểu diễn "Số lượng mua" theo Track (Quantity = 1 Constraint):**
   - Bản chất của dữ liệu nhạc số Chinook là mỗi dòng giao dịch là một lượt mua quyền sở hữu duy nhất (`Quantity = 1`), mỗi cặp `(CustomerId, TrackId)` xuất hiện tối đa 1 lần.
   - Do đó, không gian đặc trưng chỉ sử dụng: (1) Nhị phân tương tác, (2) Đếm số track đã mua trong mục (đo chiều sâu quan tâm), (3) Tỷ lệ phủ catalog (`coverage_ratio`), và (4) Tỷ trọng phân bổ giỏ hàng (`share`).
3. **Cách ly hoàn toàn thông tin nhân khẩu học và quy mô chi tiêu (No Leakage & Demographics Isolation):**
   - `CustomerId`, `FirstName`, `LastName`, `Company`, `Address`, `City`, `State`, `Country`, `Email`: **KHÔNG ĐƯỢC ĐƯA VÀO ĐẶC TRƯNG HUẤN LUYỆN**, chỉ lưu ở bảng riêng [outputs/customer_interpretation_metadata.parquet](file:///e:/ML/Chinook/outputs/customer_interpretation_metadata.parquet) để diễn giải và đặt tên cụm sau khi mô hình hội tụ.
   - Tổng chi tiêu (`total_spending`), số lượng hóa đơn, tần suất mua: **KHÔNG DÙNG LÀM ĐẶC TRƯNG PHÂN CỤM**. EDA đã chứng minh chi tiêu (~37-39$) và số bài mua (~36-38 bài) của toàn bộ 59 khách hàng có hệ số biến thiên ($CV < 0.05$) gần như hằng số. Đưa vào sẽ không giúp phân hóa thị hiếu âm nhạc mà còn có nguy cơ để "quy mô" lấn át "gu thưởng thức".
4. **Chuẩn hóa $L_2$ hai lần để khoảng cách Euclid tương đương khoảng cách Cosine:**
   - Chuẩn hóa $L_2$ theo hàng trước SVD giúp cân bằng vector khách hàng.
   - Chuẩn hóa $L_2$ lần cuối sau SVD đảm bảo mọi vector tiềm ẩn đều nằm trên mặt cầu đơn vị ($\|u\|_2 = 1$), do đó:
     $$\|u - v\|_2^2 = \|u\|_2^2 + \|v\|_2^2 - 2 \langle u, v \rangle = 2(1 - \cos(u, v))$$
   - Giúp các thuật toán phân cụm dựa trên khoảng cách Euclid (K-Means, Ward Hierarchical) hoạt động tương đương tối ưu trên khoảng cách Cosine.

---

## 2. Bảng So Sánh Chi Tiết 6 Biến Thể Đặc Trưng (Variant Comparison Table)

Tất cả 6 biến thể được chốt cấu hình trước khi chạy mô hình tại [config/features.yaml](file:///e:/ML/Chinook/config/features.yaml) và đã được sinh ra đầy đủ tại thư mục `outputs/`:

| Biến Thể | Tên Biến Thể | Thực Thể (Grain) | Metric Đo Lường | Shape Ma Trận Thô | Mật Độ Thô (Density) | Cột Bị Lọc (`min_cust>=2`) | Cột Giữ Lại | Shape Không Gian Tiềm Ẩn | Explained Variance ($k$) | Tệp Lưu Trữ Parquet |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **V1** | **Album Binary + IDF** | `AlbumId` | Nhị phân $\in \{0, 1\}$ | $59 \times 347$ | 6.35% (1.301 ô) | 82 cột (43 chưa bán, 39 hiếm) | 265 cột | $59 \times 10$ | **39.94%** ($k=10$) | [`outputs/feature_matrix_V1_album_binary.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V1_album_binary.parquet) |
| **V2** | **Album Count + IDF** | `AlbumId` | Đếm track $\in \mathbb{N}_{\ge 0}$ | $59 \times 347$ | 6.35% (1.301 ô) | 82 cột (43 chưa bán, 39 hiếm) | 265 cột | $59 \times 10$ | **34.68%** ($k=10$) | [`outputs/feature_matrix_V2_album_count.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V2_album_count.parquet) |
| **V3** | **Album Coverage** | `AlbumId` | Tỷ lệ phủ $\in [0, 1]$ | $59 \times 347$ | 6.35% (1.301 ô) | 82 cột (43 chưa bán, 39 hiếm) | 265 cột | $59 \times 10$ | **31.99%** ($k=10$) | [`outputs/feature_matrix_V3_album_coverage.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V3_album_coverage.parquet) |
| **V4** | **Genre Share** | `GenreId` | Tỷ trọng $\in [0, 1]$ | $59 \times 25$ | 29.83% (440 ô) | 1 cột (Opera chưa bán) | 24 cột | $59 \times 5$ | **74.78%** ($k=5$) | [`outputs/feature_matrix_V4_genre_share.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V4_genre_share.parquet) |
| **V5** | **Artist Binary + IDF** | `ArtistId` | Nhị phân $\in \{0, 1\}$ | $59 \times 275$ | 5.69% (923 ô) | 142 cột (110 chưa bán, 32 hiếm) | 133 cột | $59 \times 10$ | **45.44%** ($k=10$) | [`outputs/feature_matrix_V5_artist_binary.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V5_artist_binary.parquet) |
| **V6** | **Composite Album + Genre** | Ghép khối | Kết hợp trọng số ($0.7 / 0.3$) | $59 \times 372$ | 7.93% (1.741 ô) | 83 cột (82 album + 1 genre) | 289 cột | $59 \times 10$ | **35.49%** ($k=10$) | [`outputs/feature_matrix_V6_composite_album_genre.parquet`](file:///e:/ML/Chinook/outputs/feature_matrix_V6_composite_album_genre.parquet) |

---

## 3. Phân Tích Chi Tiết Từng Biến Thể Đặc Trưng

### 3.1. Biến Thể V1: Album Binary + IDF (`V1_album_binary`)
- **Định nghĩa toán học:**
  $$X_{ij}^{(\text{raw})} = \mathbb{I}(\text{Khách hàng } i \text{ đã mua ít nhất 1 bài hát thuộc Album } j)$$
  Áp dụng trọng số làm mượt tần suất ngược (Smooth IDF):
  $$\text{idf}_j = \ln\left(\frac{N + 1}{\text{df}_j + 1}\right) + 1.0$$
  Sau đó chuẩn hóa vector hàng theo chuẩn $L_2$, chiếu qua TruncatedSVD ($k=10$), và chuẩn hóa $L_2$ lần cuối.
- **Kích thước & Mật độ:** Ma trận thô $59 \times 347$, mật độ 6.35% (trên tập bán là 7.25%). Sau `ItemFilter(min_customers=2)`, giữ lại 265 album, loại bỏ 82 album không mang tín hiệu cụm. Không gian tiềm ẩn có kích thước $59 \times 10$.
- **Lý do lựa chọn (Why):** Trọng tâm bài toán kinh doanh đã xác định ở Pha B (Null Model $p = 0.0033$). Nhị phân tập trung vào sự hiện diện của gu âm nhạc, không bị méo mó bởi các album dài tập nhiều bài.
- **Rủi ro & Hạn chế:** Không phân biệt được giữa khách hàng chỉ mua vãng lai 1 bài trong album với khách hàng mua trọn vẹn 12 bài của album.

### 3.2. Biến Thể V2: Album Count + IDF (`V2_album_count`)
- **Định nghĩa toán học:**
  $$X_{ij}^{(\text{raw})} = \text{Số lượng bài hát thuộc Album } j \text{ mà Khách hàng } i \text{ đã mua}$$
  Áp dụng `ItemFilter(min_customers=2)` $\rightarrow$ `IDFTransformer` $\rightarrow$ `Normalizer(L2)` $\rightarrow$ `TruncatedSVD(k=10)` $\rightarrow$ `Normalizer(L2)`.
- **Kích thước & Mật độ:** Ma trận thô $59 \times 347$. Tổng số lượt mua trong ma trận đúng bằng 2.240 lượt (khớp 100% với `fact_purchase`).
- **Lý do lựa chọn (Why):** Bổ sung **chiều sâu quan tâm (Depth of Interest)**. Khách hàng mua 5 bài hát của album Iron Maiden thể hiện độ gắn kết và sở thích mạnh mẽ hơn nhiều so với khách hàng chỉ mua 1 bài ngẫu nhiên.
- **Rủi ro & Hạn chế:** Các album có quy mô bài hát lớn (như album tuyển tập greatest hits 20-30 bài) có thể có giá trị đếm lớn hơn tự nhiên so với album thông thường (8-10 bài).

### 3.3. Biến Thể V3: Album Coverage Ratio (`V3_album_coverage`)
- **Định nghĩa toán học:**
  $$X_{ij}^{(\text{raw})} = \frac{\text{Số lượng bài hát đã mua thuộc Album } j}{\text{Tổng số bài hát có trong catalog của Album } j (\text{dim\_album.total\_tracks})}$$
  Giá trị mỗi ô nằm hoàn toàn trong đoạn $[0.0, 1.0]$. Giá trị trung bình của các ô có mua là $16.79\%$.
- **Kích thước & Mật độ:** Ma trận thô $59 \times 347$. Sau lọc còn 265 cột, nén thành vector 10 chiều.
- **Lý do lựa chọn (Why):** Khắc phục nhược điểm quy mô album của V2. Phản ánh trực tiếp mức độ hoàn thành album, phục vụ trực tiếp chương trình kinh doanh **"Complete My Album"** và **"Bundle Completion"**.
- **Rủi ro & Hạn chế:** Một khách mua 1 bài của album mini 2 bài sẽ có coverage = 50%, cao hơn khách mua 4 bài của album 10 bài (coverage = 40%). Do đó cần kết hợp đánh giá song song với V2.

### 3.4. Biến Thể V4: Genre Share (`V4_genre_share`)
- **Định nghĩa toán học:**
  $$X_{ij}^{(\text{raw})} = \frac{\text{Số bài hát thuộc Thể loại } j \text{ mà Khách hàng } i \text{ đã mua}}{\text{Tổng số bài hát Khách hàng } i \text{ đã mua trên mọi thể loại}}$$
  Tổng mỗi hàng đúng bằng $1.0000$ ($\sum_j X_{ij} = 1.0$).
- **Kích thước & Mật độ:** Ma trận thô $59 \times 25$. Mật độ đạt 29.83% (440 ô khác 0). 24 thể loại đã bán đều có $\ge 2$ khách hàng (chỉ có thể loại *Opera* chưa bán bị loại). Chiếu qua `TruncatedSVD(k=5)`.
- **Lý do lựa chọn (Why):** **Tuyệt đối không dùng nhị phân cho Thể loại.** Do catalog chỉ có 24 thể loại và các thể loại đại chúng như Rock, Latin, Metal xuất hiện ở hầu như mọi khách hàng (~31% ô có giá trị), biểu diễn nhị phân sẽ gần như toàn số 1 (triệt tiêu phân hóa). Tỷ trọng phân bổ giỏ hàng (Share) phản ánh chính xác cấu trúc ngân sách thưởng thức (ví dụ: 65% Rock vs 10% Rock).
- **Rủi ro & Hạn chế:** Độ thô cao (chỉ có 24 chiều), không thể dùng để thiết kế khuyến mãi bundle chi tiết theo từng album cụ thể.

### 3.5. Biến Thể V5: Artist Binary + IDF (`V5_artist_binary`)
- **Định nghĩa toán học:**
  $$X_{ij}^{(\text{raw})} = \mathbb{I}(\text{Khách hàng } i \text{ đã mua ít nhất 1 bài hát của Nghệ sĩ } j)$$
  Sau đó áp dụng `ItemFilter(min_customers=2)`, `IDFTransformer`, và chiếu qua `TruncatedSVD(k=10)`.
- **Kích thước & Mật độ:** Ma trận thô $59 \times 275$. Lọc bỏ 142 nghệ sĩ (110 chưa bán và 32 nghệ sĩ chỉ có đúng 1 người mua), giữ lại 133 nghệ sĩ có giao thoa $\ge 2$ khách hàng.
- **Lý do lựa chọn (Why):** Không gian trung gian giữa Album và Genre, độ đậm đặc 9.48% (Pha B: $p = 0.0033$). Rất hiệu quả cho việc đặt tên cụm và thấu hiểu lòng trung thành với nghệ sĩ (Artist Loyalty).
- **Rủi ro & Hạn chế:** Số lượng nghệ sĩ nhiều hơn thể loại nhưng ít hơn album, có một số nghệ sĩ chỉ ra mắt 1 album duy nhất (tương đương với album).

### 3.6. Biến Thể V6: Composite Album + Genre (`V6_composite_album_genre`)
- **Định nghĩa toán học:**
  Sử dụng `ColumnTransformer` ghép nối hai khối độc lập được chuẩn hóa và cân bằng trọng số:
  $$X_{\text{composite}} = \left[ 0.7 \times \text{Norm}_{L_2}(X_{\text{album\_count}} \cdot \text{IDF}),\ 0.3 \times \text{Norm}_{L_2}(X_{\text{genre\_share}}) \right]$$
  Sau đó chuẩn hóa $L_2$ ma trận ghép, giảm chiều qua `TruncatedSVD(k=10)` và chuẩn hóa $L_2$ lần cuối.
- **Kích thước & Mật độ:** Ma trận thô $59 \times 372$ (347 cột Album + 25 cột Genre). Mật độ 7.93% (1.741 ô). Lọc bỏ 83 cột (82 album + 1 genre), giữ lại 289 cột đặc trưng.
- **Lý do lựa chọn (Why):** **Phương án đa tầng hoàn thiện nhất.** Kết hợp hài hòa giữa độ phân giải sản phẩm cụ thể của Album (trọng số 70%) với tính ổn định và bao quát của Thể loại âm nhạc (trọng số 30%). Tránh việc cụm bị phân mảnh quá mức theo từng album đơn lẻ.
- **Rủi ro & Hạn chế:** Phức tạp hơn về mặt kiến trúc; cần cố định trọng số khối (0.7 / 0.3) trong cấu hình để đảm bảo tính tái lập.

---

## 4. Phân Tích Phương Sai Giải Thích SVD Trên Lưới Số Chiều (SVD Grid Analysis)

Bảng tổng hợp tỷ lệ phương sai giải thích tích lũy (Cumulative Explained Variance Ratio) được tính toán tự động qua hàm `evaluate_svd_grid()` trên lưới $k \in \{5, 8, 10, 15\}$:

| Biến Thể Đặc Trưng | $k = 5$ | $k = 8$ | $k = 10$ (Số chiều chọn) | $k = 15$ | Nhận Xét & Lựa Chọn Số Chiều |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V1 (Album Binary + IDF)** | 20.03% | 32.33% | **39.94%** | 56.93% | Với $N=59$, chọn $k=10$ giữ được xấp xỉ 40% phương sai mà không bị quá khớp. |
| **V2 (Album Count + IDF)** | 17.39% | 28.17% | **34.68%** | 49.39% | Phương sai phân tán nhẹ do độ sâu đếm, $k=10$ đạt gần 35%. |
| **V3 (Album Coverage)** | 15.98% | 25.89% | **31.99%** | 45.98% | Coverage có độ biến thiên mượt, $k=10$ là điểm cân bằng tối ưu. |
| **V4 (Genre Share)** | **74.78%** | 86.56% | 91.25% | 97.11% | Do chỉ có 24 chiều, chọn $k=5$ đã giải thích được gần **75%** cấu trúc thể loại. |
| **V5 (Artist Binary + IDF)** | 22.82% | 37.09% | **45.44%** | 61.81% | Artist cô đọng hơn Album, $k=10$ đạt trên **45%** phương sai. |
| **V6 (Composite Album + Genre)** | 18.10% | 28.94% | **35.49%** | 50.05% | Khối ghép ổn định, $k=10$ giải thích 35.49% toàn bộ thông tin tương tác. |

---

## 5. Lý Do Loại Bỏ Không Gian Khách × Track (Track-Level Rejection)

Theo kết luận tại [reports/problem_selection.md](file:///e:/ML/Chinook/reports/problem_selection.md), việc tạo biến thể phân cụm ở cấp độ Track đã bị **LOẠI BỎ CHÍNH THỨC** vì các căn cứ thực nghiệm không thể chối cãi:
1. **Sụp đổ khoảng cách trực giao (Orthogonal Collapse):** **97.49% số cặp khách hàng** có khoảng cách Cosine $\ge 0.95$. Trung vị khoảng cách là $1.0000$ (tuyệt đối không có bài hát chung).
2. **Thất bại trước Null Model:** Điểm Silhouette của K-Means trên ma trận Track chỉ đạt $0.0027$ với $p$-value $= 0.0797 > 0.05$. Thuật toán phân cụm không thể phân biệt dữ liệu thật với ma trận ngẫu nhiên hoán vị.
3. **Phân mảnh Long-tail cực đoan:** 87.1% bài hát chỉ có đúng 1 người mua duy nhất trong lịch sử. Nếu cố tình gom cụm trên Track, mô hình chỉ phân cụm dựa trên nhiễu số học.
