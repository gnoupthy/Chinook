# BÁO CÁO RÀ SOÁT CHUYÊN SÂU PHA C2 (PHASE C2 AUDIT REPORT)
**Dự án**: Phân khúc Khách hàng Chinook - Phân tích Thói quen Âm nhạc  
**Tác giả**: Kỹ sư Học máy & Khoa học Dữ liệu  
**Trạng thái**: Hoàn thành Rà soát C2 - Sẵn sàng Trình duyệt trước Pha C3  
**Tập tin dữ liệu liên quan**:
- `outputs/null_model_results.csv` (1.010 dòng kết quả Null Model qua 200 lượt lặp Curveball)
- `outputs/model_comparison.csv` (1.184 cấu hình thực nghiệm từ $k=2..10$ qua 16 pipeline)
- `figures/model_selection_metrics_by_k.png` (Hệ thống 4 panel chuẩn hóa, legend đặt ngoài)
- `figures/model_selection_kmeans_elbow.png` (Biểu đồ Elbow mở rộng $k=2..10$)
- `figures/model_selection_gmm_bic.png` (Biểu đồ BIC mở rộng $k=2..10$)

---

## 1. KHẮC PHỤC TRIỆT ĐỂ NULL MODEL (CURVEBALL BIPARTITE PERMUTATION)

### 1.1. Bản chất sai sót của Null Model cũ và Mã tạo mới
- **Sai sót trước đây**: Báo cáo ban đầu sử dụng giá trị phân vị $\text{P95} = 0.0301$ như một hằng số cố định cho mọi $k$ và mọi thuật toán. Giá trị này thực chất được ước lượng thô trên ma trận nhị phân 347 chiều ở Pha B, hoàn toàn không phản ánh cấu trúc hình học của không gian sau khi nén qua Pipeline ($\text{IDF} \to \text{L2} \to \text{TruncatedSVD} \to \text{L2}$).
- **Quy trình tạo Null Model chuẩn xác**:
  1. Sử dụng thuật toán **Curveball** (Strona et al., 2014) để xáo trộn cạnh trên đồ thị hai phía Khách hàng $\times$ Mục (Album/Genre/Artist). Thuật toán này đảm bảo **bảo toàn bất biến bậc hàng** (số lượng album/nghệ sĩ mà mỗi khách hàng đã mua) và **bất biến bậc cột** (tổng số khách hàng đã mua từng album/nghệ sĩ).
  2. Mỗi ma trận ngẫu nhiên sinh ra được đưa qua **chính xác cùng Pipeline biến thể** ($\text{ItemFilter} \to \text{IDFTransformer} \to \text{TruncatedSVD} \to \text{Normalizer L2}$).
  3. Huấn luyện từng mô hình và đo `fast_cosine_silhouette` riêng cho từng bộ $(\text{variant}, \text{svd\_dim}, \text{model}, k)$ với **$N = 200$ lần lặp độc lập**.

```python
def curveball(adj_list: list[set], num_swaps: int, rng: np.random.Generator) -> list[set]:
    """Hoán vị bảo toàn chính xác bậc hàng và bậc cột của ma trận nhị phân tương tác."""
    n = len(adj_list)
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        set_i, set_j = adj_list[i], adj_list[j]
        shared = set_i & set_j
        diff_i, diff_j = set_i - shared, set_j - shared
        if len(diff_i) == 0 or len(diff_j) == 0:
            continue
        len_diff_i = len(diff_i)
        union_diff = list(diff_i | diff_j)
        rng.shuffle(union_diff)
        adj_list[i] = shared | set(union_diff[:len_diff_i])
        adj_list[j] = shared | set(union_diff[len_diff_i:])
    return adj_list
```

### 1.2. Bảng kết quả Null Model thực nghiệm ($N=200$ lần lặp)
Dưới đây là so sánh phân phối Silhouette của ma trận ngẫu nhiên (Null) và dữ liệu thực tế trên mô hình KMeans đối với 2 không gian SVD điển hình:

#### Biến thể V2_album_count (Không gian SVD 5D):
| Số cụm $k$ | Null Mean | Null Std | Null P95 | Silhouette Thật | Khoảng cách (Thật - Mean) | Khoảng cách (Thật - P95) | Vượt ngưỡng Null |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $k=2$ | 0.2884 | 0.0218 | **0.3242** | 0.3957 | +0.1073 | **+0.0715** | ĐẠT |
| $k=3$ | 0.3140 | 0.0318 | **0.3667** | 0.4468 | +0.1328 | **+0.0801** | ĐẠT |
| $k=4$ | 0.3533 | 0.0366 | **0.4093** | 0.5286 | +0.1753 | **+0.1193** | ĐẠT |
| $k=5$ | 0.3780 | 0.0437 | **0.4441** | **0.5941** | **+0.2161** | **+0.1500** | **ĐẠT (TỐI ƯU)** |
| $k=6$ | 0.3786 | 0.0406 | **0.4411** | 0.5976 | +0.2190 | **+0.1565** | ĐẠT |
| $k=7$ | 0.3771 | 0.0390 | 0.4391 | 0.6186 | +0.2415 | +0.1795 | Vi phạm min size |
| $k=8$ | 0.3779 | 0.0388 | 0.4355 | 0.6384 | +0.2605 | +0.2029 | Vi phạm min size |
| $k=9$ | 0.3737 | 0.0390 | 0.4377 | 0.6548 | +0.2811 | +0.2171 | Vi phạm min size |
| $k=10$ | 0.3738 | 0.0427 | 0.4468 | 0.6720 | +0.2982 | +0.2252 | Vi phạm min size |

#### Biến thể V1_album_binary (Không gian SVD 10D):
| Số cụm $k$ | Null Mean | Null Std | Null P95 | Silhouette Thật | Khoảng cách (Thật - Mean) | Khoảng cách (Thật - P95) | Vượt ngưỡng Null |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $k=2$ | 0.1320 | 0.0112 | **0.1497** | 0.1953 | +0.0633 | **+0.0456** | ĐẠT |
| $k=3$ | 0.1416 | 0.0124 | **0.1612** | 0.2152 | +0.0736 | **+0.0540** | ĐẠT |
| $k=4$ | 0.1552 | 0.0163 | **0.1852** | 0.2461 | +0.0909 | **+0.0609** | ĐẠT |
| $k=5$ | 0.1693 | 0.0175 | **0.1982** | **0.3040** | **+0.1347** | **+0.1058** | **ĐẠT (TỐI ƯU)** |
| $k=6$ | 0.1766 | 0.0191 | **0.2137** | 0.3683 | +0.1917 | **+0.1546** | ĐẠT |
| $k=7$ | 0.1850 | 0.0210 | 0.2313 | 0.4120 | +0.2270 | +0.1807 | Vi phạm min size |
| $k=8$ | 0.1934 | 0.0215 | 0.2384 | 0.4480 | +0.2546 | +0.2096 | Vi phạm min size |
| $k=9$ | 0.1989 | 0.0221 | 0.2448 | 0.4815 | +0.2826 | +0.2367 | Vi phạm min size |
| $k=10$ | 0.2044 | 0.0242 | 0.2557 | 0.5112 | +0.3068 | +0.2555 | Vi phạm min size |

### 1.3. Giải thích hình học không gian SVD số chiều thấp
- **Tại sao Null Silhouette ở SVD 10D tăng từ 0.13 lên 0.20 - 0.25 (đúng kỳ vọng lý thuyết)?**
  Trong không gian 10 chiều, khi nén 59 điểm ngẫu nhiên đồng đều lên mặt siêu cầu đơn vị $S^9$, hiện tượng "tập trung độ đo" (*concentration of measure*) khiến khoảng cách giữa các cặp điểm có xu hướng co cụm quanh trung vị $\approx \sqrt{2}$. Khi $k$ tăng, thuật toán KMeans phân chia không gian thành nhiều ô Voronoi nhỏ hơn, làm giảm khoảng cách nội cụm $a(i)$ nhanh hơn khoảng cách liên cụm $b(i)$, khiến Silhouette tăng tự nhiên từ $0.13$ lên $\approx 0.20$.
- **Tại sao Null Silhouette ở SVD 5D lại cao hơn đáng kể (đạt 0.28 - 0.38)?**
  Khi số chiều giảm xuống $d=5$, mặt cầu $S^4$ có diện tích bề mặt nhỏ hơn rất nhiều. Với cùng số lượng $n=59$ điểm, mật độ điểm trên đơn vị diện tích mặt cầu cao gấp hàng chục lần so với 10D. Khi đó, các điểm ngẫu nhiên tình cờ rơi gần nhau theo xác suất hình học, tạo ra các "cụm giả ngẫu nhiên" (*spurious clusters*). Do đó, ngưỡng P95 của Null Model ở 5D đạt mức $\approx 0.44$.
- **Ý nghĩa kiểm định**: Dữ liệu thật ở V2 SVD 5D đạt Silhouette $0.5941$ tại $k=5$, **vượt xa ngưỡng Null P95 (0.4441) với khoảng cách dôi dư cực lớn $+0.1500$**. Điều này bác bỏ hoàn toàn giả thuyết H0 (dữ liệu phân bố ngẫu nhiên), khẳng định cấu trúc phân khúc khách hàng là thực chất và có cơ sở hành vi vững chắc.

---

## 2. PHÂN ĐỊNH RÕ RÀNG KHÔNG GIAN TÍNH METRIC (SVD SPACE VS RAW SPACE)

| Không gian đánh giá | Số chiều | Độ dày (Density) | Trung vị Cosine giữa các khách | Silhouette Score ($k=5$, KMeans V2) | Bản chất hình học & Ý nghĩa phân tích |
|:---|:---:|:---:|:---:|:---:|:---|
| **SVD Space (L2-normalized)** | **5 chiều** | **100% (Đặc)** | **0.62** | **0.5941** | Không gian đa tạp sau khi khử nhiễu và gom các chiều tương quan. Đo lường sự phân tách thực tế giữa các nhóm thị hiếu. |
| **Raw Matrix Space (L2-normalized)** | **347 chiều** | **2.61% (Cực thưa)** | **1.00 (Trực giao)** | **0.0665** | Ma trận đếm gốc sau chuẩn hóa L2. Do độ thưa quá lớn, hầu hết khách hàng không mua trùng đúng album/track cụ thể nên vector trực giao nhau. |

### Kết luận chuẩn hóa metric:
1. **Toàn bộ quyết định thuật toán** (KMeans, Agglomerative, GMM, Spectral) và hàm mục tiêu phân cụm được thực thi trên **không gian SVD (5D/10D)**. Do đó, các chỉ số Silhouette, Davies-Bouldin và Calinski-Harabasz chính thức được định danh là **SVD-space metrics**.
2. **Silhouette trên ma trận thô (`silhouette_raw_space`)** đã được tính song song cho toàn bộ 1.184 cấu hình trong `outputs/model_comparison.csv`. Mức Silhouette thô $\approx 0.06 - 0.07$ là hoàn toàn bình thường đối với dữ liệu giỏ hàng bán lẻ có độ thưa $>97\%$; nó phản ánh đúng tính chất trực giao của ma trận mục và khẳng định sự cần thiết tuyệt đối của tầng giảm chiều SVD.

---

## 3. MỞ RỘNG DẢI $k \in [2, 10]$ & CHẨN ĐOÁN PHÂN RÃ CỤM (`min_cluster_size`)

### 3.1. Bảng chẩn đoán kích thước cụm theo $k$ (KMeans, V2_album_count, SVD 5D)
| $k$ | Nhóm phân loại | Silhouette (SVD) | Silhouette (Raw) | Min Size | Max Size | Phân bổ kích thước cụm | Vi phạm `min_size < 5` |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| 2 | Vùng chọn | 0.3957 | 0.0412 | 26 | 33 | `[33, 26]` | Không |
| 3 | Vùng chọn | 0.4468 | 0.0489 | 15 | 24 | `[24, 20, 15]` | Không |
| 4 | Vùng chọn | 0.5286 | 0.0592 | 11 | 18 | `[18, 17, 13, 11]` | Không |
| **5** | **Vùng chọn** | **0.5941** | **0.0665** | **10** | **15** | `[15, 12, 11, 11, 10]` | **Không (CỰC KỲ CÂN ĐỐI)** |
| 6 | Vùng chọn | 0.5976 | 0.0731 | 7 | 13 | `[13, 12, 10, 9, 8, 7]` | Không |
| 7 | **Vùng chẩn đoán** | 0.6186 | 0.0765 | **4** | 12 | `[12, 11, 10, 9, 7, 6, 4]` | **VI PHẠM (size = 4)** |
| 8 | **Vùng chẩn đoán** | 0.6384 | 0.0812 | **3** | 12 | `[12, 10, 9, 8, 7, 6, 4, 3]` | **VI PHẠM (size = 3)** |
| 9 | **Vùng chẩn đoán** | 0.6548 | 0.0845 | **3** | 11 | `[11, 9, 8, 8, 7, 6, 4, 3, 3]` | **VI PHẠM (size = 3)** |
| 10 | **Vùng chẩn đoán** | 0.6720 | 0.0890 | **2** | 10 | `[10, 8, 8, 7, 7, 6, 4, 4, 3, 2]` | **VI PHẠM (size = 2)** |

### 3.2. Kết luận then chốt từ thực nghiệm mở rộng $k=2..10$:
- Mặc dù Silhouette và Calinski-Harabasz tiếp tục tăng cơ học khi $k \ge 7$ (do số tâm cụm tăng làm giảm khoảng cách nội cụm), nhưng **toàn bộ các cấu hình từ $k=7$ trở đi đều xuất hiện các cụm cô lập chỉ có 2-4 khách hàng**.
- Trong bài toán kinh doanh, việc tạo ra một phân khúc chỉ có 2-3 khách hàng là vô nghĩa về mặt thương mại (không thể thiết kế chiến dịch bundle hay email automation riêng cho 2 người) và tiềm ẩn nguy cơ *overfitting* nghiêm trọng.
- Điều này chứng minh tính đúng đắn và chặt chẽ của quy tắc chặn trên $k \le 6$ đã thiết lập trong `config/decision_rules.yaml`.

---

## 4. GIẢI MÃ HIỆN TƯỢNG BẤT THƯỜNG CỦA GMM_SPHERICAL

### 4.1. Bản chất nghịch lý số liệu
Trên biến thể V1 (SVD 10D), mô hình Gaussian Mixture với ma trận hiệp phương sai hình cầu (`GMM_spherical`) thể hiện kết quả thoạt nhìn rất mâu thuẫn:
- **Davies-Bouldin**: $1.12 - 1.18$ (Thấp nhất - tốt nhất toàn bộ bảng so sánh).
- **Silhouette**: $0.12 - 0.16$ (Tệ nhất toàn bộ bảng).
- **Calinski-Harabasz**: $4.4 - 5.8$ (Tệ nhất toàn bộ bảng).

### 4.2. Nguyên nhân kỹ thuật: Suy biến phương sai (*Variance Collapse*) & Outlier Trapping
Qua kiểm tra chi tiết phân bổ nhãn và trọng số thành phần của mô hình:
1. **Phân bổ kích thước cụm bị thoái hóa cực đoan**:
   - Tại $k=2$: Phân bổ cụm là `[55, 4]` (1 cụm khổng lồ 55 khách, 1 cụm tí hon 4 khách).
   - Tại $k=3$: Phân bổ cụm là `[51, 4, 4]`.
   - Tại $k=5$: Phân bổ cụm là `[43, 4, 4, 4, 4]`.
2. **Cơ chế toán học của chỉ số Davies-Bouldin**:
   Công thức Davies-Bouldin dựa trên tỷ số giữa độ phân tán nội cụm $s_i, s_j$ và khoảng cách giữa hai tâm cụm $d(c_i, c_j)$:
   $$R_{ij} = \frac{s_i + s_j}{d(c_i, c_j)}, \quad DB = \frac{1}{k} \sum_{i=1}^k \max_{j \ne i} R_{ij}$$
   Khi thuật toán EM hội tụ vào 4 điểm ngoại lai xa xôi, phương sai của các thành phần tí hon này bị sụp đổ (*variance collapse*), $\sigma^2 \to 0$ ($\sigma^2 \approx 0.006$ so với $0.075$ của cụm chính). Do $s_j \approx 0$, tử số $s_i + s_j$ bị triệt tiêu mạnh, làm cho tỷ số $R_{ij}$ giảm xuống mức rất nhỏ, tạo ra một **ảo ảnh toán học rằng DB cực kỳ tốt**.
3. **Phá hủy Silhouette và Calinski-Harabasz**:
   Do 43-55 khách hàng còn lại bị nhồi nhét vào một cụm khổng lồ không hề có cấu trúc phân hóa, khoảng cách nội cụm của nhóm này rất lớn, khiến điểm Silhouette trung bình toàn cục rớt thảm hại về $0.12$ và CH sụp xuống $4.4$.
4. **Loại bỏ hợp lệ theo quy tắc**: Toàn bộ các cấu hình GMM spherical đều có cụm nhỏ nhất là $4 < 5$, do đó đã bị **bộ lọc cứng `passed_min_size = False` loại bỏ tự động 100%**, không thể lọt vào danh sách ứng viên.

---

## 5. CẢI TIẾN HỆ THỐNG BIỂU ĐỒ KỸ THUẬT (FIGURES REDESIGN)

Toàn bộ 3 biểu đồ trong thư mục `figures/` đã được tạo lại hoàn toàn mới:

1. **`figures/model_selection_metrics_by_k.png` (4 Panels chuẩn hóa)**:
   - **Bố cục Legend**: Chuyển toàn bộ Legend lên phía trên cùng bên ngoài vùng vẽ (`bbox_to_anchor=(0.5, 1.02)`, 4 cột). **Triệt tiêu hoàn toàn hiện tượng che khuất đường dữ liệu** ở panel Calinski-Harabasz.
   - **Panel A (Silhouette)**: Bổ sung 2 đường Null Model theo $k$ (đường nét đứt màu đỏ thể hiện Mean, đường nét chấm màu đỏ sẫm thể hiện ngưỡng P95).
   - **Panel B (Davies-Bouldin) & Panel C (Calinski-Harabasz)**: Thể hiện diễn biến rõ nét qua $k=2..10$.
   - **Panel D (Bổ sung mới - Min Cluster Size)**: Vẽ số khách hàng trong cụm nhỏ nhất theo $k$, kèm đường vạch đỏ nằm ngang và dải màu cảnh báo tại ngưỡng `min_size = 5`.
   - **Vùng phân định**: Đường thẳng đứng $k=6.5$ phân tách trực quan giữa **Vùng chọn mô hình ($k \le 6$)** và **Vùng chẩn đoán mở rộng ($k \in [7, 10]$)**.
2. **`figures/model_selection_kmeans_elbow.png`**:
   - Mở rộng đường Inertia từ $k=2$ đến $k=10$ cho cả 6 biến thể. Đưa legend ra ngoài góc phải trên (`bbox_to_anchor=(1.02, 1)`).
3. **`figures/model_selection_gmm_bic.png`**:
   - So sánh đường cong BIC của GMM diag vs GMM spherical qua $k=2..10$, legend đặt ngoài vùng vẽ.

---

## 6. KHẲNG ĐỊNH TÍNH ĐƠN ĐIỆU CỦA METRIC & TIÊU CHÍ CHỌN $k$

### 6.1. Nhận định cốt lõi: Các metric nội bộ đơn điệu theo $k$, không có đỉnh nội tại
- Trên không gian nén SVD, cả Silhouette và Calinski-Harabasz đều có xu hướng **tăng đơn điệu khi $k$ tăng từ 2 đến 10**, trong khi Davies-Bouldin và Inertia **giảm đơn điệu**.
- Hiện tượng này hoàn toàn bình thường trong học không giám sát trên không gian liên tục: việc thêm tâm cụm luôn làm giảm khoảng cách từ các điểm tới tâm gần nhất.
- **Hệ quả phương pháp luận**: **Không thể chọn $k$ tối ưu chỉ bằng cách tìm điểm cực trị của metric nội bộ**. Việc lựa chọn $k$ bắt buộc phải là một quyết định đa tiêu chí, kết hợp giữa:
  1. **Ràng buộc kích thước kinh doanh**: `min_cluster_size >= 5` (loại bỏ $k \ge 7$).
  2. **Kiểm định thống kê vượt trội**: Silhouette thực tế phải vượt ngưỡng Null P95 tại chính $k$ đó với khoảng cách an toàn (gap $> 0.10$).
  3. **Độ ổn định phân cụm (Clustering Stability)**: Đo bằng chỉ số Adjusted Rand Index (ARI) qua tái lấy mẫu Bootstrap/Subsampling tại Pha C3.

### 6.2. Danh sách 3 Ứng viên Xuất sắc nhất bước vào Pha C3
Dựa trên kết quả sau rà soát, 3 cấu hình dẫn đầu toàn diện (đạt điểm cao nhất, cân đối nhất, vượt trội Null Model) được chốt lại gồm:

| Xếp hạng | Mã cấu hình ứng viên | Biến thể đặc trưng | Mô hình | $k$ | Silhouette (SVD) | Silhouette (Raw) | Null P95 | Khoảng cách vượt Null | Min Size | Max Size | Tỷ lệ cân đối |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1 (Khuyên dùng)** | `V2_album_count_svd5_dfFalse_KM_k5` | V2 (Album Count + IDF) | **KMeans** | **5** | **0.5941** | 0.0665 | 0.4441 | **+0.1500** | **10** | **15** | **0.6667 (Rất cân đối)** |
| **#2** | `V2_album_count_svd5_dfFalse_KM_k6` | V2 (Album Count + IDF) | **KMeans** | **6** | **0.5976** | 0.0731 | 0.4411 | **+0.1565** | 7 | 13 | 0.5385 |
| **#3** | `V2_album_count_svd5_dfFalse_Agg_complete_k5` | V2 (Album Count + IDF) | **Agglomerative (Complete)** | **5** | **0.5728** | 0.0614 | 0.4010 | **+0.1718** | 11 | 15 | 0.7333 (Cân đối nhất) |

*(Toàn bộ các ứng viên trên đều có kích thước cụm nhỏ nhất $\ge 7$, vượt trội hoàn toàn so với Null P95, và sẽ được đưa vào bài kiểm định độ ổn định subsampling/bootstrap chi tiết tại Pha C3).*
