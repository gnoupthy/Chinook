# THẺ MÔ HÌNH HỌC MÁY (MODEL CARD)
**Tên mô hình:** Chinook Customer Taste Clustering Pipeline  
**Mã cấu hình:** `V2_album_count_svd5_dfFalse_KM_k5`  
**Phiên bản:** `1.0.0`  
**Ngày phát hành:** `2026-10-22`  
**Định dạng lưu trữ:** [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib)  
**Mã nguồn suy luận:** [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py)  
**Mã nguồn kiểm thử:** [`tests/test_reproducibility.py`](file:///e:/ML/Chinook/tests/test_reproducibility.py)  

---

## 1. THÔNG TIN CHUNG VỀ MÔ HÌNH (MODEL DETAILS)

- **Nhà phát triển:** Đội ngũ Kỹ thuật & Khoa học Dữ liệu Chinook.
- **Loại mô hình:** Mô hình Phân cụm Không giám sát (Unsupervised Clustering Pipeline) tích hợp tiền xử lý và giảm chiều trong một đối tượng tuần tự Scikit-Learn.
- **Thuật toán chính:**
  - Tiền xử lý: `ItemFilter(min_customers=2)` $\to$ `IDFTransformer(smooth_idf=True)` $\to$ `Normalizer(norm='l2')`.
  - Giảm chiều: `TruncatedSVD(n_components=5, algorithm='randomized', random_state=42)` $\to$ `Normalizer(norm='l2')`.
  - Gom cụm: `KMeans(n_clusters=5, n_init=30, random_state=42)`.
- **Môi trường phần mềm:**
  - Python: `3.13.1`
  - Scikit-Learn: `1.6.1`
  - NumPy: `2.2.3`
  - Pandas: `2.2.3`
  - SciPy: `1.15.2`
  - Joblib: `1.4.2`

---

## 2. MỤC ĐÍCH SỬ DỤNG (INTENDED USE)

### Trường hợp Sử dụng Dự kiến (Intended Uses):
- **Phân khúc thị hiếu âm nhạc (Music Taste Segmentation):** Nhận diện các nhóm khách hàng có sở thích âm nhạc đồng điệu dựa trên lịch sử mua album để phục vụ marketing mục tiêu.
- **Thiết kế gói sản phẩm (Album Bundle Design):** Gợi ý các gói album đồng thương hiệu / cùng thể loại tương ứng với đặc trưng Lift của từng phân khúc (Pha D).
- **Cá nhân hóa nội dung email & trang chủ:** Tự động điều chỉnh banner và đề xuất danh mục bài hát theo Persona của từng cụm khi khách hàng đăng nhập.

### Đối tượng Người dùng Mục tiêu:
- Nhóm Marketing Campaign Managers và Product Managers tại nền tảng Chinook Digital Media.

---

## 3. CÁC TRƯỜNG HỢP KHÔNG NÊN DÙNG (OUT-OF-SCOPE / MISUSE)

1. **KHÔNG dùng để dự báo tổng chi tiêu hoặc xếp hạng giá trị khách hàng (Customer Lifetime Value):**
   - Kết quả kiểm định Kruskal-Wallis khẳng định không có sự khác biệt có ý nghĩa thống kê về chi tiêu giữa các cụm ($p = 0.3057$, $\eta^2 = 0.0153$). Việc gán nhãn "VIP" hay "High-Value" dựa trên phân cụm này là hoàn toàn sai lệch về mặt thống kê.
2. **KHÔNG dùng để dự báo việc mua lại cùng một album cũ trong tương lai:**
   - Kiểm định theo thời gian chứng minh khách hàng âm nhạc số không mua lại album đã sở hữu ($p = 0.9640$ so với ngẫu nhiên ở mức album). Để dự báo danh mục tương lai, bắt buộc phải dùng Collaborative Filtering Item-Item hoặc phân loại ở cấp độ Thể loại (Genre).
3. **KHÔNG dùng cho khách hàng quá mới (Cold-Start Problem với $< 2$ album):**
   - Nếu khách hàng chỉ mới mua 1 album, vector đại diện trong không gian SVD chưa đủ thông tin định hướng, độ tin cậy gán cụm sẽ rất thấp ($< 0.40$).

---

## 4. DỮ LIỆU HUẤN LUYỆN (TRAINING DATA)

- **Nguồn dữ liệu:** Cơ sở dữ liệu Chinook SQLite chuẩn, đã qua tầng làm sạch ETL tại [`data/processed/`](file:///e:/ML/Chinook/data/processed/).
- **Tập dữ liệu đầu vào:** [`data/processed/fact_purchase.parquet`](file:///e:/ML/Chinook/data/processed/fact_purchase.parquet)
  - MD5 Data Hash: `cố định và lưu trong file joblib`
  - Số lượng khách hàng: **59** ($n = 59$)
  - Số lượng album trong từ vựng: **347** album
  - Tổng số giao dịch mua: **2.240** bản ghi
- **Quy tắc tiền xử lý dữ liệu:**
  - Mỗi cặp (Khách hàng, Track) là duy nhất.
  - Ma trận đặc trưng $59 \times 347$ biểu diễn số lượng bài hát đã mua trong từng album (`metric="count"`).
  - Thuật toán `ItemFilter` tự động loại bỏ các album chỉ có 1 khách hàng mua để triệt tiêu nhiễu đơn lẻ (*hapax legomena*).

---

## 5. THAM SỐ VÀ CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

### Tham số Siêu hình học (Hyperparameters):
- $k = 5$ cụm.
- $n\_components = 5$ (Không gian SVD 5 chiều giải thích $\approx 55\%$ phương sai hữu ích).
- $drop\_first\_component = \text{False}$ (Giữ lại thành phần 1 do đã áp dụng IDF làm phẳng độ phổ biến).
- $metric = \text{"cosine"}$ (Tương đương chuẩn hóa L2 trên mặt cầu đơn vị $S^4$).

### Các Chỉ số Đạt được:

| Chỉ số Đánh giá | Giá trị Đạt được | Mốc So sánh / Ngưỡng Quy định | Kết luận |
| :--- | :---: | :---: | :--- |
| **Silhouette (SVD 5D)** | **0.5941** | Null Model P95 = **0.4441** | **Vượt trội (+0.1500)**; phân tách cụm xuất sắc. |
| **Silhouette (Raw L2 347D)**| **0.0665** | Baseline thô = 0.0571 | Phản ánh tính thưa thớt của không gian gốc 347D. |
| **Davies-Bouldin Index** | **0.8234** | Baseline Heuristics > 2.80 | Cụm có độ nén cao, giảm hơn 70% sai số. |
| **Calinski-Harabasz Index** | **32.54** | Baseline Random $\approx 1.0$ | Phương sai liên cụm vượt trội so với nội cụm. |
| **Kích thước cụm nhỏ nhất** | **10 khách** | Ngưỡng cứng: $\text{min\_size} \ge 5$ | Đạt; phân bổ cân đối: `[15, 12, 12, 10, 10]`. |
| **Độ nhạy Random Seed** | **Mean ARI = 0.9945** | Ngưỡng kỳ vọng $\ge 0.90$ | Hội tụ vững chắc, không nhạy với khởi tạo. |
| **Hold-out 80/20 ARI Median** | **0.4038** | Ngưỡng lý tưởng $\ge 0.60$ | **Độ ổn định trung bình (Moderate)**; P95 = 0.7966. |
| **Độ nhất quán Thể loại tương lai**| **Median = 0.5065** | Mức ngẫu nhiên = 0.0500 | Tính nhất quán gu âm nhạc theo thời gian rất cao. |

---

## 6. HẠN CHẾ VÀ RỦI RO KHOA HỌC (LIMITATIONS)

1. **Cỡ mẫu nhỏ ($n = 59$):**  
   Mô hình được huấn luyện trên toàn bộ khách hàng hiện có của Chinook ($n=59$). Đây là cỡ mẫu hạn chế trong học máy. Mặc dù cấu hình đã vượt trội so với Null Model hoán vị, độ ổn định mẫu con (Hold-out ARI trung vị đạt 0.4038) cho thấy các ranh giới cụm vẫn có sự dao động nhất định khi tập dữ liệu bị biến động.
2. **Hiện tượng "Rock là dòng nhạc độc tôn" (Rock Dominance Bias):**  
   Thể loại Rock chiếm hơn 70% dung lượng catalog Chinook. Vì vậy, cả 5 phân khúc đều có nền tảng nghe Rock. Sự phân hóa giữa các cụm nằm ở các thể loại ngách đi kèm (Easy Listening/Soundtrack ở Cụm 0; Blues/Acoustic ở Cụm 1; Fusion/World ở Cụm 2; Metal/Urban ở Cụm 3; Punk/Symphonic ở Cụm 4).
3. **Khuyến cáo triển khai:**  
   Không nên xem phân cụm này là cố định vĩnh viễn. Cần chạy lại pipeline phân cụm định kỳ (Re-clustering) khi số lượng khách hàng tăng lên $n \ge 200$.

---

## 7. HƯỚNG DẪN SỬ DỤNG VÀ SUY LUẬN (HOW TO USE)

### 1. Suy luận cho Khách hàng Mới:
Sử dụng hàm `assign_cluster` trong file [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py):

```python
from src.predict import assign_cluster

# Khách hàng mới mua 4 album có ID: 1, 2, 120, 141
purchases = [1, 2, 120, 141]
result = assign_cluster(purchases, model_path="models/final_model.joblib")

print(f"Cụm gán: {result['cluster_id']}")
print(f"Tên phân khúc: {result['cluster_name']}")
print(f"Độ tin cậy: {result['confidence']:.4f}")
print(f"Khoảng cách tới tâm: {result['distance_to_centroid']:.4f}")
```

### 2. Kiểm tra Tính Tái lập Toàn vẹn:
Chạy bộ kiểm thử tự động với `pytest`:
```bash
pytest tests/test_reproducibility.py -v
```
*(Kết quả kiểm thử phải báo 4/4 test PASSED với tỷ lệ trùng khớp nhãn 100%).*
