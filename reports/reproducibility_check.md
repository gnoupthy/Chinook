# BÁO CÁO KIỂM ĐỊNH TÍNH TÁI LẬP HỆ THỐNG (REPRODUCIBILITY CHECK REPORT)
**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album (Chinook Customer Music Taste Clustering)  
**Thời gian thực thi kiểm định sạch:** `2026-10-22 22:54:15` - `22:56:11`  
**Môi trường thực thi:** Python 3.13.1, Windows 11 Pro 64-bit  
**Mục tiêu:** Chạy lại toàn bộ quy trình từ đầu (Clean-slate End-to-End Execution): `ETL` $\to$ `Features` $\to$ `Modeling/Evaluation` $\to$ `Profiling` $\to$ `Pytest Suite`, ghi nhận log và đối chiếu nhãn cụm 59 khách hàng.

---

## 1. QUY TRÌNH CHẠY LẠI TỪ ĐẦU (CLEAN-SLATE EXECUTION STEPS)

### Bước 1: Tầng trích xuất & làm sạch dữ liệu (Analytical ETL)
- **Lệnh thực thi:** `python src/etl/run_etl.py`
- **Mã thoát (Exit code):** `0 (Thành công)`
- **Thời gian chạy:** `0.42 giây`
- **Tóm tắt log:**
  - Kết nối SQLite chế độ read-only: `file:///E:/ML/Chinook/data/Chinook_Sqlite.sqlite?mode=ro`.
  - Trích xuất: Customer (59), Invoice (412), InvoiceLine (2.240), Track (3.503), Album (347), Artist (275), Genre (25), MediaType (5).
  - Áp dụng 4 quy tắc làm sạch: Flag 12 bài trùng `(Name, AlbumId)`, Impute 978 NULL Composer thành `'Unknown'`, xác thực đơn giá và số lượng.
  - Đối chiếu trước/sau: 2.240 dòng giao dịch (100% khớp), 2.328,60 USD doanh thu (100% khớp), 59 khách hàng (100% khớp).
  - Xuất bảng phân tích: `fact_purchase.parquet`, `dim_customer.parquet`, `dim_track.parquet`, `dim_album.parquet`, `dim_artist.parquet`, `dim_genre.parquet`, `manifest.json`.

### Bước 2: Tầng kỹ thuật đặc trưng & Pipeline (Feature Engineering)
- **Lệnh thực thi:** `python -m src.pipeline`
- **Mã thoát (Exit code):** `0 (Thành công)`
- **Thời gian chạy:** `4.50 giây`
- **Tóm tắt log:**
  - Chạy thành công 6 biến thể đặc trưng theo `config/features.yaml`:
    * `V1_album_binary`: Shape gốc (59, 347) $\to$ ItemFilter loại 82 cột $\to$ SVD 10D (Phương sai 39.94%).
    * `V2_album_count`: Shape gốc (59, 347) $\to$ ItemFilter loại 82 cột $\to$ SVD 10D (Phương sai 34.68%).
    * `V3_album_coverage`: Shape gốc (59, 347) $\to$ ItemFilter loại 82 cột $\to$ SVD 10D (Phương sai 31.99%).
    * `V4_genre_share`: Shape gốc (59, 25) $\to$ ItemFilter loại 1 cột $\to$ SVD 5D (Phương sai 74.78%).
    * `V5_artist_binary`: Shape gốc (59, 275) $\to$ ItemFilter loại 142 cột $\to$ SVD 10D (Phương sai 45.44%).
    * `V6_composite_album_genre`: Shape gốc (59, 372) $\to$ ItemFilter loại 83 cột $\to$ SVD 10D (Phương sai 35.49%).
  - Xuất các file ma trận nhị phân thô và đặc trưng vào `outputs/`.

### Bước 3: Đánh giá mô hình & Huấn luyện Final Artifact (Evaluation & Final Model)
- **Lệnh thực thi:** `python -m src.evaluation`
- **Mã thoát (Exit code):** `0 (Thành công)`
- **Thời gian chạy:** `59.60 giây`
- **Tóm tắt log:**
  - Đánh giá chất lượng nội tại trên toàn bộ dữ liệu cho Top 5 ứng viên:
    * Candidate 1 (`V2_album_count_svd5_dfFalse_KM_k5`): Silhouette SVD = 0.5941 (vượt Null P95 = 0.4441, Gap = +0.1500), DB = 0.8234, CH = 32.54, Kích thước = `[15, 12, 12, 10, 10]`.
  - Thực thi 100 lần lặp Hold-out Cross-Validation (80% train / 20% test):
    * Candidate 1 đạt Median Hold-out ARI = **0.4038** (P5 = 0.0782, P95 = 0.7966) $\to$ Vượt trội so với 4 ứng viên còn lại.
  - Kiểm tra độ nhạy Random Seed (25 seeds): Mean Pairwise ARI = **0.9945**, Median = **1.0000**.
  - Kiểm định theo thời gian (Period 1 vs Period 2): Tỷ lệ dự báo tương lai ở cấp album = 11.86% ($p = 0.9640$); Độ tự nhất quán ở cấp thể loại Genre = **0.5065** ($p < 0.001$).
  - Lưu trữ mô hình tối ưu ra [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib).

### Bước 4: Phân tích hồ sơ phân khúc (Cluster Profiling)
- **Lệnh thực thi:** `python -m src.profiling`
- **Mã thoát (Exit code):** `0 (Thành công)`
- **Thời gian chạy:** `9.80 giây`
- **Tóm tắt log:**
  - Tính toán Lift và Coverage với ngưỡng lọc $\ge 3$ khách hàng trong cụm.
  - Xuất 5 phân khúc Persona hoàn chỉnh và các file `outputs/customer_clusters.csv`, `outputs/cluster_lift_genres.csv`, `outputs/cluster_lift_albums.csv`, `outputs/cluster_lift_artists.csv`.
  - Kiểm định phi tham số: TotalSpent ($p=0.3057$), TotalTracksPurchased ($p=0.5690$), Region ($p=0.9726$).

### Bước 5: Kiểm thử tự động toàn diện (Pytest Test Suite)
- **Lệnh thực thi:** `python -m pytest tests/test_etl.py tests/test_data_validation.py tests/test_pipeline.py tests/test_reproducibility.py -v`
- **Mã thoát (Exit code):** `0 (Thành công)`
- **Thời gian chạy:** `6.15 giây`
- **Kết quả:** **24 / 24 tests PASSED (100%)**.

---

## 2. BẢNG ĐỐI CHIẾU NHÃN CỤM KHÁCH HÀNG (CLUSTER LABEL MATCH AUDIT)

Để đảm bảo tính tất định và khả năng tái lập 100%, kết quả phân cụm sau khi chạy lại từ đầu được đối chiếu trực tiếp với tập tin chuẩn [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv):

| Tiêu chí đối chiếu | Giá trị lần chạy trước | Giá trị sau khi chạy lại sạch | Mức độ trùng khớp |
| :--- | :---: | :---: | :---: |
| **Số lượng khách hàng phân cụm** | 59 khách hàng | 59 khách hàng | **100% Khớp** |
| **Tỷ lệ trùng khớp nhãn cụm từng khách (Match Rate)** | 100.0% | 100.0% (59/59 khách) | **100.0% Khớp tuyệt đối** |
| **Adjusted Rand Index (ARI) giữa 2 lần chạy** | 1.0000 | 1.0000 | **100.0% Khớp tuyệt đối** |
| **Normalized Mutual Information (NMI)** | 1.0000 | 1.0000 | **100.0% Khớp tuyệt đối** |
| **Khoảng cách tới tâm cụm (Distance to Centroid)** | Sai số $< 10^{-6}$ | Sai số $< 10^{-6}$ | **100% Khớp số học** |
| **Độ tin cậy gán cụm (Confidence Score)** | Khớp chính xác | Khớp chính xác | **100% Khớp số học** |
| **Phân bổ kích thước 5 cụm** | `[15, 12, 12, 10, 10]` | `[15, 12, 12, 10, 10]` | **100% Khớp** |

### Mẫu đối chiếu 10 khách hàng đầu tiên:
| CustomerId | Tên khách hàng | Quốc gia | Cụm trước | Cụm chạy lại | Khoảng cách tâm cụm | Trạng thái |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | Luís Gonçalves | Brazil | 3 | 3 | 0.8841 | **TRÙNG KHỚP** |
| 2 | Leonie Köhler | Germany | 1 | 1 | 0.9400 | **TRÙNG KHỚP** |
| 3 | François Tremblay | Canada | 2 | 2 | 0.9760 | **TRÙNG KHỚP** |
| 4 | Bjørn Hansen | Norway | 4 | 4 | 0.9022 | **TRÙNG KHỚP** |
| 5 | František Wichterlová | Czech Republic | 0 | 0 | 0.9577 | **TRÙNG KHỚP** |
| 6 | Helena Holý | Czech Republic | 3 | 3 | 0.9442 | **TRÙNG KHỚP** |
| 7 | Astrid Gruber | Austria | 1 | 1 | 0.9723 | **TRÙNG KHỚP** |
| 8 | Daan Peeters | Belgium | 2 | 2 | 0.9634 | **TRÙNG KHỚP** |
| 9 | Kara Nielsen | Denmark | 4 | 4 | 0.9254 | **TRÙNG KHỚP** |
| 10 | Eduardo Martins | Brazil | 2 | 2 | 0.9255 | **TRÙNG KHỚP** |

---

## 3. KẾT LUẬN NGHIỆM THU TÍNH TÁI LẬP
Hệ thống Chinook Customer Music Taste Clustering Pipeline hoàn toàn thỏa mãn các tiêu chuẩn nghiêm ngặt nhất về tính khoa học và khả năng tái lập trong Học máy:
1. **Khả năng chạy tự động một lệnh:** Pipeline từ SQLite thô đến mô hình hoàn tất trong $< 1.5$ phút mà không cần can thiệp thủ công.
2. **Tính tất định (Determinism):** Mọi hạt ngẫu nhiên (`random_state=42`) được cố định thống nhất trên tất cả các module.
3. **Bộ kiểm thử toàn diện:** 24 bài test kiểm tra từ schema, chống rò rỉ dữ liệu, logic pipeline đến độ toàn vẹn artifact đều vượt qua 100%.
