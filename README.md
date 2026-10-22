# CHINOOK CUSTOMER MUSIC TASTE CLUSTERING PIPELINE
> **Dự án Khoa học Dữ liệu & Học máy:** Phân khúc Khách hàng Chinook theo Thị hiếu Âm nhạc dựa trên Lịch sử Mua Album & Bài hát  
> **Tác giả:** Đội ngũ Kỹ thuật & Khoa học Dữ liệu Chinook  
> **Phiên bản:** `1.0.0` (Production-Ready Reproducible Pipeline)  
> **Môi trường:** Python 3.13.1 | Hệ điều hành: Windows / Linux / macOS  

---

## 1. MỤC TIÊU DỰ ÁN (PROJECT OBJECTIVES)
Dự án nhằm giải quyết bài toán kinh doanh thực tế tại nền tảng bán lẻ âm nhạc số **Chinook Digital Media**:
- **Khám phá và nhận diện phân khúc thị hiếu âm nhạc:** Khắc phục thách thức ma trận mua hàng cực thưa (độ thưa **98.92%** ở cấp độ bài hát) và sự đồng nhất về quy mô chi tiêu (~39.5 USD/khách hàng) bằng cách chiếu dữ liệu lên không gian tiềm ẩn đa tạp thông qua **TruncatedSVD** và **K-Means**.
- **Thiết kế gói sản phẩm thông minh (Album Bundles):** Tạo cơ sở dữ liệu định lượng (Lift & Coverage $\ge 3$ khách) để đề xuất các gói sản phẩm đóng gói (như Complete My Album, Vintage Cinema, World Fusion, Heavy Metal Pack) cho từng nhóm khách hàng mục tiêu.
- **Xây dựng Pipeline học máy tái lập 100%:** Thiết kế kiến trúc khép kín từ trích xuất SQLite thô, làm sạch Star-Schema, kiểm định rò rỉ dữ liệu, huấn luyện mô hình, kiểm tra ổn định Hold-out đến module suy luận độc lập (`src/predict.py`).

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN (DIRECTORY STRUCTURE)

```
Chinook/
├── .pytest_cache/                  # Cache thực thi của Pytest
├── config/                         # Các tệp cấu hình tham số tập trung (YAML)
│   ├── decision_rules.yaml         # Bộ luật quyết định đặt trước (Pre-registered Rules)
│   ├── etl.yaml                    # Cấu hình nguồn và đích của tầng ETL
│   └── features.yaml               # Danh sách 6 biến thể đặc trưng và tham số pipeline
├── data/                           # Dữ liệu nguồn và dữ liệu đã xử lý
│   ├── Chinook_Sqlite.sqlite       # Cơ sở dữ liệu SQLite gốc (Read-only)
│   └── processed/                  # Tầng dữ liệu phân tích sạch (Star-Schema Parquet)
│       ├── fact_purchase.parquet   # Bảng sự kiện hạt nhân (2.240 giao dịch)
│       ├── dim_customer.parquet    # Bảng chiều khách hàng (59 khách)
│       ├── dim_track.parquet       # Bảng chiều bài hát (3.503 bài)
│       ├── dim_album.parquet       # Bảng chiều album (347 album)
│       ├── dim_artist.parquet      # Bảng chiều nghệ sĩ (275 nghệ sĩ)
│       ├── dim_genre.parquet       # Bảng chiều thể loại (25 thể loại)
│       └── manifest.json           # Siêu dữ liệu kiểm tra tính toàn vẹn ETL
├── figures/                        # Hệ thống biểu đồ trực quan hóa khoa học
│   ├── cluster_pca_projection.png  # Chiếu không gian 2D PCA & t-SNE theo 5 cụm
│   ├── cluster_profiles_lift.png   # Biểu đồ Lift thể loại theo từng cụm
│   ├── cluster_silhouette_plot.png # Biểu đồ Silhouette từng khách hàng
│   ├── cluster_stability_boxplots.png # Hộp phân bổ ARI & Silhouette qua 100 lần Hold-out
│   ├── cluster_temporal_prediction.png# Kiểm định dự báo tương lai và tính nhất quán thể loại
│   └── eda_*.png                   # Hệ thống biểu đồ khám phá dữ liệu (Doanh thu, thưa, long-tail)
├── models/                         # Artifact mô hình tối ưu đã lưu trữ
│   └── final_model.joblib          # Artifact mô hình hoàn thiện (Pipeline, Centroids, Vocab, Hash)
├── outputs/                        # Dữ liệu xuất và bảng tổng hợp thực nghiệm
│   ├── baseline_comparison.csv     # Bảng so sánh mô hình với các Baseline chuẩn
│   ├── cleaning_log.csv            # Nhật ký 4 quy tắc làm sạch dữ liệu ETL
│   ├── cluster_lift_*.csv          # Bảng chỉ số Lift & Coverage cho Album, Artist, Genre
│   ├── cluster_profiles_summary.json# Toàn bộ hồ sơ 5 cụm và kiểm định phi tham số
│   ├── customer_clusters.csv       # Kết quả phân cụm chi tiết 59 khách hàng
│   ├── internal_quality_comparison.csv # So sánh chất lượng nội tại Top 5 ứng viên
│   ├── model_comparison.csv        # Kết quả 1.184 cấu hình duyệt qua 16 pipeline
│   ├── null_model_results.csv      # Phân phối Null Model qua 200 lượt hoán vị Curveball
│   └── stability_summary.csv       # Tóm tắt độ ổn định Hold-out qua 100 lần lặp
├── reports/                        # Báo cáo chuyên sâu theo từng giai đoạn
│   ├── business_problem.md         # Xác định bài toán kinh doanh (Bước 4)
│   ├── cluster_profiles.md         # Báo cáo hồ sơ phân khúc & khuyến nghị kinh doanh (Bước 9)
│   ├── data_quality.md             # Đánh giá 38 tiêu chí chất lượng dữ liệu (Bước 1)
│   ├── eda_findings.md             # 10 phát hiện kinh doanh cốt lõi từ EDA (Bước 3)
│   ├── evaluation.md               # Báo cáo đánh giá & kiểm định tổng quát hóa (Bước 8)
│   ├── final_checklist.md          # Bảng checklist nghiệm thu 10 bước đề bài
│   ├── final_report.md             # Báo cáo tổng kết toàn diện dự án (Pha D)
│   ├── model_card.md               # Thẻ mô hình học máy theo chuẩn kỹ thuật (Model Card)
│   ├── model_comparison.md         # Báo cáo so sánh và tuyển chọn mô hình (Bước 7)
│   ├── problem_selection.md        # Nghiên cứu định lượng chọn bài toán (Bước 5)
│   └── reproducibility_check.md    # Nhật ký kiểm định tái lập sạch từ đầu
├── src/                            # Toàn bộ mã nguồn thực thi dự án
│   ├── etl/                        # Mô đun tầng trích xuất & làm sạch
│   │   ├── extract.py              # Đọc SQLite chế độ read-only
│   │   ├── transform.py            # Áp dụng quy tắc làm sạch & tạo Star-Schema
│   │   ├── load.py                 # Lưu bảng Parquet & ghi manifest
│   │   └── run_etl.py              # Script chạy tự động một lệnh tầng ETL
│   ├── features.py                 # Xây dựng ma trận tương tác (Khách x Album/Track/Genre)
│   ├── pipeline.py                 # Pipeline tiền xử lý, lọc nhiễu, IDF và TruncatedSVD
│   ├── evaluation.py               # Đánh giá nội tại, Hold-out CV, Seed & Temporal test
│   ├── profiling.py                # Tính Lift, Coverage và kiểm định phi tham số
│   └── predict.py                  # Module suy luận phân cụm cho khách hàng mới
├── tests/                          # Bộ kiểm thử tự động toàn diện (Pytest)
│   ├── test_etl.py                 # Kiểm thử tính toàn vẹn và lũy kế tầng ETL
│   ├── test_data_validation.py     # Kiểm thử schema, chống rò rỉ dữ liệu và phân phối
│   ├── test_pipeline.py            # Kiểm thử tính cô lập fit/transform và thành phần pipeline
│   └── test_reproducibility.py     # Kiểm thử tái lập 100% nhãn cụm và module predict
├── requirements.txt                # Danh mục các thư viện phụ thuộc
└── README.md                       # Tài liệu hướng dẫn sử dụng và giới thiệu dự án
```

---

## 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY TỪNG PHA (HOW TO RUN)

### 3.1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
# Tạo môi trường ảo (tùy chọn)
python -m venv venv
venv\Scripts\activate      # Trên Windows
# source venv/bin/activate # Trên Linux/macOS

# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

### 3.2. Chạy từng pha bằng dòng lệnh độc lập

#### Bước 1 & 2: Chạy tầng Analytical ETL
Trích xuất từ SQLite gốc, áp dụng 4 quy tắc làm sạch và lưu các bảng Parquet:
```bash
python src/etl/run_etl.py
```

#### Bước 6: Chạy tầng Feature Engineering & Pipeline
Xây dựng 6 biến thể ma trận đặc trưng và nén không gian tiềm ẩn:
```bash
python -m src.pipeline
```

#### Bước 7 & 8: Chạy Đánh giá mô hình & Lưu trữ Artifact tối ưu
Thực thi kiểm tra chất lượng nội tại, 100 lần Hold-out CV, kiểm tra độ nhạy Seed, kiểm định theo thời gian và lưu file `models/final_model.joblib`:
```bash
python -m src.evaluation
```

#### Bước 9: Chạy phân tích hồ sơ phân khúc & tính Lift
Tính toán Lift/Coverage $\ge 3$ khách, gán Persona và chạy kiểm định Kruskal-Wallis:
```bash
python -m src.profiling
```

---

## 4. CHẠY BỘ KIỂM THỬ TỰ ĐỘNG (RUNNING TESTS)
Dự án được bảo vệ bởi **24 bài kiểm thử tự động** bao phủ 100% các khâu kỹ thuật quan trọng:
```bash
# Chạy toàn bộ test suite với log chi tiết
python -m pytest tests/test_etl.py tests/test_data_validation.py tests/test_pipeline.py tests/test_reproducibility.py -v
```

**Nội dung kiểm tra:**
- `tests/test_etl.py`: Đối chiếu số dòng (2.240), doanh thu (2.328,60 USD), không mất khách hàng, gán cờ bài trùng, điền Composer, tính lũy kế (idempotency).
- `tests/test_data_validation.py`: Kiểm tra schema, không có NaN/Inf, chống rò rỉ dữ liệu nhân khẩu học, độ dài L2 = 1.0.
- `tests/test_pipeline.py`: Kiểm tra ItemFilter, IDFTransformer, ComponentDropper, tính cô lập fit trên tập train khi Cross-Validation.
- `tests/test_reproducibility.py`: Xác thực cấu trúc file joblib, kiểm tra predict lại 59 khách hàng khớp 100% nhãn cụm (ARI = 1.0), kiểm tra hàm `assign_cluster`.

---

## 5. MÔ TẢ CÁC TẬP TIN ĐẦU RA CHÍNH (OUTPUT FILES)

| Tên tập tin | Định dạng | Mục đích & Nội dung |
| :--- | :---: | :--- |
| `models/final_model.joblib` | Binary (Joblib) | Chứa toàn bộ Pipeline tiền xử lý, SVD, KMeans model, unit-normalized centroids, từ vựng 347 album, MD5 data hash và phiên bản thư viện. |
| `outputs/customer_clusters.csv` | CSV | Bảng gán cụm cho 59 khách hàng hiện hữu gồm: `CustomerId`, `cluster`, `persona_name`, `confidence`, `distance_to_centroid`, chi tiêu và quốc gia. |
| `outputs/cluster_profiles_summary.json` | JSON | Cấu trúc đầy đủ về đặc trưng 5 cụm: Top Lift genres/albums/artists, tỷ lệ độ phủ và kết quả các kiểm định phi tham số. |
| `outputs/cluster_lift_genres.csv` | CSV | Bảng chỉ số Lift và Coverage của 25 thể loại âm nhạc theo từng cụm (đã lọc $\ge 3$ khách). |
| `outputs/cluster_lift_albums.csv` | CSV | Bảng chỉ số Lift và Coverage của 347 album theo từng cụm (đã lọc $\ge 3$ khách). |
| `outputs/stability_summary.csv` | CSV | Tóm tắt độ ổn định Hold-out qua 100 lượt Cross-Validation của Top 5 ứng viên mô hình. |
| `reports/final_report.md` | Markdown | Báo cáo khoa học tổng kết toàn diện toàn bộ 10 bước của dự án với đầy đủ dẫn nguồn. |

---

## 6. HƯỚNG DẪN SỬ DỤNG MODULE SUY LUẬN (`src/predict.py`)
Module [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py) cung cấp giao diện suy luận độc lập (Inference Engine) cho phép gán cụm cho khách hàng mới mà không cần cài đặt lại pipeline.

### 6.1. Dự đoán cho một khách hàng mới (`assign_cluster`)
Hàm `assign_cluster(purchases)` chấp nhận đầu vào linh hoạt: danh sách mã album (`list[int]`), mảng nhị phân (`np.ndarray`), hoặc từ điển đếm `{AlbumId: count}`:

```python
from src.predict import assign_cluster

# Ví dụ 1: Khách hàng mới mua 3 album cụ thể (Rock/Metal & Funk)
sample_purchases = [1, 4, 120]  # AlbumId: 1 (For Those About To Rock), 4 (Let There Be Rock), ...

result = assign_cluster(sample_purchases)

print("Mã cụm:", result["cluster_id"])
print("Tên phân khúc:", result["cluster_name"])
print("Độ tin cậy:", result["confidence"])
print("Khoảng cách tới tâm cụm:", result["distance_to_centroid"])
print("Số album hợp lệ trong từ vựng:", result["matched_albums_count"])
```

**Định dạng kết quả trả về (Dict):**
```python
{
    'cluster_id': 4,
    'cluster_name': 'Cluster 4: Classic Blues-Rock, Punk & Symphonic Fans',
    'confidence': 0.7642,
    'distance_to_centroid': 0.9125,
    'matched_albums_count': 3
}
```

### 6.2. Dự đoán hàng loạt theo bảng dữ liệu (`assign_clusters_batch`)
Dùng khi cần phân cụm định kỳ cho một tập khách hàng mới từ ma trận đếm:

```python
import pandas as pd
from src.predict import assign_clusters_batch

# Giả sử new_customers_df là DataFrame có kích thước (M khách hàng, 347 album)
# với index là CustomerId và tên cột là AlbumId
pred_df = assign_clusters_batch(new_customers_df, model_path="models/final_model.joblib")

print(pred_df[["cluster", "persona_name", "confidence", "distance_to_centroid"]].head())
```

---

## 7. CẢNH BÁO SỬ DỤNG NGHIỆP VỤ (BUSINESS GUARDRAILS)
1. **Không dùng để phân hạng VIP/Chi tiêu:** Kiểm định Kruskal-Wallis khẳng định chi tiêu giữa các cụm không có sự khác biệt có ý nghĩa thống kê ($p = 0.3057$). Phân cụm này phản ánh **Gu âm nhạc**, hoàn toàn không phản ánh giá trị tài chính.
2. **Không áp dụng A/B Testing tiêu chuẩn:** Cỡ mẫu 59 khách hàng không đủ công suất thống kê. Doanh nghiệp nên triển khai thử nghiệm Pilot định tính trên nhóm nhỏ thay vì chạy A/B test tự động.
3. **Hiện tượng Cold-Start:** Với khách hàng mới chỉ mua $< 2$ album, độ tin cậy gán cụm sẽ thấp ($< 0.40$). Cần thu thập thêm lịch sử nghe/mua trước khi gán cụm cố định.
