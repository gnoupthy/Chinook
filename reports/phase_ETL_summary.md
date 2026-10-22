# BÁO CÁO TỔNG KẾT PHA ETL: XÂY TẦNG DỮ LIỆU PHÂN TÍCH
**(PHASE ETL SUMMARY & HANDOFF REPORT)**

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Giai đoạn hoàn thành:** PHA ETL (Extract - Transform - Load & Data Quality Enforcement)  
**Ngày thực hiện:** 2026-10-22  
**Tài liệu thiết kế chi tiết:** [reports/etl_design.md](file:///e:/ML/Chinook/reports/etl_design.md)  
**Tài liệu tham chiếu:** [reports/data_quality.md](file:///e:/ML/Chinook/reports/data_quality.md) & [reports/problem_selection.md](file:///e:/ML/Chinook/reports/problem_selection.md)  
**Trạng thái:** **HOÀN THÀNH 100% - DỪNG CHỜ DUYỆT TRƯỚC KHI SANG PHA C1**  

---

## 1. Tóm Tắt Kết Quả Triển Khai (Executive Summary)

Pha ETL đã xây dựng thành công tầng dữ liệu phân tích chuẩn hóa dạng hình sao (Star Schema), chuyển hóa toàn bộ các quyết định nghiệp vụ và làm sạch từ [reports/data_quality.md](file:///e:/ML/Chinook/reports/data_quality.md) thành code Python tự động, có cấu hình bật/tắt minh bạch và kiểm thử nghiêm ngặt:

1. **Một lệnh thực thi toàn trình (Single-Command Reproducibility):**
   ```bash
   python src/etl/run_etl.py
   ```
   Pipeline tự động đọc SQLite chế độ Read-Only, áp dụng quy tắc làm sạch, dựng schema sao, kiểm tra tính lũy đẳng (Idempotency), lưu trữ định dạng Parquet nén `snappy`, và sinh manifest có kèm SHA-256 hash.
2. **Tuân thủ nguyên tắc không lọc ngầm (Transparent Rules):**
   - Không có bất kỳ dòng nào bị xóa bỏ âm thầm. 100% các thao tác lọc đều có switch trong [config/etl.yaml](file:///e:/ML/Chinook/config/etl.yaml) và được ghi log chi tiết số dòng, số khách bị ảnh hưởng.
3. **Bảo tồn trọn vẹn 12 track trùng lặp logic:**
   - 6 cặp track có cùng `(Name, AlbumId)` được giữ nguyên 100% trong catalog; gắn cờ kỹ thuật `is_duplicate_name_album = True` đúng theo kết luận Data Quality.
4. **Kiểm thử tự động xanh tuyệt đối (Pytest 7/7 PASSED):**
   - Toàn bộ 7 bài kiểm thử đối soát số lượng, kiểm tra khóa chính/ngoại, doanh thu, tính lũy đẳng và manifest đều vượt qua trong 3.05 giây.

---

## 2. Bảng Đối Chiếu Số Liệu EDA Trước và Sau ETL (Reconciliation Table)

Dưới đây là bảng đối chiếu chi tiết giữa các chỉ số thống kê đã công bố ở Pha A (EDA / Data Quality) và Pha B (Problem Selection) so với dữ liệu thực tế trích xuất từ tầng phân tích sạch `data/processed/*.parquet`:

| STT | Chỉ Số Thống Kê / Nghiệp Vụ | Nguồn Trước ETL (SQLite Thô / EDA / Pha B) | Kết Quả Sau ETL (`data/processed/*.parquet`) | Chênh Lệch | Trạng Thái Đối Soát | Giải Thích Bản Chất Kỹ Thuật |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Số lượng khách hàng (`CustomerId`)** | 59 khách hàng | 59 khách hàng (`dim_customer`) | **0** | **Khớp 100%** | Toàn bộ 59/59 khách hàng được bảo tồn, không mất khách nào. |
| 2 | **Số lượng hóa đơn (`InvoiceId`)** | 412 hóa đơn | 412 hóa đơn (`fact_purchase`) | **0** | **Khớp 100%** | Đủ 412 hóa đơn giao dịch lịch sử. |
| 3 | **Tổng số dòng giao dịch mua bài hát** | 2.240 dòng (`InvoiceLine`) | 2.240 dòng (`fact_purchase`) | **0** | **Khớp 100%** | Mỗi dòng là 1 cặp `(CustomerId, TrackId)` duy nhất. |
| 4 | **Tổng doanh thu lịch sử toàn hệ thống** | **$2.328,60** (`Invoice.Total`) | **$2.328,60** (`fact_purchase`) | **$0,00** | **Khớp 100%** | Khớp từng xu với sổ sách tài chính. |
| 5 | **Số lượng bài hát catalog (`dim_track`)** | 3.503 bài hát | 3.503 bài hát | **0** | **Khớp 100%** | Giữ toàn bộ catalog, không xóa track nào. |
| 6 | **Số lượng bài hát đã phát sinh giao dịch** | 1.984 bài hát (56.64%) | 1.984 bài hát (`is_sold=True`) | **0** | **Khớp 100%** | Nhận diện chính xác qua cờ `is_sold`. |
| 7 | **Số lượng Album catalog (`dim_album`)** | 347 album | 347 album | **0** | **Khớp 100%** | Đủ 347 album với trường `total_tracks`. |
| 8 | **Số lượng Album đã bán (`is_sold=True`)** | 304 album (87.61%) | 304 album (`is_sold=True`) | **0** | **Khớp 100%** | 43 album chưa từng bán được giữ nguyên. |
| 9 | **Số lượng Artist catalog (`dim_artist`)** | 275 nghệ sĩ | 275 nghệ sĩ | **0** | **Khớp 100%** | Đủ 275 nghệ sĩ với `total_albums`, `total_tracks`. |
| 10 | **Số lượng Artist đã bán (`is_sold=True`)** | 165 nghệ sĩ (60.00%) | 165 nghệ sĩ (`is_sold=True`) | **0** | **Khớp 100%** | Khớp tuyệt đối với số liệu Pha B. |
| 11 | **Số lượng Thể loại catalog (`dim_genre`)** | 25 thể loại | 25 thể loại | **0** | **Khớp 100%** | Đủ 25 thể loại. |
| 12 | **Số lượng Thể loại đã bán (`is_sold=True`)** | 24 thể loại (96.00%) | 24 thể loại (`is_sold=True`) | **0** | **Khớp 100%** | Thể loại *Opera* (`GenreId=20`) chưa từng bán (`is_sold=False`). |
| 13 | **Độ đậm đặc Khách × Track (Density Sold)** | **1.914%** (Pha B) | **1.914%** ($\frac{2240}{59 \times 1984}$) | **0.000%** | **Khớp 100%** | Độ thưa 98.086% không đổi. |
| 14 | **Độ đậm đặc Khách × Album (Density Sold)** | **7.254%** (Pha B) | **7.254%** ($\frac{1301}{59 \times 304}$) | **0.000%** | **Khớp 100%** | Độ thưa 92.746% không đổi. |
| 15 | **Tín hiệu trung bình mua / khách hàng** | 37.97 ± 0.26 bài | 37.97 ± 0.26 bài | **0.00** | **Khớp 100%** | Khoảng biến thiên [36, 38] bài/khách hàng. |
| 16 | **Track trùng tên logic (Name, AlbumId)** | 12 tracks (6 cặp) | 12 tracks mang cờ `is_duplicate` | **0** | **Khớp 100%** | Không bị xóa; gắn cờ chính xác 6 cặp. |
| 17 | **Track khuyết Composer (NULL -> 'Unknown')** | 978 tracks khuyết | 978 tracks điền `'Unknown'` | **0** | **Khớp 100%** | 0 giá trị NULL trong `dim_track.Composer`. |

---

## 3. Kết Quả Kiểm Thử Tự Động (Test Execution Results)

Lệnh kiểm thử đã chạy:
```bash
python -m pytest -v tests/test_etl.py
```

Kết quả thực tế từ hệ thống:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.1, pytest-9.1.1, pluggy-1.7.0 -- C:\Users\admin\AppData\Local\Programs\Python\Python313\python.exe
cachedir: .pytest_cache
rootdir: E:\ML\Chinook
plugins: anyio-4.13.0
collecting ... collected 7 items

tests/test_etl.py::test_reconciliation_total_rows_and_revenue PASSED     [ 14%]
tests/test_etl.py::test_keys_uniqueness_and_referential_integrity PASSED [ 28%]
tests/test_etl.py::test_no_customer_loss_without_logging PASSED          [ 42%]
tests/test_etl.py::test_flagged_duplicate_tracks PASSED                  [ 57%]
tests/test_etl.py::test_imputed_composer PASSED                          [ 71%]
tests/test_etl.py::test_idempotency_run_twice PASSED                     [ 85%]
tests/test_etl.py::test_manifest_schema_and_metadata PASSED              [100%]

============================== 7 passed in 3.05s ==============================
```

---

## 4. Danh Mục Hiện Vật Bàn Giao (Deliverables Manifest)

Toàn bộ các tệp phục vụ Pha ETL đã được khởi tạo và lưu trữ đầy đủ trong dự án:

1. **Mã nguồn ETL (`src/etl/*`):**
   - [src/etl/__init__.py](file:///e:/ML/Chinook/src/etl/__init__.py): Khởi tạo module ETL package.
   - [src/etl/extract.py](file:///e:/ML/Chinook/src/etl/extract.py): Trích xuất SQLite ở chế độ Read-Only (`uri=True`, `mode=ro`), ghi nhận số dòng/cột.
   - [src/etl/transform.py](file:///e:/ML/Chinook/src/etl/transform.py): Các hàm làm sạch module hóa (`flag_duplicate_tracks`, `fill_null_composer`, hàm lọc an toàn, dựng `fact_purchase` và các `dim_*`).
   - [src/etl/load.py](file:///e:/ML/Chinook/src/etl/load.py): Ghi bảng Parquet, tính SHA-256 hash (nội dung & tệp), sinh `manifest.json` và audit logs.
   - [src/etl/run_etl.py](file:///e:/ML/Chinook/src/etl/run_etl.py): Bộ điều phối trung tâm toàn trình (Orchestrator CLI).
2. **Cấu hình & Nhật ký:**
   - [config/etl.yaml](file:///e:/ML/Chinook/config/etl.yaml): Cấu hình đường dẫn, định dạng nén Parquet, công tắc bật/tắt toàn bộ các quy tắc lọc.
   - [logs/etl_run.log](file:///e:/ML/Chinook/logs/etl_run.log): Log chi tiết quá trình chạy pipeline.
   - [data/processed/manifest.json](file:///e:/ML/Chinook/data/processed/manifest.json): Metadata phiên bản, hash SHA-256 từng bảng, thời điểm chạy.
   - [data/processed/etl_audit_log.csv](file:///e:/ML/Chinook/data/processed/etl_audit_log.csv) & [json](file:///e:/ML/Chinook/data/processed/etl_audit_log.json): Bảng kiểm thẩm các quy tắc làm sạch.
3. **Tầng dữ liệu phân tích sạch (`data/processed/*.parquet`):**
   - [data/processed/fact_purchase.parquet](file:///e:/ML/Chinook/data/processed/fact_purchase.parquet): 2.240 dòng, 8 cột.
   - [data/processed/dim_customer.parquet](file:///e:/ML/Chinook/data/processed/dim_customer.parquet): 59 dòng, 13 cột.
   - [data/processed/dim_track.parquet](file:///e:/ML/Chinook/data/processed/dim_track.parquet): 3.503 dòng, 11 cột.
   - [data/processed/dim_album.parquet](file:///e:/ML/Chinook/data/processed/dim_album.parquet): 347 dòng, 5 cột.
   - [data/processed/dim_artist.parquet](file:///e:/ML/Chinook/data/processed/dim_artist.parquet): 275 dòng, 5 cột.
   - [data/processed/dim_genre.parquet](file:///e:/ML/Chinook/data/processed/dim_genre.parquet): 25 dòng, 4 cột.
4. **Kiểm thử chất lượng:**
   - [tests/test_etl.py](file:///e:/ML/Chinook/tests/test_etl.py): Bộ test suite tự động kiểm tra 7 khía cạnh then chốt.
5. **Báo cáo kỹ thuật:**
   - [reports/etl_design.md](file:///e:/ML/Chinook/reports/etl_design.md): Tài liệu đặc tả kiến trúc, sơ đồ luồng Mermaid và Star Schema.
   - [reports/phase_ETL_summary.md](file:///e:/ML/Chinook/reports/phase_ETL_summary.md): Tài liệu tổng kết và nghiệm thu hiện tại.

---

## 5. Kết Luận & Điểm Dừng Phê Duyệt (Stop & Await Approval)

> [!IMPORTANT]
> **CAM KẾT THỰC HIỆN ĐÚNG PHẠM VI:**  
> - Tầng dữ liệu phân tích sạch đã sẵn sàng 100% với độ tin cậy tuyệt đối.
> - **CHƯA THỰC HIỆN BẤT KỲ BƯỚC NÀO CỦA FEATURE ENGINEERING HAY HUẤN LUYỆN MÔ HÌNH.**
> - Pipeline hiện tại đã dừng lại và chờ User nghiệm thu toàn bộ kết quả của Pha ETL trước khi bắt đầu chuyển sang Pha C1 (Feature Engineering & Pipeline Validation).
