# BÁO CÁO TỔNG KẾT PHA A: HIỂU DỮ LIỆU -> ĐÁNH GIÁ CHẤT LƯỢNG -> EDA

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Thời gian hoàn thành:** 2026-10-22  
**Trạng thái:** Hoàn tất toàn bộ Bước 1, Bước 2, Bước 3 — Chờ nghiệm thu  

---

## 1. Những công việc đã thực hiện trong Pha A

1. **Khảo sát cấu trúc & Lập Từ điển dữ liệu (Bước 1):**
   - Đã khảo sát 100% (11/11) bảng trong file CSDL `data/Chinook_Sqlite.sqlite` ở chế độ chỉ đọc (Read-Only).
   - Xác định rõ mức độ chi tiết (Grain), kiểu dữ liệu, ràng buộc khóa chính (PK), khóa ngoại (FK) và đường nối nghiệp vụ:  
     `Customer (59)` $\rightarrow$ `Invoice (412)` $\rightarrow$ `InvoiceLine (2.240)` $\rightarrow$ `Track (3.503)` $\rightarrow$ `Album (347)` $\rightarrow$ `Artist (275)`.
   - Xuất bản tài liệu: [reports/data_dictionary.md](file:///e:/ML/Chinook/reports/data_dictionary.md).
   - Vẽ và xuất bản sơ đồ liên kết thực thể độ phân giải cao: [figures/erd.png](file:///e:/ML/Chinook/figures/erd.png).

2. **Kiểm tra chất lượng dữ liệu & Nhật ký làm sạch (Bước 2):**
   - Thực thi bộ kiểm tra chất lượng 7 hạng mục: Khuyết thiếu (NULLs), trùng khóa chính, toàn vẹn khóa ngoại (mồ côi), đơn giá/số lượng bất thường, khoảng thời gian hóa đơn, tính khớp số liệu tài chính `Invoice.Total == SUM(InvoiceLine)`.
   - Đã khảo sát chi tiết **6 cặp trùng (Name, AlbumId)** trong bảng `Track`: xác định đây là các tệp định dạng/thời lượng khác nhau trong catalog, không phải lỗi duplicate kỹ thuật.
   - Phân loại rõ ràng các cột NULL hợp lệ (`Company`, `Fax`, `State` của `Customer`, `ReportsTo` của `Employee`, `Composer` của `Track`).
   - Khẳng định tính trung thực: Dữ liệu sạch, chuẩn hóa cao (3NF), không tạo vấn đề giả để sửa CSDL gốc.
   - Xuất bản tài liệu: [reports/data_quality.md](file:///e:/ML/Chinook/reports/data_quality.md) và nhật ký [outputs/cleaning_log.csv](file:///e:/ML/Chinook/outputs/cleaning_log.csv).

3. **Khám phá dữ liệu chuyên sâu (EDA) & Trực quan hóa (Bước 3):**
   - Tính toán toàn bộ các chỉ số thống kê, phân phối và hệ số biến thiên (CV) của khách hàng.
   - Đánh giá độ phủ, độ thưa (sparsity) và hiện tượng Long-tail ở cả hai cấp độ Track và Album.
   - Phân tích ma trận đồng xuất hiện giữa các thể loại và nghệ sĩ.
   - Khảo sát bảng `Playlist` / `PlaylistTrack` và giải trình lý do loại bỏ khỏi tín hiệu quan tâm của khách hàng.
   - Xuất bản 5 bộ biểu đồ trực quan hóa chi tiết trong thư mục `figures/`:
     * [figures/eda_revenue_trends.png](file:///e:/ML/Chinook/figures/eda_revenue_trends.png)
     * [figures/eda_customer_behavior_variance.png](file:///e:/ML/Chinook/figures/eda_customer_behavior_variance.png)
     * [figures/eda_coverage_longtail.png](file:///e:/ML/Chinook/figures/eda_coverage_longtail.png)
     * [figures/eda_density_heatmaps.png](file:///e:/ML/Chinook/figures/eda_density_heatmaps.png)
     * [figures/eda_genre_artist_cooccurrence.png](file:///e:/ML/Chinook/figures/eda_genre_artist_cooccurrence.png)
   - Xuất bản báo cáo 10 phát hiện cốt lõi: [reports/eda_findings.md](file:///e:/ML/Chinook/reports/eda_findings.md).

---

## 2. Các phát hiện định lượng chính (Key Quantitative Findings)

| Chỉ số / Đặc trưng | Giá trị tính toán thực tế từ code | Nhận định của Senior Data Scientist |
| :--- | :---: | :--- |
| **Số lượng khách hàng / Hóa đơn / Dòng chi tiết** | 59 khách / 412 đơn / 2.240 dòng | Khớp 100% với khảo sát sơ bộ. |
| **Phân phối Quantity** | Luôn luôn = 1 (100% số dòng) | Mỗi giao dịch là mua quyền sở hữu 1 bản nhạc số độc nhất. |
| **Số track mỗi khách mua** | Mean = 37.97, Min = 36, Max = 38 | **CV = 0.69%** $\rightarrow$ Khách hàng hoàn toàn giống nhau về số lượng track. |
| **Số hóa đơn mỗi khách** | Mean = 6.98, Min = 6, Max = 7 | **CV = 1.86%** $\rightarrow$ Tần suất mua gần như cố định cho mọi khách. |
| **Tổng chi tiêu mỗi khách** | Mean = 39.47 USD, Min = 36.64 USD, Max = 49.62 USD | **CV = 7.38%** $\rightarrow$ Chi tiêu phân bổ cực hẹp. |
| **Kết luận về phương sai hành vi** | **Đồng nhất về quy mô** | **Khách chỉ khác nhau ở NỘI DUNG NHẠC (Gu âm nhạc / Taste), không khác nhau về hành vi.** |
| **Độ thưa ma trận Khách × Track** | **98.916%** (Density = 1.084%) | Quá thưa, 87.10% track chỉ có 1 khách mua, max = 2 khách/track. |
| **Độ thưa ma trận Khách × Album** | **93.645%** (Density = 6.355%) | Đậm đặc gấp 6 lần Track, 87.17% album có $\ge 2$ khách mua, max = 15 khách/album. |
| **Tỷ lệ phủ trọn vẹn Album** | 3.77% (49/1.301 tương tác) | Khách mua lẻ bài hát (cherry-picking), độ phủ album trung bình đạt 16.79%. |
| **Cơ cấu thể loại áp đảo** | Rock (35.5%), Latin (16.4%), Metal (11.2%) | 3 dòng nhạc này chiếm 63.1% doanh thu toàn cửa hàng. |

---

## 3. Vấn đề tồn đọng & Cảnh báo kỹ thuật cho Pha B (Feature Engineering & Clustering)

1. **Rủi ro "Lời nguyền số chiều" (Curse of Dimensionality) ở cấp độ Track:**
   - Với ma trận 59 dòng $\times$ 3.503 cột và độ thưa 98.9%, khoảng cách Euclidean/Cosine giữa các cặp khách hàng sẽ gần như xấp xỉ nhau (hiện tượng distance concentration).
   - **Giải pháp cho Pha B:** Tuyệt đối không gom cụm trực tiếp trên ma trận Track thô. Bắt buộc phải áp dụng giảm chiều (TruncatedSVD / NMF) hoặc gom cụm trên không gian Album có gắn trọng số độ phủ (`coverage_ratio`), sau đó dùng Genre Profile để giải thích.
2. **Cỡ mẫu nhỏ ($N = 59$ khách hàng):**
   - Với 59 quan sát, các thuật toán dựa trên mật độ như DBSCAN có thể coi phần lớn điểm là nhiễu (noise).
   - **Giải pháp cho Pha B:** Ưu tiên **K-Means** (với số cụm $K \in [3, 6]$) và **Agglomerative Hierarchical Clustering** (với liên kết Ward hoặc Complete).

---

## 4. Checklist nghiệm thu Pha A (Acceptance Checklist)

- [x] Đã liệt kê toàn bộ bảng, cột, kiểu dữ liệu, PK, FK, grain ([data_dictionary.md](file:///e:/ML/Chinook/reports/data_dictionary.md)).
- [x] Đã vẽ ERD chuẩn, trực quan, có đường nối rõ ràng ([erd.png](file:///e:/ML/Chinook/figures/erd.png)).
- [x] Đã kiểm tra tính toàn vẹn, xác nhận 0 bản ghi mồ côi, Invoice.Total khớp 100% InvoiceLine ([data_quality.md](file:///e:/ML/Chinook/reports/data_quality.md)).
- [x] Đã giải trình rõ ràng các cột NULL hợp lệ của Customer, Employee, Track.
- [x] Đã phân tích chi tiết 6 cặp trùng (Name, AlbumId) trong Track.
- [x] Đã tạo nhật ký làm sạch dữ liệu chuẩn ([cleaning_log.csv](file:///e:/ML/Chinook/outputs/cleaning_log.csv)).
- [x] Đã thực hiện đầy đủ 4 nhóm phân tích EDA (Doanh thu, Hành vi & CV, Độ phủ & Long-tail, Độ thưa & Density).
- [x] Đã trả lời trực diện: Khách hàng đồng nhất về quy mô hành vi, chỉ khác nhau ở nội dung âm nhạc đã mua.
- [x] Đã phân tích Playlist/PlaylistTrack và giải thích rõ lý do loại bỏ.
- [x] Đã xuất bản 5 biểu đồ EDA chất lượng cao kèm insight trong `figures/`.
- [x] Đã xuất bản báo cáo 10 phát hiện cốt lõi và hàm ý cho gom cụm ([eda_findings.md](file:///e:/ML/Chinook/reports/eda_findings.md)).

---

**DỪNG LẠI CHỜ DUYỆT:**  
Tôi đã dừng lại theo đúng quy tắc làm việc. Xin mời bạn kiểm tra và đánh giá kết quả nghiệm thu Pha A trước khi chúng ta tiến hành Pha B (Trích xuất đặc trưng & Gom cụm)!
