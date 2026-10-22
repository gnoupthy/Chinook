# BÁO CÁO PHÁT HIỆN KHÁM PHÁ DỮ LIỆU (EDA FINDINGS REPORT)

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Nguồn dữ liệu:** `data/Chinook_Sqlite.sqlite` (Giai đoạn 2009 - 2013)  
**Nhà khoa học dữ liệu phụ trách:** Senior Data Scientist  

---

> [!IMPORTANT]
> **Định nghĩa cốt lõi về "Sự quan tâm" (Customer Interest):**  
> Dữ liệu hệ thống chỉ ghi nhận hành vi **mua hàng thực tế** (transactions), hoàn toàn không có logs về lượt nghe (streams), lượt click, thời lượng nghe hay lượt thêm vào giỏ.  
> Do đó, **mức độ quan tâm của khách hàng tới Track/Album được định nghĩa duy nhất qua hành vi mua**:
> 1. Có mua hay không mua (nhị phân 0/1);
> 2. Số lượng bài hát đã mua trong từng Album;
> 3. Tỷ lệ phủ của khách hàng đối với Album đó (Album Coverage Ratio).

---

## 1. Tóm tắt 10 Phát hiện Quan trọng nhất (Key Findings)

### Phát hiện 1: Khách hàng đồng nhất tuyệt đối về quy mô hành vi (Hệ số biến thiên CV siêu nhỏ)
- **Số liệu thực tế:** 
  - Số lượng track mua mỗi khách: Trung bình = **37.97**, Độ lệch chuẩn = **0.26**, Min = **36**, Max = **38**. **Hệ số biến thiên CV = 0.69%**.
  - Số lượng hóa đơn mỗi khách: Trung bình = **6.98**, Độ lệch chuẩn = **0.13**, Min = **6**, Max = **7**. **Hệ số biến thiên CV = 1.86%**.
  - Tổng chi tiêu mỗi khách: Trung bình = **39.47 USD**, Độ lệch chuẩn = **2.91 USD**, Min = **36.64 USD**, Max = **49.62 USD**. **Hệ số biến thiên CV = 7.38%**.
- **Biểu đồ minh chứng:** `figures/eda_customer_behavior_variance.png`
- **Ý nghĩa:** Trong cơ sở dữ liệu Chinook, dữ liệu hành vi mang tính chất mô phỏng nhân tạo (synthetic/template): mỗi khách hàng được phân bổ đúng ~7 đơn hàng và mua đúng ~38 bài hát độc nhất (không mua lặp lại).

### Phát hiện 2: Câu trả lời khẳng định: Khách hàng KHÔNG KHÁC NHAU về hành vi tổng quát mà CHỈ KHÁC NHAU ở NỘI DUNG NHẠC
- **Kết luận trực diện:** Sự phân hóa giữa 59 khách hàng **hoàn toàn không nằm ở số lượng đơn, số lượng bài, hay tổng số tiền chi trả** (tất cả đều xấp xỉ ngang nhau). 
- **Cốt lõi:** Điểm khác biệt duy nhất để phân cụm khách hàng là **Gu âm nhạc / Nội dung cụ thể (Taste & Content Preference)**: họ chọn mua bài hát nào, album của ai, thuộc dòng nhạc gì.
- **Hàm ý gom cụm:** Loại bỏ hoàn toàn ý tưởng phân cụm theo mô hình RFM (Recency, Frequency, Monetary). Bài toán bắt buộc phải là **Content-based / Collaborative Clustering**.

### Phát hiện 3: Ma trận Khách × Track cực kỳ thưa (Extreme Sparsity ~98.92%)
- **Số liệu thực tế:**
  - Tổng số ô trong ma trận: $59 \times 3.503 = 206.677$ ô.
  - Số ô có giao dịch: đúng **2.240** ô.
  - **Độ đậm đặc (Density): 1.084%** $\rightarrow$ **Độ thưa (Sparsity): 98.916%**.
  - Ngay cả khi chỉ xét trên 1.984 track đã từng được bán: Density cũng chỉ đạt **1.914%** (Sparsity = 98.086%).
- **Biểu đồ minh chứng:** `figures/eda_density_heatmaps.png` (Hình 4A).

### Phát hiện 4: Phân phối Long-tail cực đoan ở cấp độ Track: 87.1% track chỉ có duy nhất 1 người mua
- **Số liệu thực tế:**
  - Trong 1.984 bài hát đã bán, có tới **1.728 bài (87.10%) chỉ có ĐÚNG 1 khách hàng mua**.
  - Chỉ có **256 bài (12.90%) có 2 khách hàng cùng mua**.
  - **Số khách mua tối đa cho 1 bài hát trong toàn hệ thống là 2 khách** (không có bất kỳ bài nào có $\ge 3$ khách mua).
- **Biểu đồ minh chứng:** `figures/eda_coverage_longtail.png` (Hình 3A).
- **Hàm ý gom cụm:** Đây là thách thức toán học lớn nhất! Hai khách hàng bất kỳ hầu như không có bài hát chung. Nếu dùng vector nhị phân 3.503 chiều để tính khoảng cách Euclidean/Cosine thô, khoảng cách giữa các cặp khách hàng sẽ gần như trực giao (bằng nhau), dẫn đến phân cụm thất bại (Curse of Dimensionality).

### Phát hiện 5: Cấp độ Album tạo ra tín hiệu chồng lấn (Overlap) mạnh gấp 6 lần Track
- **Số liệu thực tế:**
  - Độ phủ Album: **304 / 347 album (87.61%)** trong catalog đã phát sinh giao dịch.
  - **265 / 304 album (87.17%) có từ 2 khách hàng cùng mua trở lên**.
  - Album bán chạy nhất có tới **15 khách hàng cùng mua**.
  - Độ đậm đặc ma trận Khách × Album: **6.355%** (toàn bộ) và **7.254%** (trên tập bán) — cao gấp 6 lần so với Track.
- **Biểu đồ minh chứng:** `figures/eda_coverage_longtail.png` (Hình 3B) và `figures/eda_density_heatmaps.png` (Hình 4B).
- **Hàm ý gom cụm:** Không gian Album là điểm tựa vững chắc hơn nhiều so với Track để gom cụm khách hàng, hoặc cần ánh xạ Track lên biểu diễn ẩn (Latent Semantic Space / SVD) để tận dụng thông tin Album/Genre.

### Phát hiện 6: Khách hàng mua bài lẻ (Cherry-picking), tỷ lệ phủ trọn vẹn Album rất thấp
- **Số liệu thực tế:**
  - Tỷ lệ phủ album trung bình trong một lần tương tác: **16.79%** số bài của album.
  - Số lần khách mua toàn bộ 100% bài hát của một album chỉ chiếm **49 / 1.301 cặp (Khách, Album)**, tương đương **3.77%**.
  - Trung bình mỗi khách mua rải rác ở **22.05 album khác nhau** (Std = 2.19, Min = 18, Max = 27).
- **Ý nghĩa:** Khách hàng không mua cả đĩa nhạc nguyên vẹn mà chọn lọc từng bài hát yêu thích. Do đó, đặc trưng "độ phủ album" (tỷ lệ bài đã mua / tổng số bài trong album) là một trọng số thực sự có giá trị để đánh giá mức độ quan tâm sâu hay cạn của khách đối với album đó.

### Phát hiện 7: Cơ cấu doanh thu bị chi phối mạnh mẽ bởi Rock, Latin và Metal
- **Số liệu thực tế:**
  - **Rock** dẫn đầu tuyệt đối với **826.65 USD** (35.5% tổng doanh thu), theo sau là **Latin (382.14 USD)** và **Metal (261.36 USD)**. Ba thể loại này chiếm trên **63%** doanh thu toàn cửa hàng.
  - Các dòng nhạc chuyên biệt (Jazz, Blues, TV Shows, Classical) có doanh số thấp hơn đáng kể.
  - Top nghệ sĩ bán chạy nhất: **Iron Maiden (138.60 USD)**, **U2 (105.93 USD)**, **Metallica (90.09 USD)**.
- **Biểu đồ minh chứng:** `figures/eda_revenue_trends.png` (Hình 1C, 1D).

### Phát hiện 8: Ma trận Khách × Genre có độ đậm đặc cao (29.83%), là tín hiệu neo giải thích (Interpretability Anchor)
- **Số liệu thực tế:**
  - Với 25 thể loại âm nhạc, ma trận Khách × Genre có mật độ lên tới **29.83%**.
  - Trung bình một khách nghe **7.8 thể loại khác nhau**.
- **Ý nghĩa:** Mặc dù không gian Track quá thưa, nhưng không gian Genre lại rất đậm đặc. Các cụm khách hàng khi hình thành ở cấp độ Track/Album hoàn toàn có thể được giải thích, đặt tên và kiểm chứng tính hợp lý thông qua hồ sơ phân bổ thể loại (Genre Profile) của từng cụm.

### Phát hiện 9: Xu hướng đồng xuất hiện (Co-occurrence) giữa các dòng nhạc
- **Số liệu thực tế:**
  - Cặp thể loại được mua cùng nhau nhiều nhất: **Rock & Latin** (34 khách cùng mua), **Rock & Metal** (31 khách cùng mua), **Rock & Alternative & Punk** (26 khách).
  - Ở cấp độ nghệ sĩ: Nhóm khách thích Iron Maiden có xác suất rất cao mua thêm Metallica hoặc Led Zeppelin.
- **Biểu đồ minh chứng:** `figures/eda_genre_artist_cooccurrence.png` (Hình 5A, 5B).
- **Hàm ý:** Tồn tại những cụm thị hiếu tự nhiên (ví dụ: Heavy Metal/Hard Rock group, Eclectic/World Music group).

### Phát hiện 10: Loại bỏ hoàn toàn Playlist / PlaylistTrack khỏi tín hiệu gom cụm
- **Khảo sát:** Bảng `Playlist` có 18 danh sách phát, bảng `PlaylistTrack` có 8.715 liên kết giữa playlist và bài hát.
- **Lý do loại bỏ:** Các playlist này là danh mục biên tập tĩnh do ban quản trị cửa hàng tạo sẵn (như "Music", "TV Shows", "90's Music", "Brazilian Music"), **hoàn toàn không có khóa ngoại `CustomerId`**. Khách hàng không tạo playlist riêng và không tương tác với bảng này. Việc đưa Playlist vào gom cụm khách hàng sẽ gây sai lệch dữ liệu nghiêm trọng.

---

## 2. Hàm ý Chiến lược cho Bài toán Gom Cụm Khách Hàng (Phase B)

Dựa trên các phát hiện định lượng trên, hướng đi kỹ thuật cho các pha tiếp theo được xác lập như sau:

| Thách thức quan sát được | Chiến lược giải quyết trong Feature Engineering & Clustering |
| :--- | :--- |
| **Ma trận Track quá thưa (98.92%, 87% bài chỉ 1 người mua)** | Không áp dụng K-Means trực tiếp trên ma trận Track nhị phân 3.503 chiều. Cần áp dụng kỹ thuật giảm chiều **TruncatedSVD / NMF** để trích xuất không gian tiềm ẩn (Latent Semantic Space). |
| **Ma trận Album có độ phủ và overlap cao gấp 6 lần** | Xây dựng đặc trưng ở cấp độ **Customer-Album**: kết hợp giữa số bài đã mua và tỷ lệ phủ album (`coverage_ratio`). Thử nghiệm mô hình gom cụm trực tiếp trên không gian Album. |
| **Khách hàng đồng nhất về số lượng bài/tiền (CV < 8%)** | Chuẩn hóa vector đặc trưng theo độ dài L2 (Normalized / TF-IDF weights) hoặc Cosine Distance để thuật toán tập trung 100% vào **tỷ trọng phân bổ gu âm nhạc**, triệt tiêu sai số quy mô. |
| **Số lượng khách hàng nhỏ ($N=59$)** | Với cỡ mẫu 59 điểm dữ liệu, số cụm tối ưu kỳ vọng sẽ nằm trong khoảng $K \in [3, 6]$. Cần sử dụng kết hợp **K-Means**, **Hierarchical Clustering (Agglomerative)** và đối chiếu với **Genre Profile** để nghiệm thu cụm có ý nghĩa thực tế. |

---

## 3. Danh mục hình ảnh đính kèm
1. [figures/erd.png](file:///e:/ML/Chinook/figures/erd.png) - Sơ đồ thực thể liên kết chuẩn của CSDL Chinook.
2. [figures/eda_revenue_trends.png](file:///e:/ML/Chinook/figures/eda_revenue_trends.png) - Doanh thu theo thời gian, quốc gia, thể loại và nghệ sĩ.
3. [figures/eda_customer_behavior_variance.png](file:///e:/ML/Chinook/figures/eda_customer_behavior_variance.png) - Phân tích phương sai hành vi và hệ số biến thiên CV giữa các khách hàng.
4. [figures/eda_coverage_longtail.png](file:///e:/ML/Chinook/figures/eda_coverage_longtail.png) - So sánh phân phối Long-tail và độ chồng lấn: Track vs Album.
5. [figures/eda_density_heatmaps.png](file:///e:/ML/Chinook/figures/eda_density_heatmaps.png) - Trực quan hóa độ thưa của các ma trận tương tác.
6. [figures/eda_genre_artist_cooccurrence.png](file:///e:/ML/Chinook/figures/eda_genre_artist_cooccurrence.png) - Ma trận đồng xuất hiện giữa các thể loại và nghệ sĩ.
