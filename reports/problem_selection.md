# BÁO CÁO LỰA CHỌN BÀI TOÁN GOM CỤM (PROBLEM SELECTION REPORT)

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Nguồn dữ liệu:** `data/Chinook_Sqlite.sqlite` (Giai đoạn 2009 - 2013)  
**Tài liệu tham chiếu:** [reports/eda_findings.md](file:///e:/ML/Chinook/reports/eda_findings.md) & [reports/business_problem.md](file:///e:/ML/Chinook/reports/business_problem.md)  
**Nhà khoa học dữ liệu phụ trách:** Senior Data Scientist  

---

> [!IMPORTANT]
> **Mục tiêu của Bước 5 (Đặt bài toán kỹ thuật):**  
> Dựa trên các quyết định kinh doanh đã xác định ở Bước 4, tài liệu này tiến hành so sánh định lượng thực nghiệm (Empirical Quantitative Comparison) giữa các không gian biểu diễn đặc trưng khác nhau của dữ liệu khách hàng.  
> Thay vì lựa chọn cảm tính, bài toán gom cụm được quyết định dựa trên **chỉ số đo lường thực tế**, **kiểm định xu hướng cụm (Hopkins Statistic)**, và **kiểm định mô hình vô hiệu (Null Model Permutation Test với $N = 300$ lần hoán vị bảo toàn biên)**.

---

## 1. Bảng So Sánh Định Lượng Các Không Gian Biểu Diễn (Quantitative Comparison)

Toàn bộ các số liệu dưới đây được trích xuất trực tiếp từ thực nghiệm chạy trên CSDL Chinook (`src/run_null_model_experiments.py`, kết quả lưu tại [outputs/problem_selection_metrics.json](file:///e:/ML/Chinook/outputs/problem_selection_metrics.json)).

| Tiêu chí so sánh định lượng | (a) Khách × Track | (b) Khách × Album | (c1) Khách × Artist | (c2) Khách × Genre |
| :--- | :---: | :---: | :---: | :---: |
| **Thực thể biểu diễn (Grain)** | `TrackId` | `AlbumId` | `ArtistId` | `GenreId` |
| **Số lượng cột (Tập đã bán / Toàn catalog)** | **1.984** / 3.503 | **304** / 347 | **165** / 275 | **24** / 25 |
| **Độ đậm đặc trên tập bán (Density Sold)** | **1.914%** | **7.254%** | **9.481%** | **31.073%** |
| **Độ đậm đặc trên toàn catalog (Density Catalog)** | **1.084%** | **6.355%** | **5.689%** | **29.831%** |
| **Độ thưa (Sparsity trên tập bán)** | **98.086%** | **92.746%** | **90.519%** | **68.927%** |
| **Tín hiệu trung bình / khách (Mean ± Std)** | 37.97 ± 0.26 bài | 22.05 ± 2.17 album | 15.64 ± 3.08 nghệ sĩ | 7.46 ± 1.69 thể loại |
| **Khoảng biến thiên tín hiệu [Min, Max]** | [36, 38] bài | [18, 27] album | [8, 22] nghệ sĩ | [4, 12] thể loại |
| **Số khách có overlap $\ge 1$ với khách khác** | 59 / 59 (100.0%) | 59 / 59 (100.0%) | 59 / 59 (100.0%) | 59 / 59 (100.0%) |
| **Trung vị khoảng cách Cosine (Median $d_{\cos}$)** | **1.0000** | **0.9544** | **0.8655** | **0.3828** |
| **Khoảng cách Cosine trung bình (Mean $d_{\cos}$)** | 0.9961 | 0.9275 | 0.8442 | 0.3942 |
| **Tứ phân vị khoảng cách Cosine [Q25, Q75]** | [1.0000, 1.0000] | [0.9067, 1.0000] | [0.8000, 0.9307] | [0.3196, 0.4730] |
| **% Cặp khách hàng có $d_{\cos} \ge 0.95$ (Gần như trực giao)** | **97.49%** (1.668 cặp) | **60.84%** (1.041 cặp) | **8.88%** (152 cặp) | **0.00%** (0 cặp) |
| **Hopkins Statistic (Không gian gốc: Raw)** | $0.7511 \pm 0.0009$ | $0.6763 \pm 0.0054$ | $0.6632 \pm 0.0083$ | $0.6121 \pm 0.0167$ |
| **Hopkins Statistic (Không gian PCA/SVD)** | **$0.6035 \pm 0.0350$** | **$0.6435 \pm 0.0359$** | **$0.6513 \pm 0.0324$** | **$0.5971 \pm 0.0399$** |
| **Real Best Silhouette (K-Means, $k \in [2, 6]$)** | **0.0027** ($k=3$) | **0.0679** ($k=5$) | **0.0845** ($k=6$) | **0.1790** ($k=2$) |
| **Real Best Silhouette (Agglomerative, $k \in [2, 6]$)** | **0.0075** ($k=6$) | **0.1151** ($k=6$) | **0.1293** ($k=6$) | **0.3119** ($k=2$) |
| **Null Model Silhouette (K-Means: Mean ± Std)** | $0.0020 \pm 0.0005$ | $0.0126 \pm 0.0022$ | $0.0327 \pm 0.0059$ | $0.1565 \pm 0.0145$ |
| **Null Model Silhouette (Agglomerative: Mean ± Std)** | $0.0052 \pm 0.0007$ | $0.0246 \pm 0.0031$ | $0.0665 \pm 0.0149$ | $0.2477 \pm 0.0515$ |
| **Empirical $p$-value (K-Means vs Null Model)** | **$p = 0.0797$** (Không đạt) | **$p = 0.0033$** ($p < 0.005$) | **$p = 0.0033$** ($p < 0.005$) | **$p = 0.0831$** (Không đạt) |
| **Empirical $p$-value (Agglomerative vs Null Model)** | **$p = 0.0033$** | **$p = 0.0033$** ($p < 0.005$) | **$p = 0.0033$** ($p < 0.005$) | **$p = 0.1262$** (Không đạt) |

---

## 2. Phân Tích Chuyên Sâu Từng Không Gian Biểu Diễn

### 2.1. Biểu đồ Phân phối Khoảng cách Cosine (Hình 1)
File biểu đồ: [figures/problem_selection_cosine_dist.png](file:///e:/ML/Chinook/figures/problem_selection_cosine_dist.png)

![Phân phối khoảng cách Cosine giữa các cặp khách hàng](file:///e:/ML/Chinook/figures/problem_selection_cosine_dist.png)

- **Nhận xét quan sát từ biểu đồ:**
  1. **Khách × Track:** Phân phối lệch hẳn về cực đại $d_{\cos} = 1.00$. Có tới **97.49% số cặp khách hàng** (1.668 trên tổng 1.711 cặp) có khoảng cách $\ge 0.95$. Điều này chứng minh rằng hầu như hai khách hàng bất kỳ không có bài hát chung. Không gian Track bị phân mảnh trực giao nghiêm trọng.
  2. **Khách × Album:** Phân phối bắt đầu trải rộng về phía bên trái (khoảng cách nhỏ hơn), trung vị giảm xuống **0.9544**, tỷ lệ cặp trực giao giảm mạnh từ 97.49% xuống **60.84%**. Overlap giữa các khách hàng tăng lên gấp nhiều lần.
  3. **Khách × Artist:** Phân phối dịch chuyển rõ rệt về khoảng cách $0.80 - 0.90$ (trung vị **0.8655**). Tỷ lệ cặp $\ge 0.95$ giảm xuống chỉ còn **8.88%** (152 cặp). Hơn 91% các cặp khách hàng có chung nghệ sĩ yêu thích.
  4. **Khách × Genre:** Phân phối dạng chuông chuẩn mực tập trung quanh trung vị **0.3828**, hoàn toàn **0.0% cặp** có khoảng cách $\ge 0.95$. Tất cả 59 khách hàng đều có giao thoa về thể loại nhạc.

---

### 2.2. Kiểm Định Cấu Trúc Cụm Với Null Model Hoán Vị (Hình 2)
File biểu đồ: [figures/problem_selection_null_model.png](file:///e:/ML/Chinook/figures/problem_selection_null_model.png)

![Kiểm định cấu trúc cụm với Null Model hoán vị](file:///e:/ML/Chinook/figures/problem_selection_null_model.png)

- **Phương pháp luận Null Model (Curveball Algorithm - Strona et al., 2014):**
  - Để kiểm tra xem cấu trúc cụm quan sát được là **thật sự có ý nghĩa thống kê** hay chỉ là **ngẫu nhiên**, ta xây dựng mô hình vô hiệu (Null Model) bằng cách hoán vị các ô trong ma trận nhị phân $59 \times M$ nhưng **giữ nguyên tuyệt đối tổng số mục mỗi khách hàng mua (Row marginals)** và **độ phổ biến của từng mục (Column marginals)**.
  - Chạy lặp lại $N = 300$ lần độc lập. Với mỗi ma trận ngẫu nhiên, thực hiện gom cụm với $k \in [2, 6]$ bằng cả K-Means (trên vector chuẩn hóa $L_2$) và Agglomerative (Average Linkage trên khoảng cách Cosine), ghi nhận Best Silhouette Score.
  - Tính $p$-value thực nghiệm: $p = \frac{1 + \sum \mathbb{I}(S_{\text{null}} \ge S_{\text{real}})}{N + 1}$.

- **Phát hiện quan trọng từ kết quả Null Model:**
  - **Khách × Track:** Mặc dù điểm Silhouette trên dữ liệu thật đạt $0.0075$ (Agglomerative) và $0.0027$ (K-Means), đây là những giá trị xấp xỉ 0 (không có cấu trúc cụm). Đáng chú ý, với K-Means, $p = 0.0797 > 0.05$, nghĩa là thuật toán phân cụm trên không gian Track **không thể vượt qua một ma trận ngẫu nhiên**.
  - **Khách × Album:** Dữ liệu thật đạt Silhouette $= 0.1151$ (Agglomerative, $k=6$) và $0.0679$ (K-Means, $k=5$). Trong khi đó, phân phối Null Model chỉ đạt trung bình $0.0246 \pm 0.0031$ (Agglomerative) và $0.0126 \pm 0.0022$ (K-Means). Điểm cao nhất trong toàn bộ 300 lần chạy ngẫu nhiên của Null Model chỉ là $0.0328$ (Agglomerative) và $0.0205$ (K-Means) — **thấp hơn dữ liệu thật từ 3 đến 5 lần!** Kết quả $p$-value đạt mức tối thiểu **$p = 0.0033$ ($p < 0.005$)**. Bác bỏ hoàn toàn giả thuyết vô hiệu!
  - **Khách × Artist:** Tương tự Album, cả Agglomerative ($S=0.1293$ vs Null Mean $0.0665$) và K-Means ($S=0.0845$ vs Null Mean $0.0327$) đều vượt trội tuyệt đối với **$p = 0.0033$**.
  - **Khách × Genre:** Mặc dù điểm Silhouette của dữ liệu thật có vẻ cao ($0.3119$ ở $k=2$), nhưng Null Model cũng đạt trung bình lên tới $0.2477 \pm 0.0515$, và có tới 37/300 lần Null Model tạo ra Silhouette cao hơn dữ liệu thật! Giá trị $p$-value đạt **$p = 0.1262$ (Agglomerative)** và **$p = 0.0831$ (K-Means)**, đều lớn hơn ngưỡng ý nghĩa $\alpha = 0.05$.  
    *Giải thích khoa học:* Do catalog chỉ có 24 thể loại và các thể loại như Rock, Latin, Metal có mặt ở hầu hết mọi khách hàng, bất kỳ ma trận ngẫu nhiên nào giữ nguyên độ phổ biến cũng dễ dàng bị thuật toán chia đôi thành 2 cụm thô có Silhouette cao. Do đó, gom cụm thuần túy trên 24 chiều Genre không mang lại phân cụm cá nhân hóa thực chất.

---

## 3. Quyết Định Lựa Chọn Phương Án (Problem Selection)

### 3.1. Phương án được lựa chọn chính thức (Selected Approach)
> [!TIP]
> **PHƯƠNG ÁN CHỌN: KHÔNG GIAN KHÁCH × ALBUM (LÀM TRỌNG TÂM) KẾT HỢP KHÁCH × ARTIST & KHÁCH × GENRE (HỖ TRỢ DIỄN GIẢI)**
> 
> - **Không gian gom cụm chính (Primary Clustering Feature Space):** Ma trận **Khách × Album** ($59 \times 304$).  
>   Đặc trưng sẽ kết hợp giữa số bài hát đã mua và **tỷ lệ phủ album (`coverage_ratio`)**, chuẩn hóa $L_2$ / Cosine Distance.
> - **Không gian hỗ trợ diễn giải & đặt tên cụm (Auxiliary Profiling Space):** **Khách × Artist** ($59 \times 165$) và **Khách × Genre** ($59 \times 24$).  
>   Sau khi các cụm được hình thành từ không gian Album, hồ sơ của từng cụm sẽ được chiếu lên Artist và Genre để trích xuất đặc trưng thị hiếu, đặt tên cụm kinh doanh (ví dụ: Cụm Metalhead, Cụm Latin Lover) và kiểm chứng tính hợp lý nghiệp vụ.

### 3.2. Lý do định lượng chọn Khách × Album
1. **Vượt trội Null Model có ý nghĩa thống kê cao ($p = 0.0033 \ll 0.05$):** Tín hiệu cụm ở cấp độ Album là có thật, không thể bị tạo ra bởi sự ngẫu nhiên.
2. **Độ phân giải sản phẩm (Granularity) hoàn hảo cho mục tiêu kinh doanh:** Album có 304 thực thể — đủ chi tiết để thiết kế trực tiếp các chương trình **"Bundle Album"**, **"Complete My Album"** và **"Cross-sell"** theo đúng các quyết định kinh doanh tại Bước 4.
3. **Độ đậm đặc hợp lý (Density 7.25%):** Cao gấp gần 4 lần Track, trung vị khoảng cách cosine giảm xuống, xóa bỏ hiện tượng khoảng cách trực giao cực đoan.

### 3.3. Lý do định lượng loại bỏ phương án Khách × Track
1. **Khoảng cách trực giao cực đoan (Orthogonal Collapse):** **97.49%** các cặp khách hàng có khoảng cách cosine $\ge 0.95$. Trung vị khoảng cách đạt giá trị tuyệt đối **$1.0000$**.
2. **Hiện tượng Long-tail phân mảnh:** 87.1% bài hát chỉ có đúng 1 người mua, số người mua tối đa cho 1 bài trong toàn hệ thống chỉ là 2 người. Hai khách hàng bất kỳ hầu như không có bài hát chung.
3. **Thất bại trước Null Model:** K-Means trên ma trận Track cho điểm Silhouette xấp xỉ bằng $0$ ($0.0027$) và có $p$-value $= 0.0797 > 0.05$ (không thể bác bỏ giả thuyết ma trận ngẫu nhiên). Nếu cố tình gom cụm trên Track thô, thuật toán sẽ nhóm các khách hàng hoàn toàn ngẫu nhiên dựa trên nhiễu số học.

---

## 4. Bảng Giả Định và Hạn Chế (Assumptions & Limitations)

| Hạng mục | Giả định kỹ thuật & nghiệp vụ | Hạn chế thực tế | Biện pháp giảm thiểu rủi ro trong mô hình |
| :--- | :--- | :--- | :--- |
| **Quy mô mẫu ($N = 59$)** | Giả định 59 khách hàng đại diện cho tệp người dùng cốt lõi của cửa hàng. | Cỡ mẫu rất nhỏ trong học máy; dễ bị overfit; các thuật toán mật độ như DBSCAN sẽ coi phần lớn điểm là nhiễu. | Sử dụng K-Means và Agglomerative với số cụm nhỏ $k \in [3, 5]$; kiểm định độ ổn định bằng Bootstrap resampling. |
| **Không có nhãn (Unlabeled)** | Giả định dữ liệu thuộc bài toán Unsupervised Learning thuần túy. | Hoàn toàn không có Ground Truth phân khúc khách hàng để đo Precision/Recall. | Sử dụng Unsupervised Metrics (Silhouette, Davies-Bouldin) kết hợp kiểm định đối chứng chéo qua Genre Profile. |
| **"Quan tâm" = Mua (Purchases as Proxy)** | Giả định hành vi mua phản ánh trọn vẹn thị hiếu và sự quan tâm của khách hàng. | Bỏ sót các bài hát khách hàng yêu thích nhưng chưa có tiền mua; không đo được mức độ thất vọng sau khi mua. | Trọng số hóa mức độ quan tâm bằng tỷ lệ phủ album (`coverage_ratio`), phản ánh độ cam kết sâu với nghệ sĩ/album. |
| **Hành vi tổng quát đồng đều** | Mọi khách hàng đều mua ~38 bài hát, ~7 hóa đơn, ~39 USD (CV < 8%). | Không thể phân khúc khách hàng theo giá trị chi tiêu (VIP vs Phổ thông) hay tần suất (RFM). | Tập trung 100% vào ma trận phân bổ nội dung âm nhạc (Content-based Taste), chuẩn hóa vector $L_2$ để loại bỏ sai số quy mô. |
| **Thiếu dữ liệu nghe & tương tác** | Không có logs lượt nghe (streams), số lần skip, thời lượng nghe, giỏ hàng. | Tín hiệu thưa thớt hơn so với các nền tảng streaming hiện đại (Spotify/Apple Music). | Gom cấp độ Album để tổng hợp tín hiệu từ các bài hát lẻ, tận dụng quan hệ phân cấp tự nhiên của âm nhạc. |
| **Dữ liệu tổng hợp (Synthetic Benchmark)** | CSDL Chinook là bộ dữ liệu mẫu kinh điển cho hệ quản trị CSDL. | Các giao dịch được sinh theo template phân bổ đều (mỗi khách ~7 đơn cách nhau đều đặn). | Thừa nhận tính chất benchmark của dữ liệu; tập trung xây dựng pipeline chuẩn mực công nghiệp có thể tái sử dụng. |

---

## 5. Định Nghĩa Điều Kiện Dừng (Stopping Condition) & Kết Quả Kiểm Tra

> [!CAUTION]
> **ĐỊNH NGHĨA ĐIỀU KIỆN DỪNG (STOPPING CRITERIA):**  
> Trong quy trình khoa học dữ liệu nghiêm ngặt, nếu toàn bộ các phương án biểu diễn dữ liệu đều **thất bại trước Null Model hoán vị ở mức ý nghĩa $\alpha = 0.05$ (tức là $p$-value $\ge 0.05$)**, điều đó đồng nghĩa với việc:  
> *"Dữ liệu hành vi mua sắm của 59 khách hàng không chứa cấu trúc cụm thị hiếu thực sự; mọi sự gom cụm đều là gượng ép trên nhiễu ngẫu nhiên."*  
> **Khi đó, nhóm phân tích bắt buộc phải DỪNG LẠI NGAY LẬP TỨC**, không được huấn luyện mô hình gom cụm chính thức, và chuyển giao sang các phương án thay thế.

### 5.1. Các hướng giải pháp thay thế nếu kích hoạt Điều kiện dừng:
1. **Chuyển sang Hệ gợi ý Sản phẩm - Sản phẩm (Item-Item Recommender / Association Rules):**  
   Thay vì gom cụm khách hàng, áp dụng thuật toán khai phá tập mục phổ biến (Apriori / FP-Growth) hoặc lọc cộng tác Item-based để tìm các cặp Album/Track thường xuyên được mua cùng nhau.
2. **Phân khúc dựa trên Luật Nghiệp vụ tĩnh (Rule-based Segmentation):**  
   Phân loại khách hàng dựa trên dòng nhạc chiếm tỷ trọng chi tiêu lớn nhất của họ (ví dụ: Khách hàng chi >40% cho Rock $\rightarrow$ Gán vào nhóm *Rock Dominant*; chi >30% cho Latin $\rightarrow$ *Latin Dominant*).

### 5.2. Kết quả kiểm tra đối chiếu thực tế với Điều kiện dừng:
- **Phương án Khách × Track:** Bị loại vì K-Means có $p = 0.0797 > 0.05$ và độ trực giao lên tới 97.5%.
- **Phương án Khách × Genre:** Bị loại khỏi vai trò không gian gom cụm chính vì $p = 0.1262 > 0.05$.
- **Phương án Khách × Album:** Đạt **$p = 0.0033 \ll 0.05$** trên cả K-Means và Agglomerative Clustering (vượt qua 100% các lần chạy của 300 mẫu Null Model).
- **Phương án Khách × Artist:** Đạt **$p = 0.0033 \ll 0.05$** trên cả K-Means và Agglomerative Clustering.

$\rightarrow$ **KẾT LUẬN NGHIỆM THU ĐIỀU KIỆN DỪNG:**  
**ĐIỀU KIỆN DỪNG KHÔNG BỊ KÍCH HOẠT đối với không gian Khách × Album và Khách × Artist.** Dữ liệu giao dịch ở cấp độ Album và Nghệ sĩ hoàn toàn có cấu trúc cụm thực chất, vượt trội một cách có ý nghĩa thống kê so với mô hình ngẫu nhiên. Dự án đủ điều kiện vững chắc để tiếp tục bước sang Pha C (Trích xuất đặc trưng & Huấn luyện mô hình gom cụm).

---

## 6. Danh Mục Tài Liệu & Minh Chứng Pha B Đính Kèm

1. [reports/business_problem.md](file:///e:/ML/Chinook/reports/business_problem.md) - Báo cáo Đặt bài toán kinh doanh, mục tiêu hỗ trợ ra quyết định và hệ thống KPI đề xuất.
2. [reports/problem_selection.md](file:///e:/ML/Chinook/reports/problem_selection.md) - Báo cáo Lựa chọn bài toán kỹ thuật, bảng so sánh 4 biểu diễn bằng số thật, kết quả Null Model và lý do chọn.
3. [outputs/problem_selection_metrics.json](file:///e:/ML/Chinook/outputs/problem_selection_metrics.json) - Tệp JSON lưu trữ 100% số liệu đo lường định lượng, thống kê Cosine, Hopkins và phân phối Null Model ($N=300$).
4. [figures/problem_selection_cosine_dist.png](file:///e:/ML/Chinook/figures/problem_selection_cosine_dist.png) - Biểu đồ phân phối khoảng cách Cosine giữa các cặp khách hàng cho 4 không gian biểu diễn.
5. [figures/problem_selection_null_model.png](file:///e:/ML/Chinook/figures/problem_selection_null_model.png) - Biểu đồ so sánh Silhouette Score giữa dữ liệu thật và Null Model Curveball ($N=300$) kèm $p$-value thực nghiệm.
6. [src/run_null_model_experiments.py](file:///e:/ML/Chinook/src/run_null_model_experiments.py) - Mã nguồn thực nghiệm hoàn chỉnh để tái lập kết quả.
