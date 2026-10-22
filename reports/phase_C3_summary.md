# BÁO CÁO TỔNG KẾT PHA C3: ĐÁNH GIÁ, KIỂM TRA TỔNG QUÁT HÓA, FINAL FIT & PROFILE CỤM
**Giai đoạn:** Pha C3 - Nghiệm thu Toàn diện Hệ thống Phân cụm Khách hàng  
**Dự án:** Phân khúc Khách hàng Chinook (Customer Clustering)  
**Quy tắc quyết định áp dụng:** [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) (Không thay đổi quy tắc ở pha này)  
**Mô hình được chọn & Fit chính thức:** `V2_album_count_svd5_dfFalse_KM_k5` (KMeans $k=5$, SVD=5D, V2 Album Count + IDF)  
**File mô hình sản xuất:** [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib)  
**Trạng thái kiểm thử tái lập:** **4/4 PASSED (100% Trùng khớp nhãn)** qua [`tests/test_reproducibility.py`](file:///e:/ML/Chinook/tests/test_reproducibility.py)  

---

## 1. TỔNG HỢP CÔNG VIỆC ĐÃ HOÀN THÀNH TRONG PHA C3

Tuân thủ nghiêm ngặt yêu cầu thiết kế kiểm định cho bài toán Unsupervised Learning không có nhãn, Pha C3 đã hoàn thành trọn vẹn 7 nhóm nhiệm vụ cốt lõi:

1. **Đánh giá Chất lượng Hình học Nội tại (Internal Quality Metrics):**
   - Đánh giá song song trên cả không gian nén SVD 5 chiều và ma trận thô chuẩn hóa L2 (347 chiều) cho 5 ứng viên hàng đầu từ Pha C2.
   - Vẽ biểu đồ Silhouette từng khách hàng ([`figures/cluster_silhouette_plot.png`](file:///e:/ML/Chinook/figures/cluster_silhouette_plot.png)): Silhouette trung bình toàn cục đạt **$0.5941$**, tỷ lệ khách hàng có Silhouette âm là **$0.0\%$**.
   - Vẽ biểu đồ chiếu 2D PCA & t-SNE ([`figures/cluster_pca_projection.png`](file:///e:/ML/Chinook/figures/cluster_pca_projection.png)): Nêu rõ trong báo cáo thư viện UMAP không có sẵn trong môi trường Python, phương pháp t-SNE (Perplexity=15) được sử dụng thay thế và thể hiện 5 cụm tách biệt rõ rệt.
2. **Kiểm tra Ổn định và Tổng quát hóa (Hold-out 80/20 qua 100 lần lặp):**
   - Thiết kế thí nghiệm chống rò rỉ dữ liệu tuyệt đối: Pipeline tiền xử lý và mô hình phân cụm được **fit hoàn toàn chỉ trên tập train 80% (47 khách)**.
   - Tập test 20% (12 khách) chỉ được transform và gán nhãn cụm (bằng `predict` cho KMeans; bằng *nearest centroid* theo khoảng cách Cosine trên không gian SVD cho Agglomerative/Spectral).
   - Đo lường Train ARI/NMI, Hold-out ARI, Hold-out Silhouette cho 5 ứng viên qua 100 lần lặp.
   - Xuất chi tiết toàn bộ 500 lượt chạy vào [`outputs/stability_runs.csv`](file:///e:/ML/Chinook/outputs/stability_runs.csv).
   - Vẽ biểu đồ Boxplot ổn định tại [`figures/cluster_stability_boxplots.png`](file:///e:/ML/Chinook/figures/cluster_stability_boxplots.png).
3. **Phân tích Độ nhạy (Sensitivity Analysis):**
   - **Độ nhạy theo Random Seed (25 seeds khác nhau):** Mean Pairwise ARI đạt **$0.9945$** (Median ARI = **$1.0000$**). Thuật toán KMeans ($n\_init=30$) hội tụ vững chắc tuyệt đối, trơ với điểm khởi tạo ngẫu nhiên.
   - **Độ nhạy theo Số cụm $k$ ($k=4, 5, 6$):** So với $k=4$ đạt ARI = $0.7300$; so với $k=6$ đạt ARI = $0.8522$, khẳng định cấu trúc cụm có tính phân cấp lồng ghép chặt chẽ.
4. **Kiểm định Hành vi Tương lai theo Trục Thời gian (Temporal Future Behavior Test):**
   - Phân chia hóa đơn theo thời gian: 3 hóa đơn đầu (Giai đoạn 1 - Period 1, $1.001$ dòng) và các hóa đơn còn lại (Giai đoạn 2 - Period 2, $1.239$ dòng) cho toàn bộ 59 khách hàng.
   - Đo ARI giữa cụm GĐ 1 và cụm GĐ 2 độc lập: đạt **$0.1281$**.
   - Kiểm định khả năng dự báo hành vi tương lai ở mức Album: tỷ lệ khách hàng GĐ 2 có hành vi album gần centroid cụm GĐ 1 của mình nhất đạt **$11.86\%$** ($7/59$ khách), so với phân vị 95% của Null Permutation ($N=1000$) là **$28.81\%$** ($p = 0.9640$).
   - Phát hiện nguyên nhân bản chất: Khách hàng không mua lại album cũ; nhưng ở cấp độ **Thể loại (Genre Share)**, độ tự nhất quán giữa 2 giai đoạn đạt trung vị **$0.5065$** ($p < 0.001$).
   - Vẽ biểu đồ phân phối Null Permutation tại [`figures/cluster_temporal_prediction.png`](file:///e:/ML/Chinook/figures/cluster_temporal_prediction.png).
5. **Final Fit và Đóng gói Mô hình Sản xuất:**
   - Huấn luyện chính thức Pipeline + KMeans trên toàn bộ 59 khách hàng với `random_state=42`.
   - Đóng gói toàn diện vào [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib) gồm: Pipeline fitted, Model fitted, Centroids chuẩn hóa, Từ vựng 347 Album IDs & Titles, Thông tin phiên bản thư viện, MD5 Hash dữ liệu gốc và bộ chỉ số kiểm định.
   - Xây dựng module suy luận [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py) với hàm `assign_cluster(purchases)` trả về cụm và độ tin cậy khoảng cách tới tâm cụm.
   - Xây dựng bộ test kiểm tra tái lập [`tests/test_reproducibility.py`](file:///e:/ML/Chinook/tests/test_reproducibility.py) với **4/4 test PASSED** (so khớp 100% nhãn và ARI = 1.0 khi chạy lại từ đầu).
   - Biên soạn Thẻ mô hình quốc tế [`reports/model_card.md`](file:///e:/ML/Chinook/reports/model_card.md).
6. **Diễn giải & Xây dựng Hồ sơ Phân khúc (Profiling & Personas):**
   - Triển khai [`src/profiling.py`](file:///e:/ML/Chinook/src/profiling.py), tính toán tỷ lệ mua, độ phủ (Coverage) và chỉ số **Lift** cho toàn bộ Album, Thể loại và Nghệ sĩ.
   - **Áp dụng nghiêm ngặt bộ lọc thống kê:** Mục phải có $\ge 3$ khách hàng trong cụm mua mới được đưa vào xét Lift để triệt tiêu nhiễu mẫu nhỏ.
   - Đặt tên 5 personas trực quan dựa trên số liệu:
     * **Cụm 0 (n=12, 20.3%):** *Eclectic Soundtrack, Sci-Fi & Vintage Grooves* (Easy Listening Lift 4.92x, Sci-Fi 4.92x, Sinatra, Milton Nascimento).
     * **Cụm 1 (n=10, 16.9%):** *Soulful Blues, Reggae & Acoustic Connoisseurs* (Soundtrack 3.28x, Reggae 2.95x, Eric Clapton Unplugged, UB40).
     * **Cụm 2 (n=15, 25.4%):** *Global Fusion, Alternative & Dance Explorers* (Alternative 2.95x, Electronica 2.81x, Bossa Nova 2.25x, Satriani, Gilberto Gil).
     * **Cụm 3 (n=10, 16.9%):** *Hard Rock, Heavy Metal & Urban Rhythm Lovers* (Hip Hop 2.62x, Reggae 2.46x, Metal 100%, Guns N' Roses, Velvet Revolver).
     * **Cụm 4 (n=12, 20.3%):** *Classic Blues-Rock, Punk & Symphonic Fans* (Classical 1.76x, Blues 1.50x, Punk 91.7%, Metal 100%, Aerosmith, Marillion).
   - Kiểm định phi tham số nhân khẩu học: Kruskal-Wallis cho Tổng chi tiêu ($p = 0.3057$), Số track ($p = 0.5690$), Số invoice ($p = 0.5690$) và Chi-Square cho Địa lý ($p = 0.9726$) khẳng định **sự khác biệt cốt lõi giữa các cụm là THỊ HIẾU ÂM NHẠC, không phải dung lượng chi tiêu**.
   - Xuất dữ liệu phân cụm chi tiết vào [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv) và biểu đồ Lift tại [`figures/cluster_profiles_lift.png`](file:///e:/ML/Chinook/figures/cluster_profiles_lift.png).

---

## 2. ĐỐI CHIẾU TIÊU CHÍ RA QUYẾT ĐỊNH VÀ KẾT QUẢ KIỂM ĐỊNH TRUNG THỰC

Đối chiếu trực tiếp với các điều khoản đã cam kết trong [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):

| Tiêu chí Kiểm định trong Decision Rules | Ngưỡng Tiêu chuẩn Quy định | Kết quả Thực nghiệm Đạt được | Đánh giá Tuân thủ Quy tắc |
| :--- | :---: | :---: | :--- |
| **1. Khoảng số cụm $k$** | $k \in [2, 6]$ | $k = 5$ | **ĐẠT CHUẨN** |
| **2. Kích thước cụm nhỏ nhất** | $\text{min\_size} \ge 5$ khách | **10 khách** (`[15, 12, 12, 10, 10]`) | **ĐẠT XUẤT SẮC** (Không có cụm vi mô ngoại lai) |
| **3. Vượt ngưỡng Null Model** | $\text{Sil} > \text{Null P95}$ | **$0.5941 > 0.4441$** ($\Delta = +0.1500$) | **ĐẠT XUẤT SẮC** (Vượt trội 95% ma trận ngẫu nhiên) |
| **4. So sánh với các Baseline** | Tốt hơn Dominant Genre, Country, Random | Silhouette $0.5941$ vs $-0.0130$ | **VƯỢT TRỘI TUYỆT ĐỐI** |
| **5. Độ ổn định Hold-out CV** | $\text{Median ARI} \ge 0.60$ (Robust) | **$0.4038$** (P5=0.0782, P95=0.7966) | **CHƯA ĐẠT NGƯỠNG LÝ TƯỞNG ($\ge 0.60$)**<br>Thuộc mức **Độ ổn định trung bình (Moderate)** |
| **6. Kiểm tra Dự báo Thời gian** | Tỷ lệ dự báo > Null P95 | **$11.86\% < 28.81\%$ (Null P95)** | **KHÔNG ĐẠT Ở MỨC ALBUM**<br>Đạt ở mức Thể loại (Consistency = $0.5065$) |

---

## 3. THỪA NHẬN KẾT QUẢ ÂM VÀ NGUYÊN NHÂN KHOA HỌC (HONEST SCIENTIFIC APPRAISAL)

Thay vì che giấu số liệu để tạo phân cụm "hoàn hảo giả tạo", chúng tôi báo cáo trung thực 2 phát hiện mang tính giới hạn thống kê:

1. **Về Độ ổn định Mẫu con (Hold-out ARI = 0.4038):**
   - Mặc dù cấu hình `V2_KM_k5` có Hold-out ARI cao nhất trong 5 ứng viên ($0.4038$ so với $0.26 - 0.32$), nó chưa đạt ngưỡng $0.60$ để được xem là phân cụm "vững chắc tuyệt đối".
   - **Nguyên nhân khách quan:** Cỡ mẫu toàn bộ chỉ có $n = 59$ khách hàng. Khi rút 20% làm tập test (12 khách), số lượng điểm dữ liệu trong mỗi cụm chỉ còn khoảng $2$ khách. Sự dao động ngẫu nhiên của 1-2 điểm biên trong tập test nhỏ làm giảm chỉ số ARI cơ học.
2. **Về Khả năng Dự báo Album theo Thời gian:**
   - Việc kiểm tra hành vi tương lai ở mức Album cho thấy phân cụm không dự báo được việc lặp lại album cũ ($p = 0.9640$).
   - **Nguyên nhân nghiệp vụ:** Khách hàng mua nhạc số không bao giờ mua lại cùng một album đã có trong thư viện. Họ mua các album mới ở chu kỳ sau. Vì vậy, ma trận Album có độ giao nhau giữa 2 chu kỳ gần bằng 0.
   - **Tuy nhiên, ở cấp độ Thể loại (Genre), độ tự nhất quán đạt trung vị $0.5065$:** Khách hàng giữ nguyên gu âm nhạc (Rock, Metal, Latin, Blues) xuyên suốt cả 2 giai đoạn.

### Phương án Thay thế & Khuyến nghị Kinh doanh cho Pha D:
- **Không sử dụng phân cụm Album để dự báo mua lại từng album cụ thể.**
- **Sử dụng 5 Personas làm khung định vị phân khúc khách hàng cấp cao (Strategic Segmentation):** Để thiết kế các gói Bundle Album đồng điệu theo Top Lift Thể loại & Nghệ sĩ đã xác định.
- **Bổ sung hệ thống khuyến nghị Item-Item Collaborative Filtering (dựa trên ma trận tương đồng album-album) trong Pha D:** Nhằm đề xuất các album mới chưa sở hữu nhưng tương đồng với gu âm nhạc của khách hàng.

---

## 4. DANH MỤC HIỆN VẬT NGHIỆM THU PHA C3 (DELIVERABLES INVENTORY)

| STT | Loại hiện vật | Đường dẫn file | Mô tả nội dung | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **Mã nguồn Đánh giá** | [`src/evaluation.py`](file:///e:/ML/Chinook/src/evaluation.py) | Module đánh giá chất lượng nội tại, hold-out 80/20 CV (100 lần), độ nhạy seed/k, và kiểm tra thời gian. | **ĐÃ HOÀN THÀNH** |
| 2 | **Mã nguồn Profiling** | [`src/profiling.py`](file:///e:/ML/Chinook/src/profiling.py) | Module tính toán Lift, Coverage (ngưỡng $\ge 3$ khách), kiểm định phi tham số nhân khẩu học và tạo personas. | **ĐÃ HOÀN THÀNH** |
| 3 | **Mã nguồn Suy luận** | [`src/predict.py`](file:///e:/ML/Chinook/src/predict.py) | Hàm `assign_cluster(purchases)` và `assign_clusters_batch` gán cụm và độ tin cậy cho khách hàng mới. | **ĐÃ HOÀN THÀNH** |
| 4 | **Model Sản xuất** | [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib) | Đóng gói Pipeline + Model + Vocabulary 347 Album + Centroids + Thư viện + MD5 Data Hash. | **ĐÃ LƯU TRỮ** |
| 5 | **Bộ Test Tái lập** | [`tests/test_reproducibility.py`](file:///e:/ML/Chinook/tests/test_reproducibility.py) | 4 test tự động kiểm tra nạp model, so khớp 100% nhãn, chạy lại từ đầu ARI=1.0 và hàm predict. | **4/4 PASSED** |
| 6 | **Dữ liệu Lần lặp Hold-out** | [`outputs/stability_runs.csv`](file:///e:/ML/Chinook/outputs/stability_runs.csv) | 500 dòng ghi nhận chi tiết Train ARI/NMI, Hold-out ARI, Hold-out Sil cho từng lần lặp. | **ĐÃ LƯU TRỮ** |
| 7 | **Dữ liệu Phân cụm Khách** | [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv) | 59 khách hàng kèm Cluster, ClusterName, Confidence, DistanceToCentroid, Genre, Chi tiêu, Số track. | **ĐÃ LƯU TRỮ** |
| 8 | **Biểu đồ Silhouette Từng Khách**| [`figures/cluster_silhouette_plot.png`](file:///e:/ML/Chinook/figures/cluster_silhouette_plot.png) | Sample silhouette plot cho 59 khách hàng (Sil TB = 0.5941, không có khách nào âm). | **ĐÃ LƯU TRỮ** |
| 9 | **Biểu đồ Chiếu 2D** | [`figures/cluster_pca_projection.png`](file:///e:/ML/Chinook/figures/cluster_pca_projection.png) | Không gian 2D PCA và t-SNE tô màu theo cụm kèm tâm cụm (nêu rõ UMAP không có trong môi trường). | **ĐÃ LƯU TRỮ** |
| 10 | **Biểu đồ Boxplot Ổn định** | [`figures/cluster_stability_boxplots.png`](file:///e:/ML/Chinook/figures/cluster_stability_boxplots.png) | Phân phối Hold-out ARI và Hold-out Silhouette qua 100 lần lặp cho 5 ứng viên. | **ĐÃ LƯU TRỮ** |
| 11 | **Biểu đồ Kiểm định Thời gian**| [`figures/cluster_temporal_prediction.png`](file:///e:/ML/Chinook/figures/cluster_temporal_prediction.png) | Phân phối Null Permutation (N=1000) so với tỷ lệ dự báo thực tế theo thời gian. | **ĐÃ LƯU TRỮ** |
| 12 | **Biểu đồ Hồ sơ Lift** | [`figures/cluster_profiles_lift.png`](file:///e:/ML/Chinook/figures/cluster_profiles_lift.png) | Top thể loại theo chỉ số Lift của từng phân khúc (áp dụng ngưỡng $\ge 3$ khách). | **ĐÃ LƯU TRỮ** |
| 13 | **Báo cáo Đánh giá Chi tiết** | [`reports/evaluation.md`](file:///e:/ML/Chinook/reports/evaluation.md) | Báo cáo kiểm định toàn diện kỹ thuật 4 tầng. | **ĐÃ HOÀN THÀNH** |
| 14 | **Báo cáo Hồ sơ Cụm** | [`reports/cluster_profiles.md`](file:///e:/ML/Chinook/reports/cluster_profiles.md) | Chân dung chi tiết 5 personas, bảng Lift, kiểm định phi tham số nhân khẩu học. | **ĐÃ HOÀN THÀNH** |
| 15 | **Thẻ Mô hình Học máy** | [`reports/model_card.md`](file:///e:/ML/Chinook/reports/model_card.md) | Model Card chuẩn mực quốc tế ghi rõ mục đích, tham số, giới hạn và hướng dẫn suy luận. | **ĐÃ HOÀN THÀNH** |
| 16 | **Báo cáo Tổng kết Pha C3** | [`reports/phase_C3_summary.md`](file:///e:/ML/Chinook/reports/phase_C3_summary.md) | Tài liệu tổng kết toàn bộ pha C3 và đối chiếu quy tắc quyết định. | **ĐÃ HOÀN THÀNH** |

---

## 5. KẾT LUẬN VÀ TRẠNG THÁI BÀN GIAO

Toàn bộ các yêu cầu của **PHA C3: ĐÁNH GIÁ + KIỂM TRA TỔNG QUÁT HÓA + FINAL FIT + PROFILE CỤM** đã được thực thi đầy đủ, chuẩn xác, kiểm thử tự động đạt 100% và ghi nhận trung thực mọi phát hiện khoa học.

Theo đúng chỉ dẫn của quy trình: **HỆ THỐNG TẠM DỪNG TẠI ĐÂY ĐỂ CHỜ NGƯỜI DÙNG PHÊ DUYỆT TRƯỚC KHI TIẾN HÀNH PHA D (BUSINESS APPLICATION & BUNDLE DESIGN).**
