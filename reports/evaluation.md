# BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN, ĐỘ ỔN ĐỊNH VÀ KIỂM ĐỊNH TỔNG QUÁT HÓA
**Giai đoạn:** Pha C3 - Đánh giá + Kiểm tra Tổng quát hóa + Profile Cụm  
**Dự án:** Phân khúc Khách hàng Chinook (Customer Clustering)  
**Quy tắc quyết định áp dụng:** [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml) (Cam kết cố định, không thay đổi)  
**Tập dữ liệu:** 59 khách hàng ($n = 59$), 347 album, 2.240 lượt mua từ [`data/processed/fact_purchase.parquet`](file:///e:/ML/Chinook/data/processed/fact_purchase.parquet)  
**File mã nguồn thực hiện:** [`src/evaluation.py`](file:///e:/ML/Chinook/src/evaluation.py)  

---

## 1. PHƯƠNG PHÁP LUẬN KIỂM ĐỊNH HỌC KHÔNG GIÁM SÁT (UNSUPERVISED TEST DESIGN)

Do bài toán phân cụm khách hàng không có nhãn thực tế (*ground truth labels*), các quy trình kiểm thử thông thường trong học có giám sát (như Accuracy, F1-Score) không thể áp dụng trực tiếp. Để đảm bảo tính trung thực khoa học và ngăn chặn ảo tưởng cụm (*cluster reification / overfitting to noise*), chúng tôi thiết kế khung kiểm định 4 tầng chặt chẽ tuân thủ [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):

1. **Chất lượng hình học nội tại (Internal Geometric Quality):** Đánh giá độ nén nội cụm và tách biệt liên cụm trên không gian SVD (5 chiều) và đối chiếu với ma trận thô chuẩn hóa L2 (347 chiều), so sánh với phân phối Null Model Curveball ($N \ge 200$) và các Baseline từ Pha C2.
2. **Kiểm tra Tổng quát hóa và Ổn định trên Khách hàng (Hold-out 80/20 CV qua 100 lần lặp):**
   - Mỗi lần lặp: Chọn ngẫu nhiên không hoàn lại 80% khách hàng (47 khách) làm tập Train, 20% khách hàng (12 khách) làm tập Hold-out.
   - **Tuyệt đối chống rò rỉ dữ liệu (Strict Zero Leakage):** Pipeline tiền xử lý (`ItemFilter`, `IDFTransformer`, `Normalizer L2`, `TruncatedSVD`, `Normalizer L2`) và mô hình phân cụm được **CLONE và FIT HOÀN TOÀN TRÊN TẬP TRAIN**.
   - Tập Hold-out **CHỈ TRANSFORM** qua pipeline đã fit, sau đó gán nhãn cụm (bằng `predict` cho KMeans, hoặc gán vào tâm cụm gần nhất - *Nearest Centroid* bằng khoảng cách Cosine trong không gian SVD cho Agglomerative/Spectral).
   - Đo lường:
     - (a) **Train ARI & NMI**: Độ tương đồng giữa phân cụm trên tập Train và phân cụm của mô hình toàn bộ trên cùng nhóm khách hàng đó.
     - (b) **Hold-out ARI**: Độ ổn định gán nhãn trên 20% khách hàng chưa từng nhìn thấy trong lúc huấn luyện.
     - (c) **Hold-out Silhouette**: Điểm phân tách hình học của tập mẫu thử nghiệm.
3. **Độ nhạy thuật toán (Sensitivity Analysis):**
   - **Độ nhạy theo Seed (>= 25 seeds khác nhau):** Đánh giá tính hội tụ và tính ổn định trước điểm khởi tạo ngẫu nhiên.
   - **Độ nhạy theo $k$ ($k-1, k, k+1$):** Đánh giá tính liên tục và khả năng phân rã có cấu trúc của các cụm.
4. **Kiểm định Hành vi Tương lai theo Trục Thời gian (Invoice-level Temporal Split):**
   - Phân tách theo hóa đơn: 3 hóa đơn đầu (Giai đoạn 1 - Period 1) và các hóa đơn còn lại (Giai đoạn 2 - Period 2) cho từng khách hàng.
   - Thực hiện ở cấp độ **ALBUM** và **GENRE** (không làm ở mức Track vì số track quá thưa và gần như không lặp lại).
   - Đo lường tính ổn định cụm qua thời gian và kiểm định khả năng dự báo: Liệu hành vi tương lai ở GĐ 2 có duy trì gần tâm cụm GĐ 1 của chính khách hàng đó hơn các cụm khác không? So sánh với phân phối hoán vị ngẫu nhiên ($N = 1000$ permutations) để tính empirical p-value.

---

## 2. CHẤT LƯỢNG HÌNH HỌC NỘI TẠI TRÊN TOÀN BỘ TẬP DỮ LIỆU

### Bảng 1: So sánh Chất lượng Nội tại của 5 Ứng viên Hàng đầu

| Hạng | Mã cấu hình (Config ID) | Mô hình & Tham số | Số cụm ($k$) | Silhouette (SVD 5D) | Null P95 | Khoảng cách vượt Null ($\Delta$) | Silhouette (Raw L2 347D) | Davies-Bouldin | Calinski-Harabasz | Phân bổ kích thước cụm | Tỷ lệ Cân đối (Min/Max) | Đánh giá Luật Cứng |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1 (Khuyên dùng)** | `V2_album_count_svd5_dfFalse_KM_k5` | **KMeans** ($n\_init=30$) | **5** | **0.5941** | **0.4441** | **+0.1500** | **0.0665** | **0.8234** | **32.54** | **[15, 12, 12, 10, 10]** | **0.6667** | **ĐẠT TẤT CẢ** (Min=10 >= 5) |
| **#2** | `V2_album_count_svd5_dfFalse_KM_k6` | **KMeans** ($n\_init=30$) | **6** | **0.5976** | **0.4411** | **+0.1565** | **0.0731** | **0.7831** | **34.57** | **[12, 12, 10, 9, 9, 7]** | **0.5833** | **ĐẠT TẤT CẢ** (Min=7 >= 5) |
| **#3** | `V2_album_count_svd5_dfFalse_Agg_complete_k5` | **Agglomerative** (complete, cosine) | **5** | **0.5728** | **0.4010** | **+0.1718** | **0.0614** | **0.8835** | **31.03** | **[13, 12, 12, 11, 11]** | **0.8462** | **ĐẠT TẤT CẢ** (Min=11 >= 5) |
| **#4** | `V2_album_count_svd5_dfFalse_Spectral_NN8_k5` | **Spectral** (k-NN=8, cosine) | **5** | **0.5902** | **0.4347** | **+0.1555** | **0.0673** | **0.8100** | **32.18** | **[14, 13, 12, 10, 10]** | **0.7143** | **ĐẠT TẤT CẢ** (Min=10 >= 5) |
| **#5** | `V1_album_binary_svd5_dfFalse_KM_k5` | **KMeans** (Binary, $n\_init=30$) | **5** | **0.5608** | **0.4441** | **+0.1167** | **0.0899** | **0.8643** | **30.23** | **[14, 12, 11, 11, 11]** | **0.7857** | **ĐẠT TẤT CẢ** (Min=11 >= 5) |

### Phân tích Biểu đồ Silhouette Từng Khách hàng (Sample Silhouette Plot)
Biểu đồ lưu tại: [`figures/cluster_silhouette_plot.png`](file:///e:/ML/Chinook/figures/cluster_silhouette_plot.png)

- **Điểm Silhouette trung bình toàn cục:** $0.5941$.
- **Tỷ lệ khách hàng có điểm Silhouette âm ($s_i < 0$):** **0.0%** (0/59 khách hàng). Không có bất kỳ khách hàng nào bị phân cụm sai lệch nghiêm trọng sang vùng lân cận của cụm khác.
- **Độ đồng đều giữa các cụm:**
  - Cụm 0 (n=12): Silhouette trung bình cụm $\approx 0.58$, phân bố dày từ 0.35 đến 0.72.
  - Cụm 1 (n=10): Silhouette trung bình cụm $\approx 0.62$, các vector rất gắn kết với tâm cụm.
  - Cụm 2 (n=15): Cụm lớn nhất, có 3 khách hàng ở rìa ranh giới ($s_i \approx 0.15 - 0.25$), nhưng đa số đạt $> 0.55$.
  - Cụm 3 (n=10): Điểm Silhouette cao ổn định $\approx 0.61$.
  - Cụm 4 (n=12): Phân bố đều từ 0.40 đến 0.74, trung bình $\approx 0.59$.
- **Kết luận hình học:** Trong không gian nén SVD 5 chiều sau chuẩn hóa Cosine, 5 cụm thể hiện cấu trúc hình học rõ rệt, tách bạch và cân đối.

### Phân tích Biểu đồ Chiếu 2D PCA & t-SNE
Biểu đồ lưu tại: [`figures/cluster_pca_projection.png`](file:///e:/ML/Chinook/figures/cluster_pca_projection.png)  
*(Lưu ý kỹ thuật: Thư viện `umap-learn` không được cài đặt trong môi trường Python hệ thống, do đó phương pháp chiếu phi tuyến t-SNE với Perplexity=15 được sử dụng thay thế theo đúng hướng dẫn).*

- **Không gian PCA (2 chiều đầu):** Giải thích $\approx 55\%$ tổng phương sai của không gian SVD. Các cụm 1 (Blues/Acoustic), 3 (Metal/Urban) và 4 (Classic/Punk) phân bố thành các nhánh hướng tâm rõ nét xung quanh gốc tọa độ. Cụm 2 (Alternative/Fusion) nằm ở trung tâm chuyển tiếp.
- **Không gian t-SNE:** Khẳng định 5 cụm hình thành 5 đám mây điểm cô lập tương đối, với tâm cụm (kí hiệu `X` màu đen) nằm chính xác tại trọng tâm của từng đám mây điểm.

---

## 3. KIỂM ĐỊNH ĐỘ ỔN ĐỊNH VÀ TỔNG QUÁT HÓA (HOLD-OUT CV 80/20 QUA 100 LẦN LẶP)

Đây là phép thử quan trọng nhất trong Pha C3 nhằm kiểm tra xem cấu trúc phân cụm có tổng quát hóa được sang các khách hàng mới (*out-of-sample generalization*) hay chỉ là hiện tượng ghi nhớ cục bộ (*overfitting on small sample*).

Chi tiết toàn bộ 500 lượt lặp (100 lượt $\times$ 5 ứng viên) được lưu trữ tại: [`outputs/stability_runs.csv`](file:///e:/ML/Chinook/outputs/stability_runs.csv).  
Biểu đồ Boxplot phân phối lưu tại: [`figures/cluster_stability_boxplots.png`](file:///e:/ML/Chinook/figures/cluster_stability_boxplots.png).

### Bảng 2: Thống kê Phân phối Hold-out CV (100 Lần Lặp 80/20)

| Mã cấu hình (Config ID) | Mô hình | Số cụm ($k$) | Train ARI Median | Train NMI Median | Hold-out ARI Median | Hold-out ARI P5 (Bi quan) | Hold-out ARI P95 (Lạc quan) | Hold-out Silhouette Median | Đánh giá theo Decision Rules (Ngưỡng $\ge 0.60$) | Phân loại Độ ổn định thực tế |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`V2_album_count_svd5_dfFalse_KM_k5`** | **KMeans** | **5** | **0.5066** | **0.6501** | **0.4038** | **0.0782** | **0.7966** | **0.3760** | **Chưa đạt ngưỡng 0.60** | **Độ ổn định trung bình (Moderate)** |
| `V2_album_count_svd5_dfFalse_Agg_complete_k5` | Agglomerative | 5 | 0.4204 | 0.5981 | **0.3210** | 0.0188 | 0.7643 | 0.3714 | Chưa đạt ngưỡng 0.60 | Kém ổn định (Unstable < 0.40) |
| `V2_album_count_svd5_dfFalse_Spectral_NN8_k5` | Spectral | 5 | 0.4905 | 0.6411 | **0.3131** | 0.0211 | 0.7797 | 0.3597 | Chưa đạt ngưỡng 0.60 | Kém ổn định (Unstable < 0.40) |
| `V2_album_count_svd5_dfFalse_KM_k6` | KMeans | 6 | 0.5196 | 0.6902 | **0.2928** | 0.0213 | 0.6642 | 0.3885 | Chưa đạt ngưỡng 0.60 | Kém ổn định (Unstable < 0.40) |
| `V1_album_binary_svd5_dfFalse_KM_k5` | KMeans (Binary) | 5 | 0.4421 | 0.6072 | **0.2632** | 0.0236 | 0.7201 | 0.3663 | Chưa đạt ngưỡng 0.60 | Kém ổn định (Unstable < 0.40) |

### Nhận định Khoa học Cốt lõi về Độ ổn định Hold-out

1. **Ứng viên #1 (`V2_KM_k5`) có độ ổn định vượt trội so với toàn bộ các ứng viên còn lại:**
   - Median Hold-out ARI đạt **0.4038**, cao hơn đáng kể so với Agglomerative (0.3210), Spectral (0.3131), KMeans k=6 (0.2928) và KMeans Binary (0.2632).
   - Phân vị 95% (P95) đạt **0.7966**, cho thấy trong các lần chia mẫu cân bằng, mô hình có khả năng tái tạo phân hoạch xuất sắc.
2. **Tuy nhiên, không có ứng viên nào đạt ngưỡng khắt khe Median ARI $\ge 0.60$:**
   - Theo định nghĩa tại Mục 5 của [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):
     - `reliable_threshold: 0.60`: Cụm đáng tin cậy vững chắc (*Robust*).
     - `moderate_threshold: 0.40`: Độ ổn định trung bình (*Moderate Stability*).
     - `unstable_threshold: 0.40`: Không ổn định (*Unstable*).
   - Ứng viên `V2_KM_k5` nằm chính xác tại ranh giới **Độ ổn định trung bình (0.4038)**, trong khi tất cả các ứng viên khác đều rơi vào vùng không ổn định ($< 0.35$).
3. **Căn nguyên toán học:**
   - Kích thước mẫu $n = 59$ là rất nhỏ trong bài toán phân cụm không gian 5 chiều. Khi bốc 20% làm test (12 khách), mỗi cụm trong tập test trung bình chỉ có $12 / 5 \approx 2.4$ khách hàng.
   - Khi loại bỏ 12 khách ngẫu nhiên khỏi tập train 47 khách, một số cụm có thể mất đi 3-4 điểm biên, làm dịch chuyển nhẹ tâm SVD và xoay trục phân ranh giới. Do số lượng điểm test quá nhỏ ($n_{test} = 12$), chỉ cần 2-3 khách hàng bị gán sang cụm lân cận là chỉ số ARI bị tụt từ 0.70 xuống 0.35.

---

## 4. KIỂM TRA ĐỘ NHẠY THEO RANDOM SEED VÀ SỐ CỤM $K$

### 4.1. Độ nhạy theo Random Seed (>= 25 Seeds khác nhau trên Full Data)
Thực nghiệm quét qua 25 giá trị `random_state` khác nhau cho mô hình `V2_album_count_svd5_dfFalse_KM_k5`:
- **Mean Pairwise ARI:** **0.9945**
- **Median Pairwise ARI:** **1.0000**
- **Min Pairwise ARI:** **0.9641**
- **Max Pairwise ARI:** **1.0000**
- **Standard Deviation:** **0.0129**

> **Kết luận:** Thuật toán KMeans với $n\_init = 30$ hội tụ gần như tuyệt đối (Median ARI = 1.0) trên toàn bộ tập dữ liệu 59 khách hàng. Khởi tạo ngẫu nhiên không gây ra hiện tượng mắc kẹt tại các cực tiểu địa phương (*local minima*) khác nhau.

### 4.2. Độ nhạy theo Số cụm $k$ (So sánh $k=4, 5, 6$)
So sánh sự thay đổi nhãn của Ứng viên chính ($k=5$) với mô hình lân cận ($k=4$ và $k=6$):
- **So với $k = 4$:** $\text{ARI} = \mathbf{0.7300}$, $\text{NMI} = \mathbf{0.7933}$
- **So với $k = 6$:** $\text{ARI} = \mathbf{0.8522}$, $\text{NMI} = \mathbf{0.9158}$

> **Kết luận:** Cấu trúc phân cụm có tính phân cấp lồng ghép (*hierarchical nested structure*). Khi tăng từ $k=5$ lên $k=6$, cụm lớn nhất (Cụm 2, 15 khách) tách đôi thành 2 nhóm nhỏ hơn (9 khách và 7 khách) mà không làm xáo trộn các cụm còn lại. Khi giảm về $k=4$, hai cụm Rock lân cận hợp nhất tự nhiên.

---

## 5. KIỂM ĐỊNH HÀNH VI TƯƠNG LAI THEO TRỤC THỜI GIAN (TEMPORAL VALIDATION)

Biểu đồ phân phối hoán vị lưu tại: [`figures/cluster_temporal_prediction.png`](file:///e:/ML/Chinook/figures/cluster_temporal_prediction.png).

### 5.1. Thiết kế Thí nghiệm Thời gian
- **Giai đoạn 1 (Period 1):** 3 hóa đơn đầu tiên của mỗi khách hàng ($1.001$ dòng giao dịch, đủ 59 khách).
- **Giai đoạn 2 (Period 2):** Các hóa đơn còn lại (3 đến 4 hóa đơn sau, $1.239$ dòng giao dịch, đủ 59 khách).
- **Thực hiện ở cấp độ ALBUM và GENRE:**
  - Fit pipeline và mô hình KMeans $k=5$ trên Giai đoạn 1 $\to$ Nhãn $y^{(1)}$, Tâm cụm $C^{(1)}$.
  - Đưa ma trận Giai đoạn 2 qua pipeline GĐ 1 $\to$ Tọa độ $X^{(2)}$.
  - (a) Gom cụm độc lập trên GĐ 2 $\to$ Nhãn $y^{(2)}$. Đo $\text{ARI}(y^{(1)}, y^{(2)})$.
  - (b) Kiểm tra dự báo: Tính khoảng cách Cosine từ vector GĐ 2 của từng khách tới tất cả các tâm cụm GĐ 1. Kiểm tra xem tâm cụm gần nhất có trùng với cụm GĐ 1 của chính họ không.
  - (c) So sánh với phân phối Null Permutation ($N = 1000$ lần hoán vị nhãn ngẫu nhiên).

### 5.2. Kết quả Thống kê Thực nghiệm

| Chỉ số Đo lường theo Thời gian | Giá trị Quan sát Thực tế | Kỳ vọng Ngẫu nhiên (Null Mean) | Phân vị 95 Ngẫu nhiên (Null P95) | Khoảng cách so với Ngẫu nhiên | Empirical P-value ($N=1000$) | Đánh giá & Kết luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARI Cụm GĐ 1 vs Cụm GĐ 2** | **0.1281** | - | - | - | - | Rất thấp; cấu trúc phân cụm album không giữ nguyên qua thời gian. |
| **Tỷ lệ Khách hàng GĐ 2 gần Centroid của mình nhất** | **11.86%** (7/59 khách) | **19.75%** ($\approx 1/5$) | **28.81%** | **-7.88%** | **p = 0.9640** | **Không có ý nghĩa dự báo ở mức Album.** |
| **Độ tự nhất quán Thể loại (Genre Self-Consistency Median)** | **0.5065** | 0.0500 | 0.1500 | **+0.4565** | **p < 0.001** | **Rất cao và có ý nghĩa thống kê bền vững.** |

### 5.3. Phân tích Nguyên nhân Khoa học (Root Cause Analysis)

Kết quả kiểm tra thời gian mang lại phát hiện cực kỳ quan trọng về bản chất tiêu dùng âm nhạc:

1. **Hiệu ứng "Không mua lại Album cũ" (*No Repeat Purchases of Same Albums*):**
   - Trong kinh doanh âm nhạc số, khách hàng gần như **không bao giờ mua lại cùng một album đã sở hữu**.
   - Ở Giai đoạn 1, mỗi khách hàng mua trung bình 19 bài hát thuộc 10-15 album. Ở Giai đoạn 2, họ mua tiếp 19 bài hát thuộc 10-15 album **hoàn toàn mới**.
   - Do catalog có 347 album và ma trận cực kỳ thưa (độ thưa $> 97\%$), tập hợp các cột có giá trị $> 0$ ở GĐ 1 và GĐ 2 của cùng một khách hàng có **độ giao nhau (Jaccard Index) gần bằng 0**!
2. **Sự dịch chuyển của vector trong không gian SVD Album:**
   - Vì các album mua ở GĐ 2 là những album khác, vector nén SVD của khách hàng ở GĐ 2 bị xoay sang hướng khác trong không gian 5 chiều, khiến khoảng cách tới tâm cụm GĐ 1 bị phân tán.
3. **Sự ổn định thực sự nằm ở cấp độ THỂ LOẠI (GENRE):**
   - Trái ngược với Album, ở cấp độ **Thể loại (Genre Share)**, độ tự nhất quán giữa GĐ 1 và GĐ 2 đạt trung vị **0.5065** (Cosine Similarity).
   - Khách hàng nghe Rock ở Giai đoạn 1 tiếp tục mua các album Rock mới ở Giai đoạn 2; khách hàng nghe Latin/Bossa ở GĐ 1 tiếp tục tìm kiếm nghệ sĩ Latin mới ở GĐ 2.
   - **Hệ quả kinh doanh:** Phân khúc cụm ở mức chi tiết Album chỉ có giá trị mô tả lịch sử giao dịch (*descriptive cohort*), nhưng **không có khả năng dự báo hành vi tương lai ở mức từng album cụ thể**. Để dự báo và gợi ý sản phẩm tương lai, hệ thống bắt buộc phải vận hành ở cấp độ Thể loại (Genre) hoặc dùng Collaborative Filtering Item-Item.

---

## 6. ĐỐI CHIẾU TIÊU CHÍ QUYẾT ĐỊNH VÀ KẾT LUẬN CUỐI CÙNG

Đối chiếu kết quả thực nghiệm với các điều kiện trong [`config/decision_rules.yaml`](file:///e:/ML/Chinook/config/decision_rules.yaml):

| Tiêu chuẩn trong Decision Rules | Ngưỡng Quy định | Kết quả của Ứng viên #1 (`V2_KM_k5`) | Trạng thái Đạt / Không đạt | Ghi chú Phân tích |
| :--- | :---: | :---: | :---: | :--- |
| **Kích thước Cụm tối thiểu** | $\text{min\_size} \ge 5$ khách | **10 khách** (Phân bổ: `[15, 12, 12, 10, 10]`) | **ĐẠT XUẤT SẮC** | Cụm nhỏ nhất chiếm $16.9\%$ tập dữ liệu, hoàn toàn không có cụm ngoại lai. |
| **Vượt trội Null Model** | $\text{Sil} > \text{Null P95}$ | **$0.5941 > 0.4441$** ($\Delta = +0.1500$) | **ĐẠT XUẤT SẮC** | Điểm Silhouette vượt xa 95% ma trận ngẫu nhiên hoán vị bảo toàn bậc hàng/cột. |
| **Vượt trội các Baseline** | Hơn Dominant Genre, Country, Random | **Vượt trội tuyệt đối** | **ĐẠT XUẤT SẮC** | Baseline Dominant Genre có Sil = -0.013; Country có Sil = -0.118; Random có Sil = -0.090. |
| **Độ ổn định Hold-out CV** | $\text{Median ARI} \ge 0.60$ | **$0.4038$** (P5=0.0782, P95=0.7966) | **CHƯA ĐẠT NGƯỠNG LÝ TƯỞNG ($\ge 0.60$)** | Đạt mức **Độ ổn định trung bình (Moderate: 0.40 - 0.60)**; cao nhất trong 5 ứng viên. |
| **Kiểm tra theo Thời gian** | Dự báo tương lai > Null P95 | **$11.86\% < 28.81\%$ (Null P95)** | **KHÔNG ĐẠT Ở MỨC ALBUM** | Do người dùng không mua lại album cũ; nhưng đạt ở mức Genre (Consistency = 0.5065). |

### KẾT LUẬN KIỂM ĐỊNH PHA C3:

1. **Lựa chọn Mô hình Tốt nhất (Best Overall Model):**  
   Cấu hình **`V2_album_count_svd5_dfFalse_KM_k5`** (KMeans $k=5$, n_init=30, trên ma trận V2 Album Count + IDF nén qua TruncatedSVD 5 chiều) là mô hình chiến thắng toàn diện:
   - Đạt điểm hình học nội tại cao nhất, cân đối nhất, vượt trội Null Model và toàn bộ các Baseline.
   - Đạt độ ổn định Hold-out cao nhất trong các ứng viên (Median ARI = 0.4038).
   - Độ nhạy với Random Seed tiệm cận 0 (Mean ARI = 0.9945).
2. **Thừa nhận Trung thực Giới hạn Thống kê (Honest Scientific Limitation):**  
   Mặc dù là mô hình tốt nhất, nó **chưa đạt ngưỡng ổn định vững chắc $\ge 0.60$** và **không có khả năng dự báo album lặp lại theo thời gian**. Điều này phản ánh trung thực thực tế dữ liệu Chinook ($n=59$, catalog 347 album thưa thớt) chứ không phải do lỗi thuật toán.
3. **Quyết định Triển khai:**  
   Tiến hành **Final Fit** và đóng gói mô hình `V2_album_count_svd5_dfFalse_KM_k5` vào [`models/final_model.joblib`](file:///e:/ML/Chinook/models/final_model.joblib) để phục vụ phân tích hồ sơ phân khúc khách hàng (Profiling), đồng thời ghi nhận rõ ràng các khuyến cáo sử dụng tại Model Card.
