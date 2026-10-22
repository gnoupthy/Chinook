# BÁO CÁO HỒ SƠ PHÂN KHÚC KHÁCH HÀNG (CUSTOMER CLUSTER PROFILES)
**Giai đoạn:** Pha C3 - Diễn giải Phân khúc & Xây dựng Chân dung Khách hàng  
**Dự án:** Phân khúc Khách hàng Chinook (Customer Clustering)  
**Mô hình áp dụng:** `V2_album_count_svd5_dfFalse_KM_k5` (KMeans $k=5$, SVD 5D, V2 Album Count + IDF)  
**File dữ liệu kết quả:** [`outputs/customer_clusters.csv`](file:///e:/ML/Chinook/outputs/customer_clusters.csv)  
**Biểu đồ hồ sơ Lift:** [`figures/cluster_profiles_lift.png`](file:///e:/ML/Chinook/figures/cluster_profiles_lift.png)  
**File mã nguồn thực hiện:** [`src/profiling.py`](file:///e:/ML/Chinook/src/profiling.py)  

---

## 1. TỔNG QUAN PHÂN BỔ 5 PHÂN KHÚC KHÁCH HÀNG

Phân cụm trên không gian nén SVD 5 chiều từ ma trận Album Count + IDF chia 59 khách hàng thành **5 phân khúc cân đối**, không có cụm nào bị cô lập hay mang tính ngoại lai (< 5 khách):

```
Phân bổ 59 khách hàng theo 5 Phân khúc:
├── Cụm 0 (Eclectic Soundtrack, Sci-Fi & Vintage Grooves):  12 khách (20.3%)
├── Cụm 1 (Soulful Blues, Reggae & Acoustic Connoisseurs):   10 khách (16.9%)
├── Cụm 2 (Global Fusion, Alternative & Dance Explorers):    15 khách (25.4%)
├── Cụm 3 (Hard Rock, Heavy Metal & Urban Rhythm Lovers):    10 khách (16.9%)
└── Cụm 4 (Classic Blues-Rock, Punk & Symphonic Fans):       12 khách (20.3%)
Tỷ lệ cân đối (Min / Max): 10 / 15 = 0.6667 (Rất cân đối)
```

> **Nguyên tắc lọc thống kê (Statistical Thresholding Rule):**  
> Nhằm ngăn chặn hiện tượng Lift bị thổi phồng do cỡ mẫu nhỏ (ví dụ 1 album chỉ có 1 khách mua duy nhất trong cả cửa hàng thì Lift có thể bị vọt lên nhân tạo), **TOÀN BỘ các bảng Lift dưới đây bắt buộc phải thỏa mãn ngưỡng tối thiểu: Mục đó phải có $\ge 3$ khách hàng trong cụm mua**.

---

## 2. HỒ SƠ CHI TIẾT TỪNG PHÂN KHÚC KHÁCH HÀNG (CLUSTER PERSONAS)

---

### PHÂN KHÚC 0: ECLECTIC SOUNDTRACK, SCI-FI & VINTAGE GROOVES
**Quy mô:** **12 khách hàng** (chiếm **20.3%** tổng tập khách hàng)  
**Đặc trưng tài chính:** Chi tiêu TB = **$41.29** (Cao nhất trong 5 cụm) | Số track mua TB = **38.0** | Hóa đơn TB = **7.0**  
**Quốc gia chủ đạo:** Brazil (2), Czech Republic (2), USA (2), Canada (1), France (1), Germany (1), Chile (1), Hungary (1), Ireland (1)  

#### 1. Thể loại Âm nhạc Đặc trưng (Top Genres by Lift - Ngưỡng $\ge 3$ khách):
| Thể loại (Genre) | Số khách trong cụm | Độ phủ trong cụm (%) | Tỷ lệ toàn thể (%) | Chỉ số Lift | Nhận xét hành vi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Easy Listening** | 4 | 33.3% | 6.8% | **4.92x** | Sở thích độc quyền, tập trung cao gấp 4.9 lần bình quân. |
| **Science Fiction** | 4 | 33.3% | 6.8% | **4.92x** | Nhạc phim khoa học viễn tưởng và audio drama. |
| **Rock And Roll** | 3 | 25.0% | 6.8% | **3.69x** | Rock & Roll cổ điển thập niên 50-60s. |
| **Soundtrack** | 4 | 33.3% | 15.3% | **2.19x** | Nhạc nền phim điện ảnh và truyền hình. |
| **TV Shows** | 7 | 58.3% | 32.2% | **1.81x** | Hơn một nửa khách hàng trong nhóm sưu tập các series truyền hình. |

#### 2. Top Nghệ sĩ Đặc trưng (Top Artists by Lift - Ngưỡng $\ge 3$ khách):
| Nghệ sĩ (Artist) | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :---: | :---: | :---: |
| **Milton Nascimento** | 5 | 41.7% | **4.92x** |
| **Frank Sinatra** | 4 | 33.3% | **4.92x** |
| **Funk Como Le Gusta** | 4 | 33.3% | **4.92x** |
| **Battlestar Galactica** | 4 | 33.3% | **2.81x** |
| **Heroes** | 3 | 25.0% | **2.46x** |

#### 3. Top Album Đặc trưng (Top Albums by Lift - Ngưỡng $\ge 3$ khách):
| Tên Album | Nghệ sĩ | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Milton Nascimento E Fernando Brant** | Milton Nascimento | 4 | 33.3% | **4.92x** |
| **Battlestar Galactica (Classic), Season 1**| Battlestar Galactica | 3 | 25.0% | **4.92x** |
| **Heroes, Season 1** | Heroes | 3 | 25.0% | **4.92x** |

> **Chân dung Persona:** Nhóm khách hàng hoài cổ và yêu thích nghe nhìn đa phương tiện. Họ có mức chi tiêu cao nhất cửa hàng ($41.29), thích nhạc phim truyền hình (Battlestar Galactica, Heroes), nhạc cổ điển lãng mạn (Frank Sinatra) và âm nhạc dân gian đương đại Nam Mỹ (Milton Nascimento).

---

### PHÂN KHÚC 1: SOULFUL BLUES, REGGAE & ACOUSTIC CONNOISSEURS
**Quy mô:** **10 khách hàng** (chiếm **16.9%** tổng tập khách hàng)  
**Đặc trưng tài chính:** Chi tiêu TB = **$39.12** | Số track mua TB = **38.0** | Hóa đơn TB = **7.0**  
**Quốc gia chủ đạo:** Brazil (2), Germany (2), Austria (1), USA (1), France (1), Netherlands (1), Poland (1), Portugal (1)  

#### 1. Thể loại Âm nhạc Đặc trưng (Top Genres by Lift - Ngưỡng $\ge 3$ khách):
| Thể loại (Genre) | Số khách trong cụm | Độ phủ trong cụm (%) | Tỷ lệ toàn thể (%) | Chỉ số Lift | Nhận xét hành vi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Soundtrack** | 5 | 50.0% | 15.3% | **3.28x** | Tập trung rất cao vào các bản hòa âm điện ảnh. |
| **Reggae** | 6 | 60.0% | 20.3% | **2.95x** | 60% khách hàng trong cụm nghe Reggae. |
| **Pop** | 5 | 50.0% | 22.0% | **2.27x** | Tỷ lệ nghe Pop cao gấp đôi toàn thể. |
| **Blues** | 6 | 60.0% | 39.0% | **1.54x** | Dòng nhạc cốt lõi với 60% khách hàng sở hữu. |
| **Classical** | 3 | 30.0% | 23.7% | **1.26x** | Nhạc cổ điển giao hưởng. |

#### 2. Top Nghệ sĩ Đặc trưng (Top Artists by Lift - Ngưỡng $\ge 3$ khách):
| Nghệ sĩ (Artist) | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :---: | :---: | :---: |
| **Frank Zappa & Captain Beefheart** | 4 | 40.0% | **5.90x** |
| **Marcos Valle** | 4 | 40.0% | **5.90x** |
| **Incognito** | 4 | 40.0% | **5.90x** |
| **UB40** | 4 | 40.0% | **4.72x** |
| **Eric Clapton** | 5 | 50.0% | **2.95x** |

#### 3. Top Album Đặc trưng (Top Albums by Lift - Ngưỡng $\ge 3$ khách):
| Tên Album | Nghệ sĩ | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Bongo Fury** | Frank Zappa & Captain Beefheart | 4 | 40.0% | **5.90x** |
| **UB40 The Best Of - Volume Two** | UB40 | 4 | 40.0% | **4.72x** |
| **Unplugged** | Eric Clapton | 4 | 40.0% | **3.28x** |

> **Chân dung Persona:** Nhóm khách hàng sành điệu, có gu âm nhạc mộc (Acoustic), yêu thích chất giọng trầm ấm của Blues (Eric Clapton Unplugged), nhịp điệu thư thái của Reggae (UB40) và sự phá cách của Jazz-Fusion (Frank Zappa, Incognito).

---

### PHÂN KHÚC 2: GLOBAL FUSION, ALTERNATIVE & DANCE EXPLORERS
**Quy mô:** **15 khách hàng** (chiếm **25.4%** tổng tập khách hàng - Phân khúc lớn nhất)  
**Đặc trưng tài chính:** Chi tiêu TB = **$38.89** | Số track mua TB = **37.9** | Hóa đơn TB = **6.9**  
**Quốc gia chủ đạo:** Canada (3), USA (3), Denmark (1), Norway (1), Finland (1), France (1), Germany (1), United Kingdom (1), India (1), Argentina (1), Australia (1)  

#### 1. Thể loại Âm nhạc Đặc trưng (Top Genres by Lift - Ngưỡng $\ge 3$ khách):
| Thể loại (Genre) | Số khách trong cụm | Độ phủ trong cụm (%) | Tỷ lệ toàn thể (%) | Chỉ số Lift | Nhận xét hành vi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Alternative** | 3 | 20.0% | 6.8% | **2.95x** | Alternative Rock độc lập. |
| **Electronica/Dance** | 5 | 33.3% | 11.9% | **2.81x** | Nhạc điện tử, tiết tấu sôi động. |
| **Bossa Nova** | 4 | 26.7% | 11.9% | **2.25x** | Giai điệu Bossa Nova tinh tế. |
| **World** | 5 | 33.3% | 15.3% | **2.19x** | Âm nhạc thế giới đa văn hóa. |
| **Hip Hop/Rap** | 4 | 26.7% | 15.3% | **1.75x** | Nhịp điệu đường phố hiện đại. |

#### 2. Top Nghệ sĩ Đặc trưng (Top Artists by Lift - Ngưỡng $\ge 3$ khách):
| Nghệ sĩ (Artist) | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :---: | :---: | :---: |
| **O Rappa** | 3 | 20.0% | **3.93x** |
| **Jorge Ben** | 4 | 26.7% | **3.93x** |
| **Joe Satriani** | 4 | 26.7% | **3.93x** |
| **Gilberto Gil** | 4 | 26.7% | **2.62x** |
| **Caetano Veloso** | 4 | 26.7% | **2.36x** |

#### 3. Top Album Đặc trưng (Top Albums by Lift - Ngưỡng $\ge 3$ khách):
| Tên Album | Nghệ sĩ | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Surfing with the Alien** | Joe Satriani | 4 | 26.7% | **3.93x** |
| **Chill: Brazil (Disc 2)** | Đa nghệ sĩ | 3 | 20.0% | **3.93x** |
| **As Canções de Eu Tu Eles** | Gilberto Gil | 3 | 20.0% | **2.95x** |

> **Chân dung Persona:** Nhóm khách hàng trẻ trung, cởi mở và thích khám phá các dòng nhạc quốc tế. Họ không bị bó hẹp trong Rock truyền thống mà đam mê World Music, Bossa Nova, Electronica và Guitar solo điêu luyện (Joe Satriani).

---

### PHÂN KHÚC 3: HARD ROCK, HEAVY METAL & URBAN RHYTHM LOVERS
**Quy mô:** **10 khách hàng** (chiếm **16.9%** tổng tập khách hàng)  
**Đặc trưng tài chính:** Chi tiêu TB = **$39.72** | Số track mua TB = **38.0** | Hóa đơn TB = **7.0**  
**Quốc gia chủ đạo:** USA (2), Brazil (2), Czech Republic (1), France (1), Germany (1), Portugal (1), Spain (1), Sweden (1)  

#### 1. Thể loại Âm nhạc Đặc trưng (Top Genres by Lift - Ngưỡng $\ge 3$ khách):
| Thể loại (Genre) | Số khách trong cụm | Độ phủ trong cụm (%) | Tỷ lệ toàn thể (%) | Chỉ số Lift | Nhận xét hành vi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hip Hop/Rap** | 4 | 40.0% | 15.3% | **2.62x** | Xuất hiện dày đặc kết hợp với Rock nặng. |
| **Reggae** | 5 | 50.0% | 20.3% | **2.46x** | Sự giao thoa thú vị giữa Reggae và Rock. |
| **Drama** | 3 | 30.0% | 23.7% | **1.26x** | Các đoạn hội thoại / kịch nói. |
| **Jazz** | 6 | 60.0% | 54.2% | **1.11x** | Jazz kết cấu phức tạp. |
| **Metal** | 10 | **100.0%** | 93.2% | **1.07x** | 100% khách hàng trong cụm nghe Heavy Metal. |

#### 2. Top Nghệ sĩ Đặc trưng (Top Artists by Lift - Ngưỡng $\ge 3$ khách):
| Nghệ sĩ (Artist) | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :---: | :---: | :---: |
| **David Coverdale** | 4 | 40.0% | **5.90x** |
| **Vinícius De Moraes** | 4 | 40.0% | **5.90x** |
| **Velvet Revolver** | 5 | 50.0% | **5.90x** |
| **Page & Plant** | 4 | 40.0% | **3.93x** |
| **Guns N' Roses** | 5 | 50.0% | **2.46x** |

#### 3. Top Album Đặc trưng (Top Albums by Lift - Ngưỡng $\ge 3$ khách):
| Tên Album | Nghệ sĩ | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Contraband** | Velvet Revolver | 4 | 40.0% | **5.90x** |
| **Into The Light** | David Coverdale | 3 | 30.0% | **4.43x** |
| **Use Your Illusion I** | Guns N' Roses | 4 | 40.0% | **2.95x** |

> **Chân dung Persona:** Tín đồ của dòng Hard Rock thập niên 80-90 và Metal hiện đại (Guns N' Roses, Velvet Revolver, Whitesnake/David Coverdale). Họ yêu thích nhịp điệu mạnh mẽ, tiếng guitar điện distortion bùng nổ kết hợp với chất Rock đường phố cá tính.

---

### PHÂN KHÚC 4: CLASSIC BLUES-ROCK, PUNK & SYMPHONIC FANS
**Quy mô:** **12 khách hàng** (chiếm **20.3%** tổng tập khách hàng)  
**Đặc trưng tài chính:** Chi tiêu TB = **$38.45** | Số track mua TB = **38.0** | Hóa đơn TB = **7.0**  
**Quốc gia chủ đạo:** USA (3), Belgium (1), Brazil (1), Canada (1), France (1), Germany (1), United Kingdom (1), India (1), Italy (1), Portugal (1)  

#### 1. Thể loại Âm nhạc Đặc trưng (Top Genres by Lift - Ngưỡng $\ge 3$ khách):
| Thể loại (Genre) | Số khách trong cụm | Độ phủ trong cụm (%) | Tỷ lệ toàn thể (%) | Chỉ số Lift | Nhận xét hành vi |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical** | 5 | 41.7% | 23.7% | **1.76x** | Nhạc cổ điển giao hưởng thính phòng. |
| **Blues** | 7 | 58.3% | 39.0% | **1.50x** | Blues Rock thuần túy. |
| **R&B/Soul** | 5 | 41.7% | 28.8% | **1.45x** | Tiết tấu Soul trầm ấm. |
| **Alternative & Punk** | 11 | **91.7%** | 84.7% | **1.08x** | Hơn 91% khách hàng mua Punk/Alternative. |
| **Metal** | 12 | **100.0%** | 93.2% | **1.07x** | 100% khách hàng sở hữu album Metal. |

#### 2. Top Nghệ sĩ Đặc trưng (Top Artists by Lift - Ngưỡng $\ge 3$ khách):
| Nghệ sĩ (Artist) | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :---: | :---: | :---: |
| **Aerosmith** | 4 | 33.3% | **4.92x** |
| **Lulu Santos** | 4 | 33.3% | **4.92x** |
| **Marillion** | 4 | 33.3% | **4.92x** |
| **Smashing Pumpkins** | 4 | 33.3% | **3.93x** |
| **Green Day** | 6 | 50.0% | **1.97x** |

#### 3. Top Album Đặc trưng (Top Albums by Lift - Ngưỡng $\ge 3$ khách):
| Tên Album | Nghệ sĩ | Số khách trong cụm | Độ phủ (%) | Chỉ số Lift |
| :--- | :--- | :---: | :---: | :---: |
| **Big Ones** | Aerosmith | 4 | 33.3% | **4.92x** |
| **Misplaced Childhood** | Marillion | 4 | 33.3% | **4.92x** |
| **Rotten Apples** | Smashing Pumpkins | 4 | 33.3% | **3.93x** |
| **International Superhits** | Green Day | 5 | 41.7% | **2.05x** |

> **Chân dung Persona:** Những người hâm mộ trung thành của Rock kinh điển thập niên 70-90 (Aerosmith, Marillion) kết hợp với năng lượng của Punk Rock (Green Day, Smashing Pumpkins). Họ có độ phủ thể loại rất rộng, đồng thời có sở thích thưởng thức thêm giao hưởng thính phòng (Classical).

---

## 3. KIỂM ĐỊNH PHI THAM SỐ KHÁC BIỆT NHÂN KHẨU HỌC & HÀNH VI

Để làm sáng tỏ tính độc lập của các biến giải thích bổ sung (không tham gia vào quá trình gom cụm), chúng tôi thực hiện các kiểm định thống kê phi tham số với mức ý nghĩa $\alpha = 0.05$:

### Bảng 2: Kết quả Kiểm định Phi tham số giữa 5 Phân khúc

| Tên biến kiểm định | Loại biến | Phương pháp kiểm định | Thống kê kiểm định | P-value | Ý nghĩa thống kê ($\alpha=0.05$) | Thước đo Kích thước Hiệu ứng (Effect Size) | Đánh giá Mức độ Hiệu ứng |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Tổng Chi tiêu (TotalSpent)** | Liên tục | Kruskal-Wallis H-test | $H = 4.8250$ | **$p = 0.3057$** | **KHÔNG có ý nghĩa** | $\eta^2 = 0.0153$ | Hiệu ứng rất nhỏ (Small) |
| **Tổng Số Track đã mua** | Rời rạc | Kruskal-Wallis H-test | $H = 2.9333$ | **$p = 0.5690$** | **KHÔNG có ý nghĩa** | $\eta^2 = 0.0000$ | Không có khác biệt |
| **Số lượng Hóa đơn (InvoiceCount)** | Rời rạc | Kruskal-Wallis H-test | $H = 2.9333$ | **$p = 0.5690$** | **KHÔNG có ý nghĩa** | $\eta^2 = 0.0000$ | Không có khác biệt |
| **Phân bổ Địa lý (Region)** | Phân loại | Chi-Square Test ($df=16$) | $\chi^2 = 7.0320$ | **$p = 0.9726$** | **KHÔNG có ý nghĩa** | $\text{Cramér's } V = 0.1726$ | Hiệu ứng trung bình (ngẫu nhiên) |

### Nhận định Trung thực và Khách quan (Không P-hacking):

1. **Khách hàng Chinook hoàn toàn đồng nhất về hành vi tổng lượng (Volume Uniformity):**
   - Mọi khách hàng trong cơ sở dữ liệu mẫu Chinook đều có lịch sử mua hàng cực kỳ đều đặn: trung bình chính xác $38$ bài hát, phân bổ trong $6$ hoặc $7$ hóa đơn, với tổng số tiền dao động từ $\$37.62$ đến $\$49.62$.
   - Kiểm định Kruskal-Wallis cho thấy không hề có phân khúc "khách hàng VIP chi tiêu khủng" hay "khách hàng ít tương tác". Mọi cụm đều có chi tiêu bình quân xấp xỉ $\$38.5 - \$41.3$ ($p = 0.31$).
2. **Âm nhạc có tính xuyên quốc gia (No Geographic Music Bias):**
   - Kiểm định $\chi^2$ với phân bổ vùng địa lý cho $p = 0.9726$. Người nghe Heavy Metal hay Latin phân bố đều ở cả Bắc Mỹ, Châu Âu và Nam Mỹ. Vị trí địa lý không quyết định gu âm nhạc của khách hàng.
3. **Ý nghĩa Thực tiễn:**
   - **Sự khác biệt cốt lõi và duy nhất giữa các phân khúc là THỊ HIẾU DANH MỤC (TASTE PREFERENCES)**: Họ mua cái gì (Aerosmith vs Velvet Revolver vs Frank Sinatra vs Eric Clapton), chứ không phải họ mua bao nhiêu tiền.

---

## 4. GHI CHÚ VỀ ĐỘ TIN CẬY VÀ KHUYẾN NGHỊ BƯỚC SANG PHA D

1. **Lưu ý về Cỡ mẫu Nhỏ ($n = 59$):**  
   Mỗi cụm chỉ có từ 10 đến 15 khách hàng. Do đó, các giá trị Lift cần được diễn giải như những **chỉ báo xu hướng ưu tiên (*propensity indicators*)** phục vụ việc xây dựng gói sản phẩm thử nghiệm (A/B testing bundles), chứ không phải quy luật tất yếu tuyệt đối.
2. **Khuyến nghị cho Pha D (Bundle Design & Marketing Activation):**
   - Không thiết kế ưu đãi dựa trên giảm giá theo tổng hóa đơn (vì chi tiêu các nhóm đều tương đương nhau).
   - Thiết kế các gói **Bundle Album đồng điệu về thể loại và nghệ sĩ** dựa trên Top Lift đã tìm thấy cho từng persona.
   - Kết hợp mô hình gợi ý theo độ tương tự Item-Item (Collaborative Filtering) để lấp đầy khoảng trống khi khách hàng chuyển sang mua các album mới ở các chu kỳ tương lai.
