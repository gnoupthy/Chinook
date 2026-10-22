# BÁO CÁO ĐẶT BÀI TOÁN KINH DOANH (BUSINESS PROBLEM DEFINITION)

**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album  
**Nguồn dữ liệu:** `data/Chinook_Sqlite.sqlite` (CSDL Chinook giai đoạn 2009 - 2013)  
**Tài liệu tham chiếu:** [reports/eda_findings.md](file:///e:/ML/Chinook/reports/eda_findings.md)  
**Nhà khoa học dữ liệu phụ trách:** Senior Data Scientist  

---

> [!IMPORTANT]
> **Cơ sở xuất phát từ kết quả Khám phá dữ liệu (EDA Findings):**  
> 1. **Khách hàng đồng nhất về quy mô hành vi:** Mỗi khách hàng trong hệ thống đều mua trung bình ~38 bài hát (CV = 0.69%), thực hiện đúng ~7 hóa đơn (CV = 1.86%), và chi tiêu ~39.47 USD (CV = 7.38%). Không thể phân khúc khách hàng theo giá trị chi tiêu (RFM).
> 2. **Sự khác biệt cốt lõi duy nhất:** Khách hàng **chỉ khác biệt ở Gu âm nhạc / Sở thích nội dung (Content & Taste Preference)**.
> 3. **Định nghĩa sự quan tâm (Interest):** Không có dữ liệu streaming/click; sự quan tâm được định nghĩa và lượng hóa duy nhất qua lịch sử mua hàng thực tế (bài hát, album, nghệ sĩ, thể loại).

---

## 1. Các quyết định kinh doanh cần hỗ trợ (Target Business Decisions)

Gom cụm khách hàng theo thị hiếu âm nhạc hướng tới mục tiêu chuyển đổi từ **tiếp thị đại trà (mass marketing)** sang **tiếp thị cá nhân hóa theo phân khúc (targeted lifecycle marketing)**, đồng thời tối ưu hóa doanh thu và vòng quay danh mục sản phẩm (catalog). Bốn quyết định kinh doanh trọng tâm bao gồm:

### Quyết định 1: Cá nhân hóa Email Marketing & Chiến dịch Khuyến mãi (Personalized Email & Promotional Campaigns)
- **Hiện trạng:** Cửa hàng gửi thông báo phát hành nhạc mới hoặc khuyến mãi giảm giá chung cho toàn bộ 59 khách hàng. Điều này dẫn đến sự suy giảm mức độ quan tâm (email fatigue), tỷ lệ hủy đăng ký (unsubscribe) cao và tỷ lệ mở/nhấp chuột thấp.
- **Quyết định hỗ trợ:** Phân loại khách hàng vào các cụm thị hiếu rõ rệt (ví dụ: *Hard Rock / Heavy Metal Enthusiasts*, *Latin & World Music Fans*, *Classic Rock & Blues Explorers*). Khi có nhạc mới hoặc chiến dịch theo mùa, hệ thống tự động lọc danh sách gửi email tương ứng với cụm yêu thích của khách hàng, hiển thị đúng các bài hát và album phù hợp với gu âm nhạc của họ.

### Quyết định 2: Đóng gói Album & Chiến lược Giá "Hoàn tất Album" (Album Bundling & Dynamic Album Completion)
- **Hiện trạng:** Phát hiện 6 trong EDA chỉ ra rằng khách hàng chủ yếu mua bài hát đơn lẻ (cherry-picking): tỷ lệ phủ album trung bình chỉ đạt **16.79%** số bài, và chỉ có **3.77%** số lần mua là mua trọn vẹn 100% album.
- **Quyết định hỗ trợ:** Khi một khách hàng đã mua từ 2 - 3 bài lẻ trong một album thuộc gu âm nhạc của cụm họ, hệ thống kích hoạt chính sách bán theo gói (Bundle): *"Hoàn tất Album với chiết khấu 25% cho các bài còn lại"* hoặc *"Mua Album A tặng bài hát độc quyền từ Album B cùng nghệ sĩ"*.

### Quyết định 3: Bán chéo thông minh (Cross-sell & Repertoire Discovery)
- **Hiện trạng:** Phát hiện 9 trong EDA xác nhận các dòng nhạc có mức độ đồng xuất hiện (co-occurrence) rất mạnh (ví dụ: Rock đi kèm Metal với 31 khách, Rock đi kèm Latin với 34 khách; người mua Iron Maiden có xu hướng mua thêm Metallica/U2).
- **Quyết định hỗ trợ:** Đưa ra các gợi ý bán chéo (Cross-sell recommendations) tại trang xác nhận đơn hàng hoặc trong hóa đơn điện tử (Invoice Receipt): đề xuất các nghệ sĩ và album có mức độ tương đồng cao trong cùng cụm thị hiếu nhưng khách hàng đó chưa từng mua.

### Quyết định 4: Ưu tiên Nhập danh mục & Cấp phép Bản quyền (Catalog Procurement & Licensing Prioritization)
- **Hiện trạng:** Ngân sách bản quyền của cửa hàng có giới hạn. Việc mua bản quyền phân phối bài hát/album mới từ các hãng đĩa (record labels) nếu không có định hướng sẽ dẫn đến tình trạng bài hát không có người mua (Dead Inventory). EDA cho thấy 43.36% bài hát trong catalog chưa từng được bán lần nào.
- **Quyết định hỗ trợ:** Xác định cụm khách hàng nào là "cụm nòng cốt" (đóng góp doanh thu lớn nhất hoặc có độ gắn kết cao nhất), từ đó ưu tiên đàm phán bản quyền các album/bài hát mới của những nghệ sĩ thuộc thị hiếu của cụm này.

---

## 2. Các câu hỏi kinh doanh cụ thể, Giá trị & Người dùng kết quả

| STT | Câu hỏi kinh doanh cụ thể (Business Questions) | Giá trị mang lại (Business Value) | Người dùng kết quả (Stakeholders) |
| :---: | :--- | :--- | :--- |
| **Q1** | **59 khách hàng của Chinook có thể chia thành những phân khúc gu âm nhạc (Taste Segments) riêng biệt nào, và đặc trưng nội dung (top nghệ sĩ, album, dòng nhạc chủ đạo) của từng phân khúc là gì?** | Chấm dứt tiếp thị mù quáng; định danh chân dung khách hàng sắc nét; tăng tỷ lệ tương tác (CTR, Open Rate) cho các chiến dịch marketing; tạo cơ sở dữ liệu nền tảng cho chiến lược CRM. | **Marketing Lead & CRM Manager** |
| **Q2** | **Khách hàng thuộc từng cụm gu âm nhạc có xu hướng mua kèm những Album / Nghệ sĩ nào, và cơ hội đóng gói sản phẩm (Bundling / Cross-sell) nào mang lại tiềm năng gia tăng giá trị đơn hàng (AOV) cao nhất?** | Khắc phục hiện tượng khách chỉ mua 1-2 bài lẻ rồi dừng; tăng số lượng track trung bình trên mỗi hóa đơn; kích hoạt doanh thu từ các bài hát ít người biết trong cùng album. | **Merchandising & Store Operations Specialist** |
| **Q3** | **Những dòng nhạc / nghệ sĩ nào đang là 'trụ cột' của từng phân khúc, và danh mục hiện tại đang thiếu hụt những album/track nào của các nghệ sĩ này cần ưu tiên mua thêm bản quyền (procure)?** | Tối ưu hóa phân bổ ngân sách bản quyền âm nhạc; loại bỏ rủi ro nhập catalog ế ẩm; nâng cao năng lực cạnh tranh và độ thỏa mãn của các khách hàng trung thành nhất. | **Content Procurement Director & Store Owner** |

---

## 3. Hệ thống Chỉ số Hiệu quả Đề xuất (Proposed KPI Framework)

> [!WARNING]
> **LƯU Ý PHÁP LÝ & GIỚI HẠN DỮ LIỆU CỦA ĐỒ ÁN:**  
> Toàn bộ các chỉ số dưới đây là **KPI ĐỀ XUẤT (Proposed Business KPIs)** được thiết kế để đo lường hiệu quả kinh doanh khi mô hình gom cụm được triển khai vào môi trường sản xuất thực tế (Production).  
> **Dữ liệu Chinook hiện tại là cơ sở dữ liệu giao dịch tĩnh trong quá khứ (Historical Static Transactions)**, hoàn toàn không có log tương tác email, không có tracking lượt nhấp, và chưa từng chạy thử nghiệm A/B Testing. Do đó, **các KPI này hiện tại KHÔNG THỂ đo lường trực tiếp từ CSDL có sẵn**, mà đóng vai trò kim chỉ nam nghiệm thu cho giai đoạn triển khai ứng dụng (Deployment Phase).

### Bảng KPI Đề xuất:

| Mã KPI | Tên chỉ số | Công thức định nghĩa | Hiện trạng (Baseline ước tính) | Mục tiêu kỳ vọng sau gom cụm | Cách thức đo lường thực tế |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **KPI-1** | **Tỷ lệ chuyển đổi chiến dịch theo cụm (Targeted Campaign Conversion Rate)** | $\frac{\text{Số đơn hàng phát sinh từ email chiến dịch}}{\text{Tổng số email gửi đi theo cụm}} \times 100\%$ | ~2.0% - 3.0% (Ước lượng theo chuẩn ngành e-commerce âm nhạc đại trà) | **5.0% - 8.0%** (Tăng gấp 2 - 2.5 lần) | Theo dõi mã UTM Campaign / Promo Code gắn riêng cho từng cụm khách hàng trong Email Marketing. |
| **KPI-2** | **Giá trị Đơn hàng Trung bình (Average Order Value - AOV)** | $\frac{\text{Tổng doanh thu chiến dịch}}{\text{Tổng số hóa đơn phát sinh}}$ | **5.64 USD / hóa đơn** (Tính từ CSDL: 39.47 USD / ~7 hóa đơn, ~5.6 bài/đơn) | **6.80 - 7.50 USD / hóa đơn** (Tăng 20% - 33% nhờ bundle) | Tính trực tiếp trên bảng `Invoice` thông qua `Invoice.Total` của các đơn hàng có áp dụng gói khuyến mãi Album. |
| **KPI-3** | **Tỷ lệ Bán chéo thành công (Cross-sell Attachment Rate)** | $\frac{\text{Số đơn hàng mua >= 2 nghệ sĩ/album khác nhau trong gợi ý}}{\text{Tổng số đơn hàng phát sinh}}$ | ~15.0% (Ước tính từ các đơn hàng có đa nghệ sĩ hiện tại) | **28.0% - 35.0%** | Kiểm tra danh mục sản phẩm trong `InvoiceLine` của từng hóa đơn sau khi triển khai thuật toán gợi ý. |
| **KPI-4** | **Tỷ lệ Hoàn tất Album (Album Completion Rate)** | $\frac{\text{Số lượt khách mua nốt các bài còn lại trong Album}}{\text{Tổng số lượt nhận được gợi ý Complete Album}}$ | **3.77%** (Hiện trạng khách mua đủ 100% bài của một album) | **10.0% - 15.0%** | Theo dõi các giao dịch mua thêm bài của các album mà khách hàng đã từng sở hữu $\ge 2$ bài trước đó. |
| **KPI-5** | **Tần suất Mua lặp lại (Purchase Repeat Cycle)** | $\text{Khoảng cách thời gian trung bình giữa 2 hóa đơn liên tiếp}$ | **218.4 ngày** (~7.3 tháng/lần mua theo phân tích EDA) | **Giảm xuống dưới 120 ngày** (~4 tháng/lần mua) | Phân tích delta `InvoiceDate` giữa các lần mua của cùng một `CustomerId`. |

---

## 4. Tóm tắt Định hướng cho Bước 5

Để hỗ trợ đắc lực cho các quyết định kinh doanh và hệ thống KPI trên:
1. Mô hình phân cụm không được dừng lại ở việc gán nhãn toán học vô nghĩa ($C_0, C_1, C_2$), mà bắt buộc phải giải thích được **Hồ sơ thị hiếu (Taste Profile)** rõ ràng cho từng phân khúc (Ai thích gì, nghệ sĩ nào tiêu biểu, cơ hội bundle ở đâu).
2. Việc lựa chọn không gian biểu diễn (Track vs Album vs Genre/Artist) ở Bước 5 phải cân bằng giữa:
   - **Độ chi tiết nghiệp vụ (Granularity):** Cần đủ sâu để đưa ra gợi ý sản phẩm cụ thể (bài hát/album nào).
   - **Tính khả thi toán học (Clusterability):** Ma trận không được quá thưa để tránh hiện tượng khoảng cách trực giao (Curse of Dimensionality).
