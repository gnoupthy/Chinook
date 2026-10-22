# BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY REPORT)
**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album
**Ngày thực hiện:** 2026-10-22
**Kết luận tổng quan:** Dữ liệu Chinook là một bộ CSDL quan hệ **rất sạch, chuẩn hóa cao (3NF)**, không có lỗi toàn vẹn tham chiếu, không có giá trị bất thường về giá hay số lượng. Không cần tạo vấn đề giả hay thực hiện thao tác xóa dữ liệu gượng ép trong SQLite.

---

## 1. Bảng tổng hợp kết quả kiểm tra chất lượng dữ liệu

| STT | Mục Kiểm Tra | Bảng / Cột | Số Dòng Ảnh Hưởng | Tỷ Lệ (%) | Mức Độ | Hiện Trạng & Cách Xử Lý |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | Missing value: Customer.Company | `Customer.Company` | 49 | 83.05% | **Thấp (Hợp lệ)** | NULL hợp lệ: Khách hàng cá nhân không thuộc công ty hoặc không cung cấp fax/tiểu bang. -> Giữ nguyên (không xử lý/không xóa). |
| 2 | Missing value: Customer.State | `Customer.State` | 29 | 49.15% | **Thấp (Hợp lệ)** | NULL hợp lệ: Khách hàng cá nhân không thuộc công ty hoặc không cung cấp fax/tiểu bang. -> Giữ nguyên (không xử lý/không xóa). |
| 3 | Missing value: Customer.PostalCode | `Customer.PostalCode` | 4 | 6.78% | **Thấp (Hợp lệ)** | NULL hợp lệ: Khách hàng cá nhân không thuộc công ty hoặc không cung cấp fax/tiểu bang. -> Giữ nguyên (không xử lý/không xóa). |
| 4 | Missing value: Customer.Phone | `Customer.Phone` | 1 | 1.69% | **Thấp (Hợp lệ)** | NULL hợp lệ: Khách hàng cá nhân không thuộc công ty hoặc không cung cấp fax/tiểu bang. -> Giữ nguyên (không xử lý/không xóa). |
| 5 | Missing value: Customer.Fax | `Customer.Fax` | 47 | 79.66% | **Thấp (Hợp lệ)** | NULL hợp lệ: Khách hàng cá nhân không thuộc công ty hoặc không cung cấp fax/tiểu bang. -> Giữ nguyên (không xử lý/không xóa). |
| 6 | Missing value: Employee.ReportsTo | `Employee.ReportsTo` | 1 | 12.50% | **Thấp (Hợp lệ)** | NULL hợp lệ: General Manager (cấp cao nhất) không có cấp trên trực tiếp. -> Giữ nguyên. |
| 7 | Missing value: Invoice.BillingState | `Invoice.BillingState` | 202 | 49.03% | **Thấp (Hợp lệ)** | NULL hợp lệ: Địa chỉ thanh toán tại các quốc gia không dùng mã bang/bưu điện. -> Giữ nguyên. |
| 8 | Missing value: Invoice.BillingPostalCode | `Invoice.BillingPostalCode` | 28 | 6.80% | **Thấp (Hợp lệ)** | NULL hợp lệ: Địa chỉ thanh toán tại các quốc gia không dùng mã bang/bưu điện. -> Giữ nguyên. |
| 9 | Missing value: Track.Composer | `Track.Composer` | 978 | 27.92% | **Thấp (Hợp lệ)** | NULL hợp lệ: Bản ghi nhạc/audio không ghi rõ tác giả sáng tác. -> Điền 'Unknown' khi phân tích tác giả nếu cần, giữ nguyên trong DB. |
| 10 | Trùng khóa chính (PK): Album | `Album (AlbumId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 11 | Trùng khóa chính (PK): Artist | `Artist (ArtistId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 12 | Trùng khóa chính (PK): Customer | `Customer (CustomerId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 13 | Trùng khóa chính (PK): Employee | `Employee (EmployeeId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 14 | Trùng khóa chính (PK): Genre | `Genre (GenreId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 15 | Trùng khóa chính (PK): Invoice | `Invoice (InvoiceId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 16 | Trùng khóa chính (PK): InvoiceLine | `InvoiceLine (InvoiceLineId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 17 | Trùng khóa chính (PK): MediaType | `MediaType (MediaTypeId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 18 | Trùng khóa chính (PK): Playlist | `Playlist (PlaylistId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 19 | Trùng khóa chính (PK): PlaylistTrack | `PlaylistTrack (PlaylistId, TrackId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 20 | Trùng khóa chính (PK): Track | `Track (TrackId)` | 0 | 0.00% | **Hoàn hảo (Sạch)** | Không có bất kỳ khóa chính nào bị trùng lặp. |
| 21 | Trùng lặp logic (Name, AlbumId) trong Track | `Track (Name, AlbumId)` | 12 | 0.34% | **Trung bình (Cần làm rõ bản chất)** | Có đúng 6 cặp (12 tracks). Phân tích chi tiết: Khác TrackId, khác MediaTypeId/Bytes/Milliseconds. Không phải lỗi nhân đôi kỹ thuật mà là các phiên bản định dạng khác nhau của cùng bài hát. Không cần xóa dòng trong SQLite gốc; khi gom cụm theo Album thì các track này tự động quy về cùng 1 AlbumId. |
| 22 | Khóa ngoại mồ côi: Album.ArtistId -> Artist.ArtistId | `Album.ArtistId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 23 | Khóa ngoại mồ côi: Track.AlbumId -> Album.AlbumId | `Track.AlbumId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 24 | Khóa ngoại mồ côi: Track.GenreId -> Genre.GenreId | `Track.GenreId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 25 | Khóa ngoại mồ côi: Track.MediaTypeId -> MediaType.MediaTypeId | `Track.MediaTypeId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 26 | Khóa ngoại mồ côi: Customer.SupportRepId -> Employee.EmployeeId | `Customer.SupportRepId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 27 | Khóa ngoại mồ côi: Invoice.CustomerId -> Customer.CustomerId | `Invoice.CustomerId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 28 | Khóa ngoại mồ côi: InvoiceLine.InvoiceId -> Invoice.InvoiceId | `InvoiceLine.InvoiceId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 29 | Khóa ngoại mồ côi: InvoiceLine.TrackId -> Track.TrackId | `InvoiceLine.TrackId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 30 | Khóa ngoại mồ côi: PlaylistTrack.PlaylistId -> Playlist.PlaylistId | `PlaylistTrack.PlaylistId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 31 | Khóa ngoại mồ côi: PlaylistTrack.TrackId -> Track.TrackId | `PlaylistTrack.TrackId` | 0 | 0.00% | **Hoàn hảo (0 mồ côi)** | 100% khớp khóa ngoại, tính toàn vẹn tham chiếu đạt tuyệt đối. |
| 32 | Khớp số liệu tài chính: Invoice.Total == SUM(InvoiceLine) | `Invoice.Total vs InvoiceLine` | 0 | 0.00% | **Hoàn hảo (Khớp 100%)** | Tất cả 412 hóa đơn đều có Total khớp chính xác tuyệt đối với tổng các dòng InvoiceLine. |
| 33 | Đơn giá bất thường (UnitPrice <= 0) | `Track & InvoiceLine UnitPrice` | 0 | 0.00% | **Hoàn hảo (Hợp lệ)** | Tất cả đơn giá đều > 0 (các mức: 0.99$ hoặc 1.99$). |
| 34 | Phân phối Quantity trong InvoiceLine | `InvoiceLine.Quantity` | 0 | 0.00% | **Thông tin quan trọng** | Quantity luôn = 1 (Min=1, Max=1) trên toàn bộ 2.240 dòng. Xác nhận: mỗi dòng hóa đơn là mua đúng 1 bản ghi track số. |
| 35 | Khoảng thời gian giao dịch (InvoiceDate) | `Invoice.InvoiceDate` | 0 | 0.00% | **Hợp lệ** | Dữ liệu liên tục từ 2009-01-01 đến 2013-12-22 (giai đoạn 2009 - 2013, trọn vẹn 5 năm). |
| 36 | Khách hàng không phát sinh giao dịch | `Customer.CustomerId` | 0 | 0.00% | **Hoàn hảo** | Toàn bộ 59/59 khách hàng đều có ít nhất 1 hóa đơn mua hàng (không có khách ảo/khách rác). |
| 37 | Track chưa từng được mua (Unsold Tracks) | `Track vs InvoiceLine` | 1,519 | 43.36% | **Đặc trưng kinh doanh (Long-tail)** | Có 1519/3503 track (43.4%) chưa từng được mua. Chỉ có 1984 track đã từng bán. Trong phân cụm khách hàng, ma trận Customer-Track chỉ cần xét trên tập track đã phát sinh giao dịch hoặc giảm chiều. |
| 38 | Album chưa từng có track nào được mua | `Album vs InvoiceLine` | 43 | 12.39% | **Đặc trưng kinh doanh (Long-tail)** | Có 43/347 album (12.4%) chưa từng được mua bài nào. 304 album có ít nhất 1 bài được mua. |

---

## 2. Phân tích chuyên sâu các điểm cốt lõi

### 2.1. Phân biệt 'Thiếu hợp lệ' và 'Thiếu bất thường'
- **Cột Company (Customer - 49/59 NULL, 83.05%):** Khách hàng là cá nhân nghe nhạc, hầu hết không mua qua danh nghĩa doanh nghiệp. Đây là thuộc tính tùy chọn hoàn toàn hợp lệ.
- **Cột Fax (Customer - 47/59 NULL, 79.66%):** Khách hàng cá nhân gần như không sử dụng máy fax.
- **Cột State (Customer - 29/59 NULL, 49.15%):** Nhiều quốc gia ở Châu Âu/Nam Mỹ (Pháp, Na Uy, Bỉ, Đức...) không áp dụng phân cấp hành chính dạng Bang (State) như Mỹ/Canada.
- **Cột ReportsTo (Employee - 1/8 NULL, 12.5%):** Chỉ có duy nhất nhân viên Andrew Adams (General Manager) không có cấp trên trực tiếp.
- **Cột Composer (Track - 978/3.503 NULL, 27.92%):** Các bản thu không rõ nhạc sĩ sáng tác (nhạc dân tộc, nhạc cổ điển, hoặc không đăng ký bản quyền nhạc sĩ).

**Kết luận:** 100% giá trị NULL trong hệ thống đều là **NULL hợp lệ về mặt nghiệp vụ**, không có bất kỳ cột bắt buộc nào (như `CustomerId`, `TrackId`, `UnitPrice`, `Total`, `InvoiceDate`) bị khuyết thiếu.

### 2.2. Kiểm tra chi tiết 6 cặp trùng (Name, AlbumId) trong bảng Track
Qua quét toàn bộ bảng Track, phát hiện đúng **6 cặp bài hát** trùng nhau cả `Name` và `AlbumId`:

| STT | Tên Track (`Name`) | AlbumId | TrackIds | MediaTypeIds | Milliseconds | Bytes | Đã Từng Mua? |
| :---: | :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `Banditismo Por Uma Questa` | 25 | `[269, 270]` | `[1, 1]` | `[307095, 243644]` | `[10251097, 8147224]` | `[True, True]` |
| 2 | `Branch Closing` | 251 | `[3206, 3428]` | `[3, 3]` | `[1822781, 1814855]` | `[358761786, 360331351]` | `[True, True]` |
| 3 | `Company Man` | 228 | `[2854, 2855]` | `[3, 3]` | `[2601226, 2601101]` | `[493168135, 503786316]` | `[False, True]` |
| 4 | `Gimme Some Truth` | 255 | `[3260, 3272]` | `[2, 2]` | `[232778, 187546]` | `[3780807, 3060083]` | `[True, True]` |
| 5 | `Imagine` | 255 | `[3262, 3267]` | `[2, 2]` | `[192329, 219078]` | `[3136271, 3562542]` | `[True, True]` |
| 6 | `Not In Portland` | 229 | `[2875, 2876]` | `[3, 3]` | `[2637303, 2637345]` | `[499061234, 510546847]` | `[False, False]` |

**Nhận định chuyên gia:**
1. **Bản chất:** Các bài hát này có thời lượng (`Milliseconds`) và dung lượng (`Bytes`) khác nhau, hoặc định dạng `MediaTypeId` khác nhau (ví dụ file AAC được bảo vệ và file MPEG thông thường, hoặc bản thu studio khác nhau).
2. **Xử lý:** Đây **không phải lỗi nhân đôi kỹ thuật (duplicate rows)** vì mỗi bản ghi có `TrackId` riêng biệt và phản ánh các phiên bản tệp khác nhau trong catalog. Trong phân cụm khách hàng, khi gom về cấp độ Album (`AlbumId`), sự trùng lặp này hoàn toàn không ảnh hưởng tới kết quả.

### 2.3. Tính nhất quán tài chính và hành vi Quantity
- **Khớp số liệu:** `100%` (412/412) hóa đơn có `Invoice.Total` khớp chính xác với `SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity)` (chênh lệch tuyệt đối = 0.0000).
- **Quantity luôn = 1:** Trong nhạc số, khách hàng chỉ mua quyền sở hữu 1 bản copy của mỗi track, không bao giờ mua 2 bản giống hệt nhau trong cùng 1 hóa đơn.

### 2.4. Độ phủ kinh doanh & Hệ quả cho bài toán gom cụm
- **Chỉ 1,984/3,503 track (56.64%)** đã từng được mua ít nhất một lần. Có 1,519 track chưa bao giờ bán được.
- **304/347 album (87.61%)** có ít nhất 1 bài được mua. Có 43 album chưa từng phát sinh đơn hàng.
- **Hàm ý cho gom cụm:** Không gian biểu diễn phân cụm nên tập trung vào tập hợp track/album đã phát sinh giao dịch hoặc dùng kỹ thuật giảm chiều (TruncatedSVD / NMF / PCA) để nén không gian ma trận thưa.