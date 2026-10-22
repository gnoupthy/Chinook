# BÁO CÁO TỪ ĐIỂN DỮ LIỆU (DATA DICTIONARY)
**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album
**Nguồn dữ liệu:** `data/Chinook_Sqlite.sqlite` (Chế độ chỉ đọc - Read-Only)

---

## 1. Đường nối nghiệp vụ cốt lõi (Core Business Lineage)
Hành vi mua hàng và quan tâm của khách hàng được liên kết xuyên suốt qua chuỗi quan hệ:
```
Customer (59) ──(1:N)──> Invoice (412) ──(1:N)──> InvoiceLine (2.240)
                                                       │ (N:1)
                                                       ▼
Genre (25) <──(N:1)── Track (3.503) ──(N:1)──> MediaType (5)
                        │ (N:1)
                        ▼
                      Album (347) ──(N:1)──> Artist (275)
```

**Ý nghĩa chuỗi quan hệ:**
- Mỗi **Customer** phát sinh nhiều **Invoice** (hóa đơn thanh toán).
- Mỗi **Invoice** gồm nhiều dòng **InvoiceLine** (chi tiết giao dịch).
- Mỗi **InvoiceLine** chỉ rõ khách hàng mua **Track** nào.
- Từ **Track**, ta truy ngược về **Album** chứa track đó, **Artist** thể hiện, **Genre** (thể loại) và **MediaType** (định dạng).
- **Bảng Playlist & PlaylistTrack (18 playlists, 8.715 liên kết):** Là danh mục do hệ thống/cửa hàng biên soạn sẵn, hoàn toàn không có `CustomerId` gắn kèm, do đó **không phản ánh tín hiệu quan tâm cá nhân của từng khách hàng** và được loại khỏi quá trình trích xuất đặc trưng gom cụm khách hàng.

---

## 2. Chi tiết cấu trúc từng bảng (Table-level Data Dictionary)

### Bảng `Album`
- **Số lượng bản ghi (Row Count):** 347 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một album âm nhạc do một nghệ sĩ phát hành.
- **Khóa ngoại (Foreign Keys):** `ArtistId` -> `Artist.ArtistId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `AlbumId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện album - Định danh duy nhất |
| `Title` | `NVARCHAR(160)` |  | NOT NULL | 0 (0.0%) | `For Those About To Rock We Salut...` | Tựa đề của album - Tên album âm nhạc |
| `ArtistId` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `1` | Khóa ngoại trỏ về Artist.ArtistId - Nghệ sĩ phát hành |


### Bảng `Artist`
- **Số lượng bản ghi (Row Count):** 275 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một nghệ sĩ hoặc ban nhạc thực hiện tác phẩm.
- **Khóa ngoại (Foreign Keys):** Không có

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `ArtistId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện nghệ sĩ - Định danh duy nhất |
| `Name` | `NVARCHAR(120)` |  | NULL OK | 0 (0.0%) | `AC/DC` | Tên nghệ sĩ hoặc nhóm nhạc - Tên thương hiệu âm nhạc |


### Bảng `Customer`
- **Số lượng bản ghi (Row Count):** 59 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một khách hàng cá nhân đăng ký tài khoản tại cửa hàng.
- **Khóa ngoại (Foreign Keys):** `SupportRepId` -> `Employee.EmployeeId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `CustomerId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện khách hàng - Định danh duy nhất |
| `FirstName` | `NVARCHAR(40)` |  | NOT NULL | 0 (0.0%) | `Luís` | Tên khách hàng - Tên gọi |
| `LastName` | `NVARCHAR(20)` |  | NOT NULL | 0 (0.0%) | `Gonçalves` | Họ khách hàng - Họ |
| `Company` | `NVARCHAR(80)` |  | NULL OK | 49 (83.1%) | `Embraer - Empresa Brasileira de ...` | Tên công ty / tổ chức của khách hàng - Tùy chọn, nhiều giá trị NULL |
| `Address` | `NVARCHAR(70)` |  | NULL OK | 0 (0.0%) | `Av. Brigadeiro Faria Lima, 2170` | Địa chỉ đường/phố cư trú - Địa chỉ liên hệ |
| `City` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `São José dos Campos` | Thành phố cư trú - Địa chỉ |
| `State` | `NVARCHAR(40)` |  | NULL OK | 29 (49.2%) | `SP` | Tiểu bang / tỉnh - Tùy chọn ở một số quốc gia |
| `Country` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `Brazil` | Quốc gia cư trú - Quốc gia của khách |
| `PostalCode` | `NVARCHAR(10)` |  | NULL OK | 4 (6.8%) | `12227-000` | Mã bưu chính - Mã vùng bưu điện |
| `Phone` | `NVARCHAR(24)` |  | NULL OK | 1 (1.7%) | `+55 (12) 3923-5555` | Số điện thoại liên hệ - Điện thoại |
| `Fax` | `NVARCHAR(24)` |  | NULL OK | 47 (79.7%) | `+55 (12) 3923-5566` | Số fax liên hệ - Tùy chọn, đa phần để trống |
| `Email` | `NVARCHAR(60)` |  | NOT NULL | 0 (0.0%) | `luisg@embraer.com.br` | Địa chỉ email khách hàng - Dùng để liên hệ/thanh toán |
| `SupportRepId` | `INTEGER` |  | NULL OK | 0 (0.0%) | `3` | Khóa ngoại trỏ về Employee.EmployeeId - Nhân viên phụ trách hỗ trợ |


### Bảng `Employee`
- **Số lượng bản ghi (Row Count):** 8 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một nhân viên công ty (quản lý, hỗ trợ bán hàng, IT).
- **Khóa ngoại (Foreign Keys):** `ReportsTo` -> `Employee.EmployeeId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `EmployeeId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện nhân viên - Định danh duy nhất |
| `LastName` | `NVARCHAR(20)` |  | NOT NULL | 0 (0.0%) | `Adams` | Họ nhân viên - Họ |
| `FirstName` | `NVARCHAR(20)` |  | NOT NULL | 0 (0.0%) | `Andrew` | Tên nhân viên - Tên gọi |
| `Title` | `NVARCHAR(30)` |  | NULL OK | 0 (0.0%) | `General Manager` | Chức danh công việc - Ví dụ: General Manager, Sales Support Agent |
| `ReportsTo` | `INTEGER` |  | NULL OK | 1 (12.5%) | `1.0` | Khóa ngoại tự trỏ EmployeeId của quản lý trực tiếp - NULL đối với Tổng giám đốc |
| `BirthDate` | `DATETIME` |  | NULL OK | 0 (0.0%) | `1962-02-18 00:00:00` | Ngày tháng năm sinh - Định dạng YYYY-MM-DD HH:MM:SS |
| `HireDate` | `DATETIME` |  | NULL OK | 0 (0.0%) | `2002-08-14 00:00:00` | Ngày vào làm chính thức - Định dạng YYYY-MM-DD HH:MM:SS |
| `Address` | `NVARCHAR(70)` |  | NULL OK | 0 (0.0%) | `11120 Jasper Ave NW` | Địa chỉ nơi cư trú - Địa chỉ |
| `City` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `Edmonton` | Thành phố làm việc / cư trú - Thành phố |
| `State` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `AB` | Tỉnh / bang - Địa chỉ |
| `Country` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `Canada` | Quốc gia - Quốc gia |
| `PostalCode` | `NVARCHAR(10)` |  | NULL OK | 0 (0.0%) | `T5K 2N1` | Mã bưu chính - Mã vùng |
| `Phone` | `NVARCHAR(24)` |  | NULL OK | 0 (0.0%) | `+1 (780) 428-9482` | Số điện thoại liên hệ - Điện thoại |
| `Fax` | `NVARCHAR(24)` |  | NULL OK | 0 (0.0%) | `+1 (780) 428-3457` | Số fax liên hệ - Fax |
| `Email` | `NVARCHAR(60)` |  | NULL OK | 0 (0.0%) | `andrew@chinookcorp.com` | Email công vụ của nhân viên - Email nội bộ |


### Bảng `Genre`
- **Số lượng bản ghi (Row Count):** 25 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một thể loại âm nhạc/nội dung (Rock, Jazz, Metal...).
- **Khóa ngoại (Foreign Keys):** Không có

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `GenreId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện thể loại - Định danh duy nhất |
| `Name` | `NVARCHAR(120)` |  | NULL OK | 0 (0.0%) | `Rock` | Tên thể loại âm nhạc - Ví dụ: Rock, Jazz, Metal |


### Bảng `Invoice`
- **Số lượng bản ghi (Row Count):** 412 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một hóa đơn thanh toán cho một phiên mua hàng của khách.
- **Khóa ngoại (Foreign Keys):** `CustomerId` -> `Customer.CustomerId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `InvoiceId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện hóa đơn - Định danh duy nhất |
| `CustomerId` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `2` | Khóa ngoại trỏ về Customer.CustomerId - Khách hàng mua đơn này |
| `InvoiceDate` | `DATETIME` |  | NOT NULL | 0 (0.0%) | `2009-01-01 00:00:00` | Thời điểm xuất hóa đơn - Định dạng timestamp YYYY-MM-DD |
| `BillingAddress` | `NVARCHAR(70)` |  | NULL OK | 0 (0.0%) | `Theodor-Heuss-Straße 34` | Địa chỉ gửi hóa đơn / thanh toán - Địa chỉ thanh toán |
| `BillingCity` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `Stuttgart` | Thành phố gửi hóa đơn - Thành phố |
| `BillingState` | `NVARCHAR(40)` |  | NULL OK | 202 (49.0%) | `AB` | Tiểu bang / tỉnh thanh toán - Tùy chọn |
| `BillingCountry` | `NVARCHAR(40)` |  | NULL OK | 0 (0.0%) | `Germany` | Quốc gia thanh toán - Quốc gia |
| `BillingPostalCode` | `NVARCHAR(10)` |  | NULL OK | 28 (6.8%) | `70174` | Mã bưu chính thanh toán - Mã bưu chính |
| `Total` | `NUMERIC(10,2)` |  | NOT NULL | 0 (0.0%) | `1.98` | Tổng tiền của hóa đơn (USD) - Bằng tổng tiền các dòng InvoiceLine |


### Bảng `InvoiceLine`
- **Số lượng bản ghi (Row Count):** 2,240 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một mục chi tiết mua một track trong một hóa đơn (line item).
- **Khóa ngoại (Foreign Keys):** `TrackId` -> `Track.TrackId`, `InvoiceId` -> `Invoice.InvoiceId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `InvoiceLineId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính chi tiết hóa đơn - Định danh duy nhất |
| `InvoiceId` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `1` | Khóa ngoại trỏ về Invoice.InvoiceId - Đơn hàng chứa mục này |
| `TrackId` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `2` | Khóa ngoại trỏ về Track.TrackId - Bài hát được mua |
| `UnitPrice` | `NUMERIC(10,2)` |  | NOT NULL | 0 (0.0%) | `0.99` | Đơn giá bán tại thời điểm mua (USD) - Thường là 0.99 hoặc 1.99 |
| `Quantity` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `1` | Số lượng track mua trong dòng này - Hầu hết hoặc luôn = 1 |


### Bảng `MediaType`
- **Số lượng bản ghi (Row Count):** 5 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một định dạng tệp âm thanh/video kỹ thuật số.
- **Khóa ngoại (Foreign Keys):** Không có

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `MediaTypeId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện định dạng media - Định danh duy nhất |
| `Name` | `NVARCHAR(120)` |  | NULL OK | 0 (0.0%) | `MPEG audio file` | Tên định dạng lưu trữ bài hát - Ví dụ: MPEG audio file, AAC |


### Bảng `Playlist`
- **Số lượng bản ghi (Row Count):** 18 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một danh sách phát nhạc tổng hợp được tạo sẵn trong hệ thống.
- **Khóa ngoại (Foreign Keys):** Không có

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `PlaylistId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện playlist - Định danh duy nhất |
| `Name` | `NVARCHAR(120)` |  | NULL OK | 0 (0.0%) | `Music` | Tên danh sách phát nhạc - Ví dụ: Music, TV Shows |


### Bảng `PlaylistTrack`
- **Số lượng bản ghi (Row Count):** 8,715 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một liên kết xác định một track thuộc về một playlist cụ thể (bảng quan hệ N:M).
- **Khóa ngoại (Foreign Keys):** `TrackId` -> `Track.TrackId`, `PlaylistId` -> `Playlist.PlaylistId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `PlaylistId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa ngoại trỏ về Playlist.PlaylistId (PK ghép) - Playlist chứa track |
| `TrackId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `3402` | Khóa ngoại trỏ về Track.TrackId (PK ghép) - Track nằm trong playlist |


### Bảng `Track`
- **Số lượng bản ghi (Row Count):** 3,503 dòng
- **Mức độ chi tiết của dòng (Grain):** Mỗi dòng đại diện cho một bản nhạc/bài hát đơn lẻ có thể mua được.
- **Khóa ngoại (Foreign Keys):** `MediaTypeId` -> `MediaType.MediaTypeId`, `GenreId` -> `Genre.GenreId`, `AlbumId` -> `Album.AlbumId`

| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `TrackId` | `INTEGER` | PK | NOT NULL | 0 (0.0%) | `1` | Khóa chính nhận diện bài hát - Định danh duy nhất |
| `Name` | `NVARCHAR(200)` |  | NOT NULL | 0 (0.0%) | `For Those About To Rock (We Salu...` | Tên bài hát / bản nhạc - Tên tác phẩm |
| `AlbumId` | `INTEGER` |  | NULL OK | 0 (0.0%) | `1` | Khóa ngoại trỏ về Album.AlbumId - Album chứa bài hát |
| `MediaTypeId` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `1` | Khóa ngoại trỏ về MediaType.MediaTypeId - Định dạng âm thanh |
| `GenreId` | `INTEGER` |  | NULL OK | 0 (0.0%) | `1` | Khóa ngoại trỏ về Genre.GenreId - Thể loại nhạc |
| `Composer` | `NVARCHAR(220)` |  | NULL OK | 978 (27.9%) | `Angus Young, Malcolm Young, Bria...` | Tác giả / người sáng tác bài hát - Nhiều giá trị NULL nếu không có thông tin |
| `Milliseconds` | `INTEGER` |  | NOT NULL | 0 (0.0%) | `343719` | Thời lượng bài hát tính theo mili-giây - 1 giây = 1.000 ms |
| `Bytes` | `INTEGER` |  | NULL OK | 0 (0.0%) | `11170334` | Kích thước tệp tính theo Byte - Dung lượng lưu trữ |
| `UnitPrice` | `NUMERIC(10,2)` |  | NOT NULL | 0 (0.0%) | `0.99` | Giá bán niêm yết của bài hát (USD) - Giá chuẩn hiện hành |

