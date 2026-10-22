"""
Generate Data Dictionary Markdown Report from live database inspection.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_connection, get_schema_summary, load_table

sys.stdout.reconfigure(encoding='utf-8')

GRAIN_DESCRIPTIONS = {
    "Album": "Mỗi dòng đại diện cho một album âm nhạc do một nghệ sĩ phát hành.",
    "Artist": "Mỗi dòng đại diện cho một nghệ sĩ hoặc ban nhạc thực hiện tác phẩm.",
    "Customer": "Mỗi dòng đại diện cho một khách hàng cá nhân đăng ký tài khoản tại cửa hàng.",
    "Employee": "Mỗi dòng đại diện cho một nhân viên công ty (quản lý, hỗ trợ bán hàng, IT).",
    "Genre": "Mỗi dòng đại diện cho một thể loại âm nhạc/nội dung (Rock, Jazz, Metal...).",
    "Invoice": "Mỗi dòng đại diện cho một hóa đơn thanh toán cho một phiên mua hàng của khách.",
    "InvoiceLine": "Mỗi dòng đại diện cho một mục chi tiết mua một track trong một hóa đơn (line item).",
    "MediaType": "Mỗi dòng đại diện cho một định dạng tệp âm thanh/video kỹ thuật số.",
    "Playlist": "Mỗi dòng đại diện cho một danh sách phát nhạc tổng hợp được tạo sẵn trong hệ thống.",
    "PlaylistTrack": "Mỗi dòng đại diện cho một liên kết xác định một track thuộc về một playlist cụ thể (bảng quan hệ N:M).",
    "Track": "Mỗi dòng đại diện cho một bản nhạc/bài hát đơn lẻ có thể mua được."
}

COLUMN_DESCRIPTIONS = {
    # Album
    ("Album", "AlbumId"): ("Khóa chính nhận diện album", "Định danh duy nhất"),
    ("Album", "Title"): ("Tựa đề của album", "Tên album âm nhạc"),
    ("Album", "ArtistId"): ("Khóa ngoại trỏ về Artist.ArtistId", "Nghệ sĩ phát hành"),
    # Artist
    ("Artist", "ArtistId"): ("Khóa chính nhận diện nghệ sĩ", "Định danh duy nhất"),
    ("Artist", "Name"): ("Tên nghệ sĩ hoặc nhóm nhạc", "Tên thương hiệu âm nhạc"),
    # Customer
    ("Customer", "CustomerId"): ("Khóa chính nhận diện khách hàng", "Định danh duy nhất"),
    ("Customer", "FirstName"): ("Tên khách hàng", "Tên gọi"),
    ("Customer", "LastName"): ("Họ khách hàng", "Họ"),
    ("Customer", "Company"): ("Tên công ty / tổ chức của khách hàng", "Tùy chọn, nhiều giá trị NULL"),
    ("Customer", "Address"): ("Địa chỉ đường/phố cư trú", "Địa chỉ liên hệ"),
    ("Customer", "City"): ("Thành phố cư trú", "Địa chỉ"),
    ("Customer", "State"): ("Tiểu bang / tỉnh", "Tùy chọn ở một số quốc gia"),
    ("Customer", "Country"): ("Quốc gia cư trú", "Quốc gia của khách"),
    ("Customer", "PostalCode"): ("Mã bưu chính", "Mã vùng bưu điện"),
    ("Customer", "Phone"): ("Số điện thoại liên hệ", "Điện thoại"),
    ("Customer", "Fax"): ("Số fax liên hệ", "Tùy chọn, đa phần để trống"),
    ("Customer", "Email"): ("Địa chỉ email khách hàng", "Dùng để liên hệ/thanh toán"),
    ("Customer", "SupportRepId"): ("Khóa ngoại trỏ về Employee.EmployeeId", "Nhân viên phụ trách hỗ trợ"),
    # Employee
    ("Employee", "EmployeeId"): ("Khóa chính nhận diện nhân viên", "Định danh duy nhất"),
    ("Employee", "LastName"): ("Họ nhân viên", "Họ"),
    ("Employee", "FirstName"): ("Tên nhân viên", "Tên gọi"),
    ("Employee", "Title"): ("Chức danh công việc", "Ví dụ: General Manager, Sales Support Agent"),
    ("Employee", "ReportsTo"): ("Khóa ngoại tự trỏ EmployeeId của quản lý trực tiếp", "NULL đối với Tổng giám đốc"),
    ("Employee", "BirthDate"): ("Ngày tháng năm sinh", "Định dạng YYYY-MM-DD HH:MM:SS"),
    ("Employee", "HireDate"): ("Ngày vào làm chính thức", "Định dạng YYYY-MM-DD HH:MM:SS"),
    ("Employee", "Address"): ("Địa chỉ nơi cư trú", "Địa chỉ"),
    ("Employee", "City"): ("Thành phố làm việc / cư trú", "Thành phố"),
    ("Employee", "State"): ("Tỉnh / bang", "Địa chỉ"),
    ("Employee", "Country"): ("Quốc gia", "Quốc gia"),
    ("Employee", "PostalCode"): ("Mã bưu chính", "Mã vùng"),
    ("Employee", "Phone"): ("Số điện thoại liên hệ", "Điện thoại"),
    ("Employee", "Fax"): ("Số fax liên hệ", "Fax"),
    ("Employee", "Email"): ("Email công vụ của nhân viên", "Email nội bộ"),
    # Genre
    ("Genre", "GenreId"): ("Khóa chính nhận diện thể loại", "Định danh duy nhất"),
    ("Genre", "Name"): ("Tên thể loại âm nhạc", "Ví dụ: Rock, Jazz, Metal"),
    # Invoice
    ("Invoice", "InvoiceId"): ("Khóa chính nhận diện hóa đơn", "Định danh duy nhất"),
    ("Invoice", "CustomerId"): ("Khóa ngoại trỏ về Customer.CustomerId", "Khách hàng mua đơn này"),
    ("Invoice", "InvoiceDate"): ("Thời điểm xuất hóa đơn", "Định dạng timestamp YYYY-MM-DD"),
    ("Invoice", "BillingAddress"): ("Địa chỉ gửi hóa đơn / thanh toán", "Địa chỉ thanh toán"),
    ("Invoice", "BillingCity"): ("Thành phố gửi hóa đơn", "Thành phố"),
    ("Invoice", "BillingState"): ("Tiểu bang / tỉnh thanh toán", "Tùy chọn"),
    ("Invoice", "BillingCountry"): ("Quốc gia thanh toán", "Quốc gia"),
    ("Invoice", "BillingPostalCode"): ("Mã bưu chính thanh toán", "Mã bưu chính"),
    ("Invoice", "Total"): ("Tổng tiền của hóa đơn (USD)", "Bằng tổng tiền các dòng InvoiceLine"),
    # InvoiceLine
    ("InvoiceLine", "InvoiceLineId"): ("Khóa chính chi tiết hóa đơn", "Định danh duy nhất"),
    ("InvoiceLine", "InvoiceId"): ("Khóa ngoại trỏ về Invoice.InvoiceId", "Đơn hàng chứa mục này"),
    ("InvoiceLine", "TrackId"): ("Khóa ngoại trỏ về Track.TrackId", "Bài hát được mua"),
    ("InvoiceLine", "UnitPrice"): ("Đơn giá bán tại thời điểm mua (USD)", "Thường là 0.99 hoặc 1.99"),
    ("InvoiceLine", "Quantity"): ("Số lượng track mua trong dòng này", "Hầu hết hoặc luôn = 1"),
    # MediaType
    ("MediaType", "MediaTypeId"): ("Khóa chính nhận diện định dạng media", "Định danh duy nhất"),
    ("MediaType", "Name"): ("Tên định dạng lưu trữ bài hát", "Ví dụ: MPEG audio file, AAC"),
    # Playlist
    ("Playlist", "PlaylistId"): ("Khóa chính nhận diện playlist", "Định danh duy nhất"),
    ("Playlist", "Name"): ("Tên danh sách phát nhạc", "Ví dụ: Music, TV Shows"),
    # PlaylistTrack
    ("PlaylistTrack", "PlaylistId"): ("Khóa ngoại trỏ về Playlist.PlaylistId (PK ghép)", "Playlist chứa track"),
    ("PlaylistTrack", "TrackId"): ("Khóa ngoại trỏ về Track.TrackId (PK ghép)", "Track nằm trong playlist"),
    # Track
    ("Track", "TrackId"): ("Khóa chính nhận diện bài hát", "Định danh duy nhất"),
    ("Track", "Name"): ("Tên bài hát / bản nhạc", "Tên tác phẩm"),
    ("Track", "AlbumId"): ("Khóa ngoại trỏ về Album.AlbumId", "Album chứa bài hát"),
    ("Track", "MediaTypeId"): ("Khóa ngoại trỏ về MediaType.MediaTypeId", "Định dạng âm thanh"),
    ("Track", "GenreId"): ("Khóa ngoại trỏ về Genre.GenreId", "Thể loại nhạc"),
    ("Track", "Composer"): ("Tác giả / người sáng tác bài hát", "Nhiều giá trị NULL nếu không có thông tin"),
    ("Track", "Milliseconds"): ("Thời lượng bài hát tính theo mili-giây", "1 giây = 1.000 ms"),
    ("Track", "Bytes"): ("Kích thước tệp tính theo Byte", "Dung lượng lưu trữ"),
    ("Track", "UnitPrice"): ("Giá bán niêm yết của bài hát (USD)", "Giá chuẩn hiện hành")
}


def build_markdown_report():
    conn = get_connection()
    schema = get_schema_summary(conn)

    md = []
    md.append("# BÁO CÁO TỪ ĐIỂN DỮ LIỆU (DATA DICTIONARY)")
    md.append("**Dự án:** Gom cụm khách hàng theo sự quan tâm tới Track và Album")
    md.append("**Nguồn dữ liệu:** `data/Chinook_Sqlite.sqlite` (Chế độ chỉ đọc - Read-Only)")
    md.append("\n---\n")

    md.append("## 1. Đường nối nghiệp vụ cốt lõi (Core Business Lineage)")
    md.append("Hành vi mua hàng và quan tâm của khách hàng được liên kết xuyên suốt qua chuỗi quan hệ:")
    md.append("```")
    md.append("Customer (59) ──(1:N)──> Invoice (412) ──(1:N)──> InvoiceLine (2.240)")
    md.append("                                                       │ (N:1)")
    md.append("                                                       ▼")
    md.append("Genre (25) <──(N:1)── Track (3.503) ──(N:1)──> MediaType (5)")
    md.append("                        │ (N:1)")
    md.append("                        ▼")
    md.append("                      Album (347) ──(N:1)──> Artist (275)")
    md.append("```")
    md.append("\n**Ý nghĩa chuỗi quan hệ:**")
    md.append("- Mỗi **Customer** phát sinh nhiều **Invoice** (hóa đơn thanh toán).")
    md.append("- Mỗi **Invoice** gồm nhiều dòng **InvoiceLine** (chi tiết giao dịch).")
    md.append("- Mỗi **InvoiceLine** chỉ rõ khách hàng mua **Track** nào.")
    md.append("- Từ **Track**, ta truy ngược về **Album** chứa track đó, **Artist** thể hiện, **Genre** (thể loại) và **MediaType** (định dạng).")
    md.append("- **Bảng Playlist & PlaylistTrack (18 playlists, 8.715 liên kết):** Là danh mục do hệ thống/cửa hàng biên soạn sẵn, hoàn toàn không có `CustomerId` gắn kèm, do đó **không phản ánh tín hiệu quan tâm cá nhân của từng khách hàng** và được loại khỏi quá trình trích xuất đặc trưng gom cụm khách hàng.")
    md.append("\n---\n")

    md.append("## 2. Chi tiết cấu trúc từng bảng (Table-level Data Dictionary)\n")

    for table_name in sorted(schema.keys()):
        info = schema[table_name]
        df = load_table(table_name, conn)
        grain = GRAIN_DESCRIPTIONS.get(table_name, "N/A")

        md.append(f"### Bảng `{table_name}`")
        md.append(f"- **Số lượng bản ghi (Row Count):** {info['row_count']:,} dòng")
        md.append(f"- **Mức độ chi tiết của dòng (Grain):** {grain}")

        # Foreign keys list
        if info['foreign_keys']:
            fk_strs = [f"`{fk['from_col']}` -> `{fk['to_table']}.{fk['to_col']}`" for fk in info['foreign_keys']]
            md.append(f"- **Khóa ngoại (Foreign Keys):** {', '.join(fk_strs)}")
        else:
            md.append("- **Khóa ngoại (Foreign Keys):** Không có")

        md.append("\n| Tên Cột | Kiểu Dữ Liệu | Khóa | Ràng Buộc | Số Lượng NULL (%) | Giá Trị Ví Dụ | Ý Nghĩa / Ghi Chú |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :--- | :--- |")

        for col in info['columns']:
            cname = col['name']
            ctype = col['type']
            is_pk = "PK" if col['pk'] else ""
            not_null = "NOT NULL" if col['notnull'] else "NULL OK"
            null_cnt = df[cname].isnull().sum()
            null_pct = (null_cnt / len(df) * 100) if len(df) > 0 else 0
            null_str = f"{null_cnt} ({null_pct:.1f}%)" if null_cnt > 0 else "0 (0.0%)"

            sample_series = df[cname].dropna()
            if not sample_series.empty:
                val = sample_series.iloc[0]
                if isinstance(val, str) and len(val) > 35:
                    sample_val = val[:32] + "..."
                else:
                    sample_val = str(val)
            else:
                sample_val = "NULL"

            desc_tuple = COLUMN_DESCRIPTIONS.get((table_name, cname), ("Mô tả cột dữ liệu", ""))
            meaning = desc_tuple[0]
            note = f" - {desc_tuple[1]}" if desc_tuple[1] else ""

            md.append(f"| `{cname}` | `{ctype}` | {is_pk} | {not_null} | {null_str} | `{sample_val}` | {meaning}{note} |")

        md.append("\n")

    conn.close()

    with open("reports/data_dictionary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("Successfully generated reports/data_dictionary.md!")


if __name__ == "__main__":
    build_markdown_report()
