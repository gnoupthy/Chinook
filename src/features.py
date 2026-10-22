"""Module xây dựng ma trận đặc trưng phân tích (Feature Engineering).

Tuân thủ nguyên tắc:
1. Đọc dữ liệu từ tầng sạch data/processed/ (không đọc trực tiếp SQLite).
2. Item vocabulary (từ vựng cột) được lấy từ catalog đầy đủ (dim_*), không lấy từ tập train,
   để đảm bảo tính nhất quán giữa train/test và phục vụ khách hàng mới.
3. Quantity luôn = 1, mỗi cặp (khách, track) là duy nhất. Không dùng biểu diễn 'số lượng mua'
   theo track; chỉ dùng nhị phân, đếm số track đã mua theo mục, tỷ lệ phủ và tỷ trọng.
4. Thông tin nhân khẩu học và chi tiêu được tách riêng, không đưa vào không gian đặc trưng.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


def load_processed_data(
    data_dir: Union[str, Path] = "data/processed"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Đọc dữ liệu sạch từ tầng phân tích data/processed/ (Parquet).

    Args:
        data_dir: Đường dẫn thư mục chứa các bảng Parquet đã xử lý ở Pha ETL.

    Returns:
        fact_purchase: DataFrame bảng sự kiện mua hàng (2.240 dòng).
        dims: Dictionary chứa các bảng chiều (dim_customer, dim_track, dim_album, dim_artist, dim_genre).
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Thư mục dữ liệu đã xử lý không tồn tại: {path.resolve()}")

    fact_purchase = pd.read_parquet(path / "fact_purchase.parquet")

    dims = {
        "dim_customer": pd.read_parquet(path / "dim_customer.parquet"),
        "dim_track": pd.read_parquet(path / "dim_track.parquet"),
        "dim_album": pd.read_parquet(path / "dim_album.parquet"),
        "dim_artist": pd.read_parquet(path / "dim_artist.parquet"),
        "dim_genre": pd.read_parquet(path / "dim_genre.parquet"),
    }

    return fact_purchase, dims


def get_catalog_vocabulary(dims: Dict[str, pd.DataFrame], level: str) -> List[int]:
    """Trích xuất từ vựng đầy đủ từ catalog chiều tương ứng.

    Đảm bảo cột ma trận luôn cố định và nhất quán, không phụ thuộc vào tập train hay test.
    """
    level_lower = level.lower()
    if level_lower == "album":
        return sorted(dims["dim_album"]["AlbumId"].unique().tolist())
    elif level_lower == "genre":
        return sorted(dims["dim_genre"]["GenreId"].unique().tolist())
    elif level_lower == "artist":
        return sorted(dims["dim_artist"]["ArtistId"].unique().tolist())
    elif level_lower == "track":
        return sorted(dims["dim_track"]["TrackId"].unique().tolist())
    else:
        raise ValueError(
            f"Level '{level}' không hợp lệ. Phải thuộc {{'album', 'genre', 'artist', 'track'}}."
        )


def build_matrix(
    fact_purchase: pd.DataFrame,
    dims: Dict[str, pd.DataFrame],
    level: str,
    metric: str = "binary",
    customer_ids: Optional[List[int]] = None,
    return_sparse: bool = False,
) -> Union[pd.DataFrame, sparse.csr_matrix]:
    """Xây dựng ma trận tương tác Khách hàng x Mục (Customer x Item Matrix).

    Args:
        fact_purchase: Bảng sự kiện mua bài hát (mỗi dòng là 1 cặp duy nhất CustomerId, TrackId).
        dims: Dictionary chứa các bảng chiều catalog.
        level: Cấp độ tổng hợp {'album', 'genre', 'artist', 'track'}.
        metric: Cách đo lường {'binary', 'count', 'coverage', 'share'}.
            - 'binary': 1 nếu khách mua >= 1 track trong mục, ngược lại 0.
            - 'count': Số lượng track đã mua trong mục (đo độ sâu quan tâm).
            - 'coverage': (Chỉ dùng cho album) Số track mua / Tổng số track trong album.
            - 'share': Tỷ trọng số track mua trong mục / Tổng số track khách hàng đã mua.
        customer_ids: Danh sách khách hàng cần dựng ma trận (mặc định lấy toàn bộ 59 khách hàng).
        return_sparse: Nếu True trả về scipy.sparse.csr_matrix, ngược lại pd.DataFrame.

    Returns:
        Ma trận kích thước [số khách hàng, kích thước catalog vocabulary].
    """
    level_lower = level.lower()
    metric_lower = metric.lower()

    # 1. Xác định vocabulary đầy đủ từ catalog
    vocabulary = get_catalog_vocabulary(dims, level_lower)

    # 2. Xác định danh sách khách hàng (index)
    if customer_ids is None:
        all_customers = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())
    else:
        all_customers = sorted(list(customer_ids))

    # 3. Xác định cột phân nhóm trong fact_purchase
    col_map = {
        "album": "AlbumId",
        "genre": "GenreId",
        "artist": "ArtistId",
        "track": "TrackId",
    }
    item_col = col_map[level_lower]

    # Lọc fact theo customer_ids nếu có
    fact_filtered = fact_purchase[fact_purchase["CustomerId"].isin(all_customers)]

    # 4. Tính toán số lượng track đã mua theo từng cặp (CustomerId, ItemId)
    # Vì mỗi dòng fact_purchase là đúng 1 bài hát đã mua
    counts_df = (
        fact_filtered.groupby(["CustomerId", item_col])
        .size()
        .unstack(fill_value=0)
    )

    # Đảm bảo đủ tất cả các dòng khách hàng và cột vocabulary catalog
    full_counts = counts_df.reindex(index=all_customers, columns=vocabulary, fill_value=0)
    full_counts.index.name = "CustomerId"

    # 5. Áp dụng metric
    if metric_lower == "binary":
        mat = (full_counts > 0).astype(float)
    elif metric_lower == "count":
        mat = full_counts.astype(float)
    elif metric_lower == "coverage":
        if level_lower != "album":
            raise ValueError("Metric 'coverage' chỉ hỗ trợ cho level='album'.")
        # Tính tỷ lệ phủ: track đã mua / tổng số track của album
        total_tracks_series = dims["dim_album"].set_index("AlbumId")["total_tracks"]
        total_tracks_aligned = total_tracks_series.reindex(vocabulary).fillna(1.0).values
        # Tránh chia cho 0
        total_tracks_aligned = np.maximum(total_tracks_aligned, 1.0)
        mat = full_counts.astype(float).div(total_tracks_aligned, axis=1)
    elif metric_lower == "share":
        # Tỷ trọng theo từng khách: count / tổng số track khách đã mua
        row_sums = full_counts.sum(axis=1)
        row_sums_safe = row_sums.replace(0, 1.0)
        mat = full_counts.astype(float).div(row_sums_safe, axis=0)
    else:
        raise ValueError(
            f"Metric '{metric}' không hợp lệ. Phải thuộc {{'binary', 'count', 'coverage', 'share'}}."
        )

    mat = mat.fillna(0.0)

    if return_sparse:
        return sparse.csr_matrix(mat.values)
    return mat


def build_composite_raw_matrix(
    fact_purchase: pd.DataFrame,
    dims: Dict[str, pd.DataFrame],
    blocks_config: List[Dict[str, Any]],
    customer_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Xây dựng ma trận thô kết hợp nhiều khối (Composite Blocks).

    Ghép nối các ma trận thô theo chiều ngang (column concatenation),
    đặt tên cột phân biệt theo block (ví dụ: album_1, genre_1).
    """
    block_dfs = []
    for i, b in enumerate(blocks_config):
        lvl = b["level"]
        mtr = b["metric"]
        prefix = f"{lvl}"
        df_block = build_matrix(
            fact_purchase=fact_purchase,
            dims=dims,
            level=lvl,
            metric=mtr,
            customer_ids=customer_ids,
            return_sparse=False,
        )
        # Đổi tên cột để phân biệt khối
        df_block_renamed = df_block.copy()
        df_block_renamed.columns = [f"{prefix}_{col}" for col in df_block.columns]
        block_dfs.append(df_block_renamed)

    combined_df = pd.concat(block_dfs, axis=1)
    return combined_df


def get_customer_interpretation_metadata(
    dims: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Trích xuất bảng thông tin nhân khẩu học và tổng hợp hành vi của khách hàng.

    LƯU Ý QUAN TRỌNG:
    Bảng này được lưu trữ độc lập để diễn giải cụm (Cluster Profiling & Interpretation),
    TUYỆT ĐỐI KHÔNG DÙNG LÀM ĐẶC TRƯNG HUẤN LUYỆN (ML FEATURES).
    """
    dim_cust = dims["dim_customer"].copy()
    dim_cust = dim_cust.sort_values("CustomerId").set_index("CustomerId")
    return dim_cust
