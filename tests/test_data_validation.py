"""Bộ kiểm thử xác thực dữ liệu và đặc trưng (Data & Feature Validation Test Suite).

Kiểm tra:
a) Schema/shape: số hàng = 59 khách hàng; tên cột khớp catalog vocabulary; dtype số; index = CustomerId duy nhất.
b) Chất lượng: không NaN/Inf; không hàng toàn 0 sau pipeline; không giá trị âm trước SVD; liệt kê cột hằng số.
c) Rò rỉ (Leakage): không cột nào là định danh/thuộc tính khách; không dùng nhãn cụm; cách ly demographics.
d) Phân phối: mật độ ma trận khớp số EDA (sai số <= 1% tuyệt đối); norm hàng = 1 (sai số 1e-6); phân phối số mục/khách khớp EDA.
e) Nhất quán nguồn: tổng số cặp (khách, track) từ ma trận track-level khớp fact_purchase (2.240); tổng theo album/genre/artist khớp.
f) Tái lập: chạy pipeline 2 lần cùng seed cho ma trận giống hệt (np.allclose).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from src.features import (
    build_matrix,
    get_catalog_vocabulary,
    load_processed_data,
)
from src.pipeline import build_variant_pipeline, run_feature_pipeline


@pytest.fixture(scope="module")
def processed_data():
    """Nạp bảng sự kiện và các bảng chiều đã xử lý."""
    fact_purchase, dims = load_processed_data("data/processed")
    return fact_purchase, dims


@pytest.fixture(scope="module")
def features_config():
    """Nạp cấu hình features.yaml."""
    with open("config/features.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestDataValidation:
    """Kiểm thử tính đúng đắn, toàn vẹn và chất lượng của dữ liệu đặc trưng."""

    def test_schema_and_shape(self, processed_data):
        """a) Schema/shape: 59 khách hàng, tên cột khớp vocabulary catalog, dtype số, index CustomerId duy nhất."""
        fact_purchase, dims = processed_data

        for level in ["album", "genre", "artist", "track"]:
            raw_mat = build_matrix(fact_purchase, dims, level=level, metric="binary")

            # 1. Số hàng = 59
            assert raw_mat.shape[0] == 59, f"Level {level} có {raw_mat.shape[0]} hàng, kỳ vọng 59."

            # 2. Index là CustomerId duy nhất từ 1 đến 59
            expected_customers = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())
            assert list(raw_mat.index) == expected_customers, f"Index của level {level} không khớp CustomerId."
            assert raw_mat.index.is_unique, f"Index của level {level} bị trùng lặp."

            # 3. Tên cột khớp chính xác với vocabulary catalog đầy đủ
            expected_vocab = get_catalog_vocabulary(dims, level)
            assert list(raw_mat.columns) == expected_vocab, (
                f"Cột của level {level} không khớp với vocabulary catalog."
            )

            # 4. Kiểu dữ liệu phải là số (numeric)
            assert np.issubdtype(raw_mat.values.dtype, np.number), (
                f"Dtype của level {level} không phải là kiểu số."
            )

    def test_data_quality_no_nan_no_inf_no_negative_raw(self, processed_data):
        """b) Chất lượng: không NaN/Inf; không giá trị âm trước SVD; phát hiện cột hằng số."""
        fact_purchase, dims = processed_data

        for level in ["album", "genre", "artist"]:
            raw_mat = build_matrix(fact_purchase, dims, level=level, metric="count")
            arr = raw_mat.values

            # Không NaN, không Inf
            assert not np.isnan(arr).any(), f"Level {level} chứa giá trị NaN."
            assert not np.isinf(arr).any(), f"Level {level} chứa giá trị Inf."

            # Không có giá trị âm trước khi vào SVD
            assert (arr >= 0).all(), f"Level {level} chứa giá trị âm trước SVD."

            # Liệt kê cột hằng số (ví dụ mục unsold có độ lệch chuẩn = 0)
            col_stds = np.std(arr, axis=0)
            constant_cols = [col for col, s in zip(raw_mat.columns, col_stds) if s == 0.0]
            if level == "album":
                # Đúng 43 album chưa từng bán là cột hằng số 0
                assert len(constant_cols) == 43, f"Số cột hằng số của Album là {len(constant_cols)}, kỳ vọng 43."
            elif level == "genre":
                # Đúng 1 thể loại Opera chưa từng bán là cột hằng số 0
                assert len(constant_cols) == 1, f"Số cột hằng số của Genre là {len(constant_cols)}, kỳ vọng 1."
            elif level == "artist":
                # Đúng 110 nghệ sĩ chưa từng bán là cột hằng số 0
                assert len(constant_cols) == 110, f"Số cột hằng số của Artist là {len(constant_cols)}, kỳ vọng 110."

    def test_no_leakage_and_customer_demographics_isolation(self, features_config):
        """c) Rò rỉ: không cột nào là thuộc tính khách hàng hay nhãn cụm."""
        excluded = features_config["excluded_columns_for_features"]

        output_dir = Path("outputs")
        for v_key in features_config["variants"].keys():
            feat_file = output_dir / f"feature_matrix_{v_key}.parquet"
            assert feat_file.exists(), f"Tệp đặc trưng {feat_file} chưa tồn tại."

            df_feat = pd.read_parquet(feat_file)

            # Cột đặc trưng không được chứa bất kỳ thuộc tính cá nhân nào
            for col in df_feat.columns:
                assert col not in excluded, (
                    f"Rò rỉ dữ liệu: Cột nhạy cảm '{col}' xuất hiện trong ma trận đặc trưng {v_key}!"
                )
                assert not col.startswith("cluster"), (
                    f"Rò rỉ nhãn: Cột '{col}' có tiền tố cụm trong {v_key}!"
                )

        # Bảng metadata nhân khẩu học phải được lưu riêng biệt
        meta_file = output_dir / "customer_interpretation_metadata.parquet"
        assert meta_file.exists(), "Tệp customer_interpretation_metadata.parquet chưa được tạo."
        meta_df = pd.read_parquet(meta_file)
        assert meta_df.shape[0] == 59, "Bảng metadata không đủ 59 khách hàng."
        assert "Country" in meta_df.columns, "Bảng metadata thiếu cột Country."

    def test_distribution_metrics_match_eda(self, processed_data):
        """d) Phân phối: mật độ ma trận khớp số EDA (sai số <= 1% tuyệt đối); phân phối mục/khách nằm trong khoảng EDA."""
        fact_purchase, dims = processed_data

        # 1. Kiểm tra mật độ trên tập đã bán (Sold density)
        # Album: EDA = 7.254% (0.07254)
        sold_albums = dims["dim_album"][dims["dim_album"]["is_sold"]]["AlbumId"].tolist()
        album_mat = build_matrix(fact_purchase, dims, level="album", metric="binary")
        album_sold_mat = album_mat.loc[:, sold_albums]
        album_density = (album_sold_mat.values > 0).sum() / album_sold_mat.values.size
        assert abs(album_density - 0.07254) <= 0.01, (
            f"Mật độ Album bán được ({album_density:.5f}) lệch quá 1% so với EDA (0.07254)."
        )

        # Track: EDA = 1.914% (0.01914)
        sold_tracks = dims["dim_track"][dims["dim_track"]["is_sold"]]["TrackId"].tolist()
        track_mat = build_matrix(fact_purchase, dims, level="track", metric="binary")
        track_sold_mat = track_mat.loc[:, sold_tracks]
        track_density = (track_sold_mat.values > 0).sum() / track_sold_mat.values.size
        assert abs(track_density - 0.01914) <= 0.01, (
            f"Mật độ Track bán được ({track_density:.5f}) lệch quá 1% so với EDA (0.01914)."
        )

        # Genre: EDA = 31.073% (0.31073)
        sold_genres = dims["dim_genre"][dims["dim_genre"]["is_sold"]]["GenreId"].tolist()
        genre_mat = build_matrix(fact_purchase, dims, level="genre", metric="binary")
        genre_sold_mat = genre_mat.loc[:, sold_genres]
        genre_density = (genre_sold_mat.values > 0).sum() / genre_sold_mat.values.size
        assert abs(genre_density - 0.31073) <= 0.01, (
            f"Mật độ Genre bán được ({genre_density:.5f}) lệch quá 1% so với EDA (0.31073)."
        )

        # 2. Phân phối số mục mua / khách hàng khớp khoảng biến thiên EDA
        # Album per customer: [18, 27]
        album_per_cust = (album_mat.values > 0).sum(axis=1)
        assert album_per_cust.min() >= 18 and album_per_cust.max() <= 27, (
            f"Số album/khách [{album_per_cust.min()}, {album_per_cust.max()}] vượt khoảng EDA [18, 27]."
        )
        assert 21.0 <= album_per_cust.mean() <= 23.0, "Trung bình album/khách lệch khỏi EDA (22.05)."

        # Genre per customer: [4, 12]
        genre_per_cust = (genre_mat.values > 0).sum(axis=1)
        assert genre_per_cust.min() >= 4 and genre_per_cust.max() <= 12, (
            f"Số genre/khách [{genre_per_cust.min()}, {genre_per_cust.max()}] vượt khoảng EDA [4, 12]."
        )
        assert 7.0 <= genre_per_cust.mean() <= 8.0, "Trung bình genre/khách lệch khỏi EDA (7.46)."

        # Track per customer: [36, 38]
        track_per_cust = (track_mat.values > 0).sum(axis=1)
        assert track_per_cust.min() >= 36 and track_per_cust.max() <= 38, (
            f"Số track/khách [{track_per_cust.min()}, {track_per_cust.max()}] vượt khoảng EDA [36, 38]."
        )

    def test_source_consistency(self, processed_data):
        """e) Nhất quán nguồn: tổng số cặp (khách, track) khớp fact_purchase (2.240); tổng theo album/genre/artist khớp."""
        fact_purchase, dims = processed_data
        n_fact_rows = len(fact_purchase)
        assert n_fact_rows == 2240, f"Bảng fact_purchase có {n_fact_rows} dòng, kỳ vọng 2240."

        # 1. Tổng số cặp (khách, track) từ ma trận track-level phải bằng 2.240
        track_mat = build_matrix(fact_purchase, dims, level="track", metric="binary")
        assert track_mat.values.sum() == 2240, (
            f"Tổng số cặp (khách, track) trong ma trận ({track_mat.values.sum()}) không khớp fact_purchase (2240)."
        )

        # 2. Tổng số lượt mua bài hát từ ma trận album count phải bằng 2.240
        album_count_mat = build_matrix(fact_purchase, dims, level="album", metric="count")
        assert album_count_mat.values.sum() == 2240, (
            f"Tổng số track mua qua ma trận album ({album_count_mat.values.sum()}) không khớp fact_purchase (2240)."
        )

        # 3. Tổng số lượt mua bài hát từ ma trận genre count phải bằng 2.240
        genre_count_mat = build_matrix(fact_purchase, dims, level="genre", metric="count")
        assert genre_count_mat.values.sum() == 2240, (
            f"Tổng số track mua qua ma trận genre ({genre_count_mat.values.sum()}) không khớp fact_purchase (2240)."
        )

        # 4. Tổng số lượt mua bài hát từ ma trận artist count phải bằng 2.240
        artist_count_mat = build_matrix(fact_purchase, dims, level="artist", metric="count")
        assert artist_count_mat.values.sum() == 2240, (
            f"Tổng số track mua qua ma trận artist ({artist_count_mat.values.sum()}) không khớp fact_purchase (2240)."
        )

    def test_row_l2_norm_is_one(self, features_config):
        """d) Norm hàng = 1 (sai số 1e-6) sau chuẩn hóa và không có hàng toàn 0."""
        output_dir = Path("outputs")
        for v_key in features_config["variants"].keys():
            feat_file = output_dir / f"feature_matrix_{v_key}.parquet"
            df_feat = pd.read_parquet(feat_file)
            arr = df_feat.values

            # Không có hàng toàn 0
            assert (np.abs(arr).sum(axis=1) > 0).all(), f"Biến thể {v_key} có hàng toàn 0."

            # Norm hàng = 1.0 với dung sai 1e-6
            norms = np.linalg.norm(arr, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-6), (
                f"Biến thể {v_key} có hàng chưa được chuẩn hóa L2 đơn vị (min={norms.min():.6f}, max={norms.max():.6f})."
            )

    def test_reproducibility_deterministic(self, features_config, processed_data):
        """f) Tái lập: chạy pipeline 2 lần cùng seed cho ma trận giống hệt (np.allclose)."""
        fact_purchase, dims = processed_data
        random_state = features_config.get("random_state", 42)

        for v_key, v_cfg in features_config["variants"].items():
            if v_cfg.get("composite", False):
                raw_df = build_matrix(fact_purchase, dims, level="album", metric="count")
            else:
                raw_df = build_matrix(
                    fact_purchase, dims, level=v_cfg["level"], metric=v_cfg["metric"]
                )

            # Chạy pipeline lần 1
            pipe1 = build_variant_pipeline(v_key, v_cfg, random_state=random_state)
            if v_cfg.get("composite", False):
                genre_df = build_matrix(fact_purchase, dims, level="genre", metric="share")
                X_in = np.hstack([raw_df.values, genre_df.values])
            else:
                X_in = raw_df.values

            res1 = pipe1.fit_transform(X_in)

            # Chạy pipeline lần 2
            pipe2 = build_variant_pipeline(v_key, v_cfg, random_state=random_state)
            res2 = pipe2.fit_transform(X_in)

            # Kiểm tra ma trận giống hệt tuyệt đối
            assert np.allclose(res1, res2, atol=1e-12), (
                f"Biến thể {v_key} không có tính tất định (không tái lập được giữa 2 lần chạy)."
            )
