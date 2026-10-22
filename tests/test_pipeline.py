"""Bộ kiểm thử tính đúng đắn và độc lập của Sklearn Pipeline (Pipeline Test Suite).

Kiểm tra:
1. ItemFilter: lọc đúng các cột có ít hơn min_customers trên train, transform áp dụng đúng mask.
2. IDFTransformer: tính đúng trọng số IDF, downweight các mục phổ thông.
3. ComponentDropper: loại bỏ đúng thành phần SVD đầu tiên khi bật cờ.
4. Cloneability: Mọi pipeline (kể cả Composite V6) đều clone được qua sklearn.base.clone.
5. FIT-TRÊN-TRAIN ISOLATION (Không rò rỉ):
   - Fit 2 pipeline trên 2 tập khách hàng khác nhau.
   - Xác nhận tham số học được (ItemFilter.support_, IDF.idf_, SVD.components_) là KHÁC NHAU.
   - Xác nhận transform tập test KHÔNG làm thay đổi bất kỳ tham số đã học nào.
6. Lưới SVD Explained Variance: Tỷ lệ phương sai giải thích tăng đơn điệu theo số chiều {5, 8, 10, 15}.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone

from src.features import build_matrix, load_processed_data
from src.pipeline import (
    BlockScaler,
    ComponentDropper,
    IDFTransformer,
    ItemFilter,
    build_composite_pipeline,
    build_standard_pipeline,
    build_variant_pipeline,
    evaluate_svd_grid,
)


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


class TestPipelineComponents:
    """Kiểm thử từng thành phần transformer độc lập."""

    def test_item_filter_logic(self):
        """ItemFilter phải lọc đúng các cột có số khách hàng >= min_customers."""
        # Tạo ma trận giả lập: 4 khách hàng, 5 cột
        # Cột 0: có 3 khách (>0) -> giữ
        # Cột 1: có 1 khách (>0) -> loại (min_customers=2)
        # Cột 2: có 0 khách (>0) -> loại
        # Cột 3: có 2 khách (>0) -> giữ
        # Cột 4: có 4 khách (>0) -> giữ
        X = np.array([
            [1, 0, 0, 2, 5],
            [1, 1, 0, 0, 3],
            [0, 0, 0, 1, 2],
            [1, 0, 0, 0, 1],
        ], dtype=float)

        filt = ItemFilter(min_customers=2)
        filt.fit(X)

        assert filt.n_features_in_ == 5
        assert filt.n_features_out_ == 3
        assert filt.dropped_features_count_ == 2
        np.testing.assert_array_equal(filt.support_, [True, False, False, True, True])

        X_trans = filt.transform(X)
        assert X_trans.shape == (4, 3)
        np.testing.assert_array_equal(X_trans, X[:, [0, 3, 4]])

    def test_idf_transformer_logic(self):
        """IDFTransformer phải tính đúng công thức IDF và gán trọng số nhỏ hơn cho cột phổ biến."""
        # 4 tài liệu
        # Cột 0: df = 4 (xuất hiện ở cả 4 tài liệu -> phổ biến nhất)
        # Cột 1: df = 1 (chỉ xuất hiện ở 1 tài liệu -> hiếm nhất)
        X = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 5],
        ], dtype=float)

        idf = IDFTransformer(smooth_idf=True)
        idf.fit(X)

        # Với smooth_idf: idf_j = ln((4 + 1)/(df + 1)) + 1
        expected_idf_0 = np.log(5.0 / 5.0) + 1.0  # ln(1) + 1 = 1.0
        expected_idf_1 = np.log(5.0 / 2.0) + 1.0  # ln(2.5) + 1 ~ 1.9163

        assert np.isclose(idf.idf_[0], expected_idf_0)
        assert np.isclose(idf.idf_[1], expected_idf_1)
        assert idf.idf_[0] < idf.idf_[1], "Cột phổ biến phải có trọng số IDF nhỏ hơn cột hiếm."

        X_trans = idf.transform(X)
        assert np.isclose(X_trans[3, 0], 1.0 * expected_idf_0)
        assert np.isclose(X_trans[3, 1], 5.0 * expected_idf_1)

    def test_component_dropper(self):
        """ComponentDropper phải bỏ cột đầu tiên nếu drop_first=True, giữ nguyên nếu False."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

        dropper_true = ComponentDropper(drop_first=True)
        dropper_true.fit(X)
        X_dropped = dropper_true.transform(X)
        assert X_dropped.shape == (2, 2)
        np.testing.assert_array_equal(X_dropped, [[2, 3], [5, 6]])

        dropper_false = ComponentDropper(drop_first=False)
        dropper_false.fit(X)
        X_kept = dropper_false.transform(X)
        assert X_kept.shape == (2, 3)
        np.testing.assert_array_equal(X_kept, X)

    def test_pipeline_cloneable(self, features_config):
        """Mọi pipeline (kể cả composite V6) phải clone được qua sklearn.base.clone."""
        for v_key, v_cfg in features_config["variants"].items():
            pipe = build_variant_pipeline(v_key, v_cfg, random_state=42)
            cloned_pipe = clone(pipe)

            # Kiểm tra clone tạo đối tượng mới chưa fit
            assert cloned_pipe is not pipe
            assert hasattr(cloned_pipe, "named_steps")
            assert "svd" in cloned_pipe.named_steps
            # Đối tượng clone chưa được fit (chưa có components_)
            assert not hasattr(cloned_pipe.named_steps["svd"], "components_")

    def test_fit_on_train_isolation_no_leakage(self, processed_data):
        """c) FIT-TRÊN-TRAIN TEST: Dùng 2 tập khách khác nhau fit 2 pipeline.

        - Xác nhận tham số học được (ItemFilter, IDF, SVD components) KHÁC NHAU giữa 2 tập.
        - Xác nhận transform tập test KHÔNG gọi fit lại và KHÔNG làm thay đổi tham số đã học.
        """
        fact_purchase, dims = processed_data

        # Lấy ma trận Album Binary đầy đủ (59 khách x 347 album)
        album_mat = build_matrix(fact_purchase, dims, level="album", metric="binary")

        # Chia 59 khách hàng thành 2 tập con không trùng lặp:
        # Tập A: 30 khách hàng đầu (CustomerId 1 đến 30)
        # Tập B: 29 khách hàng sau (CustomerId 31 đến 59)
        X_A = album_mat.iloc[:30].values
        X_B = album_mat.iloc[30:].values

        pipe_A = build_standard_pipeline(min_customers=2, use_idf=True, n_components=5, random_state=42)
        pipe_B = build_standard_pipeline(min_customers=2, use_idf=True, n_components=5, random_state=42)

        pipe_A.fit(X_A)
        pipe_B.fit(X_B)

        # 1. Xác nhận ItemFilter học được mask khác nhau giữa tập A và tập B
        # (do các album được mua bởi >= 2 khách ở tập A khác với tập B)
        support_A = pipe_A.named_steps["filter"].support_
        support_B = pipe_B.named_steps["filter"].support_
        assert not np.array_equal(support_A, support_B), (
            "ItemFilter học mask giống hệt nhau giữa 2 tập khách hàng độc lập (kỳ vọng khác nhau)."
        )

        # 2. Xác nhận IDF weights học được khác nhau giữa tập A và tập B
        idf_A = pipe_A.named_steps["idf"].idf_
        idf_B = pipe_B.named_steps["idf"].idf_
        assert len(idf_A) != len(idf_B) or not np.allclose(idf_A, idf_B), (
            "Trọng số IDF giống hệt nhau giữa 2 tập khách hàng độc lập (kỳ vọng khác nhau)."
        )

        # 3. Xác nhận SVD components học được khác nhau giữa tập A và tập B
        svd_comp_A = pipe_A.named_steps["svd"].components_
        svd_comp_B = pipe_B.named_steps["svd"].components_
        assert svd_comp_A.shape != svd_comp_B.shape or not np.allclose(svd_comp_A, svd_comp_B), (
            "SVD components giống hệt nhau giữa 2 tập khách hàng độc lập (kỳ vọng khác nhau)."
        )

        # 4. Kiểm tra tính đóng băng tham số khi gọi transform (No Data Leakage / Idempotent Transform)
        # Lưu bản sao tham số trước khi transform tập B
        support_A_before = np.copy(support_A)
        idf_A_before = np.copy(idf_A)
        svd_comp_A_before = np.copy(svd_comp_A)

        # Gọi transform trên tập kiểm thử X_B
        out_test_B = pipe_A.transform(X_B)

        # Xác nhận output transform có shape hợp lệ
        assert out_test_B.shape == (29, 5)

        # Xác nhận không có bất kỳ tham số nào của pipe_A bị thay đổi hoặc re-fit khi transform tập test
        np.testing.assert_array_equal(pipe_A.named_steps["filter"].support_, support_A_before)
        np.testing.assert_array_equal(pipe_A.named_steps["idf"].idf_, idf_A_before)
        np.testing.assert_array_equal(pipe_A.named_steps["svd"].components_, svd_comp_A_before)

    def test_svd_explained_variance_grid(self, processed_data):
        """Kiểm tra tính đơn điệu tăng dần của phương sai giải thích trên lưới {5, 8, 10, 15}."""
        fact_purchase, dims = processed_data
        album_mat = build_matrix(fact_purchase, dims, level="album", metric="count")

        pipe = build_standard_pipeline(min_customers=2, use_idf=True, n_components=15, random_state=42)
        # Lấy ma trận sau khi filter, idf, norm1
        X_filt = pipe.named_steps["filter"].fit_transform(album_mat.values)
        X_idf = pipe.named_steps["idf"].fit_transform(X_filt)
        X_norm = pipe.named_steps["norm1"].fit_transform(X_idf)

        grid_res = evaluate_svd_grid(X_norm, components_grid=[5, 8, 10, 15], random_state=42)

        # Kiểm tra tính đơn điệu: var(5) < var(8) < var(10) < var(15)
        v5 = grid_res["5"]
        v8 = grid_res["8"]
        v10 = grid_res["10"]
        v15 = grid_res["15"]

        assert 0.0 < v5 < v8 < v10 < v15 <= 1.0, (
            f"Explained variance không tăng đơn điệu: k=5 ({v5}), k=8 ({v8}), k=10 ({v10}), k=15 ({v15})."
        )
