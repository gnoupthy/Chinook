"""Test tái lập mô hình và kiểm tra tính toàn vẹn (Reproducibility & Integrity Tests).

Kiểm tra:
1. File models/final_model.joblib tồn tại, chứa đầy đủ pipeline, model, vocabulary,
   phiên bản thư viện và hash dữ liệu.
2. Nạp model đã lưu, predict lại toàn bộ 59 khách hàng: nhãn phải trùng 100%
   với outputs/customer_clusters.csv.
3. Chạy lại pipeline từ đầu trên ma trận thô với random_state=42 cho cùng phân hoạch
   nhãn (Adjusted Rand Index == 1.0).
4. Kiểm tra hàm assign_cluster(purchases_of_new_customer) hoạt động chính xác
   với các định dạng đầu vào (list, dict, array) và trả về độ tin cậy hợp lệ.
"""

import hashlib
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Đảm bảo root directory có trong sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import build_matrix, load_processed_data
from src.pipeline import build_variant_pipeline
from src.predict import assign_cluster, assign_clusters_batch, load_model_artifact


@pytest.fixture(scope="module")
def model_artifact():
    model_path = Path("models/final_model.joblib")
    assert model_path.exists(), f"File mô hình không tồn tại tại {model_path}"
    return load_model_artifact(model_path)


@pytest.fixture(scope="module")
def expected_clusters_df():
    csv_path = Path("outputs/customer_clusters.csv")
    assert csv_path.exists(), f"File customer_clusters.csv không tồn tại tại {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def processed_data():
    return load_processed_data("data/processed")


def test_model_artifact_structure(model_artifact):
    """Kiểm tra cấu trúc và tính đầy đủ của file joblib."""
    required_keys = [
        "config_id",
        "pipeline",
        "model",
        "centroids_norm",
        "cluster_names",
        "vocabulary_album_ids",
        "album_id_to_title",
        "n_components",
        "k",
        "random_state",
        "data_hash_md5",
        "library_versions",
        "metrics_full_data",
        "metrics_validation",
    ]
    for key in required_keys:
        assert key in model_artifact, f"Thiếu key '{key}' trong model_artifact"

    assert model_artifact["k"] == 5
    assert model_artifact["n_components"] == 5
    assert len(model_artifact["vocabulary_album_ids"]) == 347
    assert len(model_artifact["cluster_names"]) == 5

    # Kiểm tra hash của file dữ liệu
    fact_path = Path("data/processed/fact_purchase.parquet")
    with open(fact_path, "rb") as f:
        current_hash = hashlib.md5(f.read()).hexdigest()
    assert model_artifact["data_hash_md5"] == current_hash, "MD5 data hash không khớp với dữ liệu hiện tại"


def test_reproducibility_on_all_59_customers(model_artifact, expected_clusters_df, processed_data):
    """Kiểm tra: predict lại 59 khách hàng phải trùng khớp 100% với customer_clusters.csv."""
    fact_purchase, dims = processed_data
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())

    raw_df = build_matrix(
        fact_purchase, dims, level="album", metric="count", customer_ids=customer_ids, return_sparse=False
    )

    pred_df = assign_clusters_batch(raw_df, model_path="models/final_model.joblib")

    # 1. Kiểm tra số lượng khách hàng
    assert len(pred_df) == 59
    assert len(expected_clusters_df) == 59

    # 2. Kiểm tra nhãn cụm trùng khớp 100% (từng dòng)
    expected_labels = expected_clusters_df.set_index("CustomerId")["cluster"]
    actual_labels = pred_df["cluster"]

    matches = (expected_labels == actual_labels).sum()
    match_rate = matches / 59.0

    assert match_rate == 1.0, f"Tỷ lệ trùng khớp nhãn cụm là {match_rate*100:.2f}% (kỳ vọng 100%)"

    # 3. Kiểm tra độ tin cậy và khoảng cách tới tâm cụm
    expected_dists = expected_clusters_df.set_index("CustomerId")["distance_to_centroid"]
    actual_dists = pred_df["distance_to_centroid"]
    np.testing.assert_allclose(actual_dists.values, expected_dists.values, atol=1e-4)


def test_pipeline_rerun_from_scratch(expected_clusters_df, processed_data):
    """Kiểm tra: huấn luyện lại pipeline từ đầu với random_state=42 phải ra cùng kết quả."""
    fact_purchase, dims = processed_data
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())

    raw_df = build_matrix(
        fact_purchase, dims, level="album", metric="count", customer_ids=customer_ids, return_sparse=False
    )

    v_cfg = {
        "level": "album",
        "metric": "count",
        "use_idf": True,
        "min_customers": 2,
        "n_components": 5,
        "drop_first_component": False,
    }
    pipeline = build_variant_pipeline("V2_album_count", v_cfg, random_state=42)
    X_svd = pipeline.fit_transform(raw_df.values)

    km = KMeans(n_clusters=5, n_init=30, random_state=42).fit(X_svd)
    scratch_labels = km.labels_
    expected_labels = expected_clusters_df["cluster"].values

    # Đo ARI giữa nhãn huấn luyện lại từ đầu và nhãn đã lưu
    ari = adjusted_rand_score(scratch_labels, expected_labels)
    assert ari == 1.0, f"Huấn luyện lại từ đầu cho ARI = {ari} (kỳ vọng 1.0 - phân hoạch trùng hoàn toàn)"


def test_single_customer_predict_function():
    """Kiểm tra hàm assign_cluster cho khách hàng mới hoạt động chính xác."""
    # Test 1: Khách hàng mua các album cụ thể
    sample_albums = [1, 2, 4, 120, 141]
    res1 = assign_cluster(sample_albums)

    assert "cluster_id" in res1
    assert "cluster_name" in res1
    assert "confidence" in res1
    assert "distance_to_centroid" in res1
    assert 0 <= res1["cluster_id"] < 5
    assert 0.0 <= res1["confidence"] <= 1.0
    assert res1["distance_to_centroid"] >= 0.0
    assert res1["matched_albums_count"] == 5

    # Test 2: Khách hàng truyền vào dictionary {AlbumId: count}
    sample_dict = {1: 3, 2: 2, 55: 1}
    res2 = assign_cluster(sample_dict)
    assert 0 <= res2["cluster_id"] < 5

    # Test 3: Khách hàng không mua album nào trong vocabulary
    sample_empty = [99999, 88888]
    res3 = assign_cluster(sample_empty)
    assert res3["matched_albums_count"] == 0
