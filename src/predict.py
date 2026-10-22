"""Module dự đoán và gán phân khúc khách hàng mới (Inference / Prediction Module).

Hàm chính:
- assign_cluster(purchases_of_new_customer):
  Nhận vào lịch sử mua hàng của 1 khách hàng mới (danh sách AlbumId hoặc dict {AlbumId: count}),
  biến đổi qua pipeline tiền xử lý (ItemFilter, IDF, L2, SVD, L2), tính khoảng cách tới
  các tâm cụm (centroids), và trả về cụm được gán kèm độ tin cậy.
- assign_clusters_batch(raw_matrix_or_df):
  Gán cụm hàng loạt cho một tập dữ liệu hoặc DataFrame.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Đảm bảo UTF-8 trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Đảm bảo root directory có trong sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd


def load_model_artifact(model_path: Union[str, Path] = "models/final_model.joblib") -> Dict[str, Any]:
    """Nạp file model đã đóng gói cùng toàn bộ metadata."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path.resolve()}")
    return joblib.load(model_path)


def assign_cluster(
    purchases: Union[List[int], Dict[int, int], pd.Series, np.ndarray],
    model_path: Union[str, Path] = "models/final_model.joblib",
) -> Dict[str, Any]:
    """Gán phân khúc cụm và tính độ tin cậy cho một khách hàng mới.
    
    Parameters
    ----------
    purchases : Union[List[int], Dict[int, int], pd.Series, np.ndarray]
        - List[int]: Danh sách các AlbumId mà khách hàng đã mua (ví dụ [1, 5, 23, 142]).
        - Dict[int, int]: Từ điển {AlbumId: số_lượng_track_đã_mua}.
        - pd.Series hoặc np.ndarray: Vector 347 chiều đã theo đúng vocabulary_album_ids.
    model_path : Union[str, Path]
        Đường dẫn tới file joblib lưu trữ mô hình.
        
    Returns
    -------
    Dict[str, Any]
        {
            "cluster_id": int,
            "cluster_name": str,
            "confidence": float,
            "distance_to_centroid": float,
            "all_distances": Dict[int, float],
            "matched_albums_count": int,
        }
    """
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    centroids_norm = artifact["centroids_norm"]
    cluster_names = artifact["cluster_names"]
    vocab = artifact["vocabulary_album_ids"]
    vocab_set = set(vocab)
    col_to_idx = {col: i for i, col in enumerate(vocab)}

    n_features = len(vocab)
    row_vector = np.zeros(n_features, dtype=float)
    matched_count = 0

    if isinstance(purchases, (list, tuple, set)):
        for album_id in purchases:
            if album_id in vocab_set:
                idx = col_to_idx[album_id]
                row_vector[idx] += 1.0
                matched_count += 1
    elif isinstance(purchases, dict):
        for album_id, count in purchases.items():
            if album_id in vocab_set:
                idx = col_to_idx[album_id]
                row_vector[idx] += float(count)
                matched_count += 1
    elif isinstance(purchases, (pd.Series, np.ndarray)):
        arr = np.asarray(purchases).ravel()
        if len(arr) == n_features:
            row_vector = arr.astype(float)
            matched_count = int(np.sum(row_vector > 0))
        else:
            raise ValueError(
                f"Độ dài vector đầu vào ({len(arr)}) không khớp với số cột vocabulary ({n_features})."
            )
    else:
        raise TypeError(f"Kiểu dữ liệu purchases không được hỗ trợ: {type(purchases)}")

    # Đưa qua Pipeline tiền xử lý và giảm chiều SVD
    X_input = row_vector.reshape(1, -1)
    X_svd = pipeline.transform(X_input)  # Shape (1, 5), vector đã được chuẩn hóa L2

    # Tính khoảng cách Cosine tới từng tâm cụm
    # Với 2 vector chuẩn hóa L2: cos_dist = 1 - (u . v)
    cos_sims = np.dot(X_svd, centroids_norm.T).ravel()  # (k,)
    cos_dists = 1.0 - cos_sims

    best_cluster = int(np.argmax(cos_sims))
    best_dist = float(cos_dists[best_cluster])
    confidence = float(np.clip(cos_sims[best_cluster], 0.0, 1.0))

    all_dists = {c: round(float(cos_dists[c]), 4) for c in range(len(cos_dists))}

    return {
        "cluster_id": best_cluster,
        "cluster_name": cluster_names.get(best_cluster, f"Cluster {best_cluster}"),
        "confidence": round(confidence, 4),
        "distance_to_centroid": round(best_dist, 4),
        "all_distances": all_dists,
        "matched_albums_count": matched_count,
    }


def assign_clusters_batch(
    raw_df_or_matrix: Union[pd.DataFrame, np.ndarray],
    model_path: Union[str, Path] = "models/final_model.joblib",
) -> pd.DataFrame:
    """Gán cụm hàng loạt cho một tập dữ liệu ma trận thô."""
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    model = artifact["model"]
    centroids_norm = artifact["centroids_norm"]
    cluster_names = artifact["cluster_names"]

    if isinstance(raw_df_or_matrix, pd.DataFrame):
        X_raw = raw_df_or_matrix.values
        index = raw_df_or_matrix.index
    else:
        X_raw = np.asarray(raw_df_or_matrix)
        index = range(len(X_raw))

    X_svd = pipeline.transform(X_raw)
    preds = model.predict(X_svd)

    sims = np.dot(X_svd, centroids_norm.T)
    cos_dists = 1.0 - sims
    assigned_sims = np.array([sims[i, preds[i]] for i in range(len(preds))])
    assigned_dists = np.array([cos_dists[i, preds[i]] for i in range(len(preds))])

    df_out = pd.DataFrame({
        "cluster": preds,
        "cluster_name": [cluster_names.get(p, f"Cluster {p}") for p in preds],
        "confidence": np.round(np.clip(assigned_sims, 0.0, 1.0), 4),
        "distance_to_centroid": np.round(assigned_dists, 4),
    }, index=index)
    df_out.index.name = "CustomerId"

    return df_out


if __name__ == "__main__":
    # Test mẫu với 1 khách hàng mới
    sample_albums = [1, 2, 4, 120, 141]  # Mua các album Rock/Metal
    res = assign_cluster(sample_albums)
    print("Dự đoán mẫu cho khách hàng mới:")
    print(res)
