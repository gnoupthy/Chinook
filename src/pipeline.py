"""Module Pipeline tiền xử lý và giảm chiều đặc trưng (Scikit-Learn Pipeline).

Bao gồm:
1. ItemFilter: Lọc bỏ các mục hiếm (xuất hiện ở ít hơn min_customers trên tập train).
2. IDFTransformer: Trọng số Inverse Document Frequency giảm ảnh hưởng mục phổ thông.
3. ComponentDropper: Tùy chọn bỏ thành phần SVD đầu tiên (thường đại diện độ phổ biến chung).
4. BlockScaler: Nhân trọng số khối cho biến thể ghép (Composite).
5. TruncatedSVD & Normalizer (L2): Giảm chiều và chuẩn hóa độ dài vector đơn vị
   để khoảng cách Euclid tương đương khoảng cách Cosine.

Mọi transformer đều kế thừa BaseEstimator, TransformerMixin và tương thích 100% với sklearn.base.clone.
Các trạng thái học được (fit) chỉ học trên tập train khi đánh giá (No Data Leakage).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from src.features import (
    build_composite_raw_matrix,
    build_matrix,
    get_customer_interpretation_metadata,
    load_processed_data,
)


class ItemFilter(BaseEstimator, TransformerMixin):
    """Transformer lọc bỏ các cột (mục) có tần suất khách hàng thấp hơn ngưỡng min_customers.

    Trạng thái được học CHỈ TRÊN TẬP TRAIN (fit):
    - Đếm số lượng khách hàng có giá trị > 0 cho từng cột.
    - Lưu lại mặt nạ support_ (True nếu >= min_customers).
    - Cột nào bị loại trên train sẽ bị loại tương ứng trên test khi gọi transform.
    """

    def __init__(self, min_customers: int = 2):
        self.min_customers = min_customers

    def fit(self, X: Any, y: Optional[Any] = None) -> "ItemFilter":
        X_arr = np.asarray(X)
        self.n_features_in_ = X_arr.shape[1]
        # Đếm số khách hàng có tương tác > 0 trên từng cột
        counts = np.asarray((X_arr > 0).sum(axis=0)).ravel()
        self.support_ = counts >= self.min_customers
        self.n_features_out_ = int(np.sum(self.support_))
        self.dropped_features_count_ = int(self.n_features_in_ - self.n_features_out_)
        self.active_customer_counts_ = counts
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"ItemFilter kỳ vọng {self.n_features_in_} cột đầu vào, nhưng nhận được {X_arr.shape[1]} cột."
            )
        return X_arr[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_


class IDFTransformer(BaseEstimator, TransformerMixin):
    """Transformer tính trọng số Inverse Document Frequency (IDF).

    idf_j = ln((N + 1) / (df_j + 1)) + 1.0 (smooth_idf=True)
    Học trên tập train (N và df). Phép transform nhân vector cột với idf_.
    """

    def __init__(self, smooth_idf: bool = True, use_idf: bool = True):
        self.smooth_idf = smooth_idf
        self.use_idf = use_idf

    def fit(self, X: Any, y: Optional[Any] = None) -> "IDFTransformer":
        X_arr = np.asarray(X)
        self.n_features_in_ = X_arr.shape[1]
        if not self.use_idf:
            self.idf_ = np.ones(self.n_features_in_, dtype=float)
            return self

        N = X_arr.shape[0]
        df = np.asarray((X_arr > 0).sum(axis=0)).ravel()
        if self.smooth_idf:
            self.idf_ = np.log((N + 1.0) / (df + 1.0)) + 1.0
        else:
            self.idf_ = np.log(np.maximum(N / np.maximum(df, 1.0), 1.0)) + 1.0
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"IDFTransformer kỳ vọng {self.n_features_in_} cột đầu vào, nhưng nhận được {X_arr.shape[1]} cột."
            )
        if not self.use_idf:
            return X_arr.copy()
        return X_arr * self.idf_


class BlockScaler(BaseEstimator, TransformerMixin):
    """Transformer nhân tỷ lệ trọng số khối cho biến thể ghép (Composite)."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def fit(self, X: Any, y: Optional[Any] = None) -> "BlockScaler":
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        return np.asarray(X) * self.weight


class ComponentDropper(BaseEstimator, TransformerMixin):
    """Transformer loại bỏ thành phần SVD đầu tiên (nếu drop_first=True).

    Thành phần thứ nhất của SVD thường mang thông tin về độ phổ biến chung
    của toàn bộ catalog (general volume / intercept). Loại bỏ nó giúp tập trung
    vào các chiều phân hóa thị hiếu tương đối giữa các cụm.
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first

    def fit(self, X: Any, y: Optional[Any] = None) -> "ComponentDropper":
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if self.drop_first and X_arr.shape[1] > 1:
            return X_arr[:, 1:]
        return X_arr


def build_standard_pipeline(
    min_customers: int = 2,
    use_idf: bool = True,
    n_components: int = 10,
    drop_first_component: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Xây dựng Scikit-Learn Pipeline tiêu chuẩn cho các biến thể đơn lẻ (V1 - V5).

    Luồng xử lý:
    ItemFilter -> [IDFTransformer] -> Normalizer(L2) -> TruncatedSVD -> [ComponentDropper] -> Normalizer(L2)
    """
    steps = [
        ("filter", ItemFilter(min_customers=min_customers)),
        ("idf", IDFTransformer(smooth_idf=True, use_idf=use_idf)),
        ("norm1", Normalizer(norm="l2")),
        (
            "svd",
            TruncatedSVD(
                n_components=n_components,
                algorithm="randomized",
                random_state=random_state,
            ),
        ),
        ("dropper", ComponentDropper(drop_first=drop_first_component)),
        ("norm2", Normalizer(norm="l2")),
    ]
    return Pipeline(steps)


def build_composite_pipeline(
    blocks_config: List[Dict[str, Any]],
    n_components: int = 10,
    drop_first_component: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Xây dựng Scikit-Learn Pipeline cho biến thể ghép khối (V6: Composite Album + Genre).

    Mỗi khối được tiền xử lý riêng (ItemFilter -> [IDF] -> Normalizer(L2) -> BlockScaler),
    ghép nối qua ColumnTransformer, sau đó qua SVD và Normalizer cuối cùng.
    """
    # Khối 1: Album (347 cột đầu)
    album_cfg = blocks_config[0]
    album_pipe = Pipeline(
        [
            ("filter", ItemFilter(min_customers=album_cfg.get("min_customers", 2))),
            ("idf", IDFTransformer(smooth_idf=True, use_idf=album_cfg.get("use_idf", True))),
            ("norm", Normalizer(norm="l2")),
            ("scale", BlockScaler(weight=album_cfg.get("weight", 0.7))),
        ]
    )

    # Khối 2: Genre (25 cột sau)
    genre_cfg = blocks_config[1]
    genre_pipe = Pipeline(
        [
            ("filter", ItemFilter(min_customers=genre_cfg.get("min_customers", 1))),
            ("norm", Normalizer(norm="l2")),
            ("scale", BlockScaler(weight=genre_cfg.get("weight", 0.3))),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("album", album_pipe, slice(0, 347)),
            ("genre", genre_pipe, slice(347, 347 + 25)),
        ]
    )

    full_pipe = Pipeline(
        [
            ("blocks", preprocessor),
            ("norm_composite", Normalizer(norm="l2")),
            (
                "svd",
                TruncatedSVD(
                    n_components=n_components,
                    algorithm="randomized",
                    random_state=random_state,
                ),
            ),
            ("dropper", ComponentDropper(drop_first=drop_first_component)),
            ("final_norm", Normalizer(norm="l2")),
        ]
    )

    return full_pipe


def build_variant_pipeline(
    variant_key: str,
    variant_config: Dict[str, Any],
    random_state: int = 42,
) -> Pipeline:
    """Tạo Pipeline tương ứng cho một biến thể cụ thể từ cấu hình yaml."""
    is_composite = variant_config.get("composite", False)
    n_components = variant_config.get("n_components", 10)
    drop_first = variant_config.get("drop_first_component", False)

    if is_composite:
        return build_composite_pipeline(
            blocks_config=variant_config["blocks"],
            n_components=n_components,
            drop_first_component=drop_first,
            random_state=random_state,
        )
    else:
        min_customers = variant_config.get("min_customers", 2)
        use_idf = variant_config.get("use_idf", True)
        return build_standard_pipeline(
            min_customers=min_customers,
            use_idf=use_idf,
            n_components=n_components,
            drop_first_component=drop_first,
            random_state=random_state,
        )


def evaluate_svd_grid(
    X_processed: np.ndarray,
    components_grid: List[int] = [5, 8, 10, 15],
    random_state: int = 42,
) -> Dict[str, float]:
    """Đánh giá tỷ lệ phương sai giải thích tích lũy (Cumulative Explained Variance) trên lưới số chiều."""
    results = {}
    max_k = min(max(components_grid), X_processed.shape[1] - 1)
    for k in components_grid:
        if k <= max_k:
            svd = TruncatedSVD(n_components=k, random_state=random_state)
            svd.fit(X_processed)
            cum_var = float(np.sum(svd.explained_variance_ratio_))
            results[str(k)] = round(cum_var, 4)
        else:
            results[str(k)] = None
    return results


def run_feature_pipeline(
    config_path: Union[str, Path] = "config/features.yaml"
) -> Dict[str, Any]:
    """Hàm điều phối toàn trình: Nạp dữ liệu, tạo ma trận, chạy pipeline và lưu trữ hiện vật."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_dir = Path(config.get("data_dir", "data/processed"))
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    random_state = config.get("random_state", 42)
    svd_grid = config.get("svd_components_grid", [5, 8, 10, 15])

    fact_purchase, dims = load_processed_data(data_dir)
    customer_ids = sorted(dims["dim_customer"]["CustomerId"].unique().tolist())

    metrics_report = {
        "random_state": random_state,
        "n_customers": len(customer_ids),
        "variants": {},
    }

    print("==================================================================")
    print("STARTING CHINOOK FEATURE ENGINEERING & PIPELINE (PHASE C1)")
    print("==================================================================")

    for v_key, v_cfg in config["variants"].items():
        print(f"\n>>> Processing Variant: {v_key} ({v_cfg['name']})...")
        is_composite = v_cfg.get("composite", False)

        # 1. Xây dựng ma trận thô (Raw Matrix)
        if is_composite:
            raw_df = build_composite_raw_matrix(
                fact_purchase=fact_purchase,
                dims=dims,
                blocks_config=v_cfg["blocks"],
                customer_ids=customer_ids,
            )
        else:
            raw_df = build_matrix(
                fact_purchase=fact_purchase,
                dims=dims,
                level=v_cfg["level"],
                metric=v_cfg["metric"],
                customer_ids=customer_ids,
                return_sparse=False,
            )

        # Lưu ma trận thô phục vụ kiểm định và audit
        raw_output_path = output_dir / f"raw_matrix_{v_key}.parquet"
        raw_df.to_parquet(raw_output_path)

        # 2. Xây dựng Pipeline Sklearn
        pipeline = build_variant_pipeline(v_key, v_cfg, random_state=random_state)

        # 3. Fit và Transform toàn bộ tập dữ liệu
        X_raw = raw_df.values
        X_transformed = pipeline.fit_transform(X_raw)

        # Tính toán thông tin ItemFilter
        if not is_composite:
            item_filter_step = pipeline.named_steps["filter"]
            dropped_cols = item_filter_step.dropped_features_count_
            kept_cols = item_filter_step.n_features_out_
            svd_step = pipeline.named_steps["svd"]
            exp_var_actual = float(np.sum(svd_step.explained_variance_ratio_))
            # Đánh giá lưới SVD trên ma trận sau IDF & L2-norm
            X_filtered = item_filter_step.transform(X_raw)
            X_idf = pipeline.named_steps["idf"].transform(X_filtered)
            X_norm1 = pipeline.named_steps["norm1"].transform(X_idf)
            grid_variance = evaluate_svd_grid(X_norm1, svd_grid, random_state=random_state)
        else:
            # Composite variant
            block_transformer = pipeline.named_steps["blocks"]
            album_filter = block_transformer.named_transformers_["album"].named_steps["filter"]
            genre_filter = block_transformer.named_transformers_["genre"].named_steps["filter"]
            dropped_cols = (
                album_filter.dropped_features_count_ + genre_filter.dropped_features_count_
            )
            kept_cols = album_filter.n_features_out_ + genre_filter.n_features_out_
            svd_step = pipeline.named_steps["svd"]
            exp_var_actual = float(np.sum(svd_step.explained_variance_ratio_))
            X_blocks = block_transformer.transform(X_raw)
            X_norm_comp = pipeline.named_steps["norm_composite"].transform(X_blocks)
            grid_variance = evaluate_svd_grid(X_norm_comp, svd_grid, random_state=random_state)

        # 4. Lưu ma trận đặc trưng thu gọn (Feature Matrix)
        n_dims = X_transformed.shape[1]
        col_names = [f"dim_{i+1}" for i in range(n_dims)]
        feat_df = pd.DataFrame(X_transformed, index=raw_df.index, columns=col_names)
        feat_df.index.name = "CustomerId"

        feat_output_path = output_dir / f"feature_matrix_{v_key}.parquet"
        feat_df.to_parquet(feat_output_path)

        # Tính mật độ ma trận thô
        raw_density = float((raw_df.values > 0).sum() / raw_df.values.size)
        raw_non_zeros = int((raw_df.values > 0).sum())

        metrics_report["variants"][v_key] = {
            "name": v_cfg["name"],
            "raw_shape": list(raw_df.shape),
            "raw_density": round(raw_density, 5),
            "raw_non_zeros": raw_non_zeros,
            "latent_shape": list(feat_df.shape),
            "dropped_columns_by_filter": dropped_cols,
            "retained_columns": kept_cols,
            "actual_explained_variance": round(exp_var_actual, 4),
            "svd_grid_explained_variance": grid_variance,
            "feature_matrix_path": str(feat_output_path),
            "raw_matrix_path": str(raw_output_path),
        }

        print(
            f"   Raw Shape: {raw_df.shape} | Density: {raw_density*100:.2f}% | Filter Dropped: {dropped_cols} cols"
        )
        print(
            f"   Latent Shape: {feat_df.shape} | Explained Var: {exp_var_actual*100:.2f}% | Saved: {feat_output_path}"
        )

    # 5. Lưu bảng thông tin nhân khẩu học giải thích cụm (Interpretation Metadata)
    metadata_df = get_customer_interpretation_metadata(dims)
    metadata_path = output_dir / "customer_interpretation_metadata.parquet"
    metadata_df.to_parquet(metadata_path)
    print(f"\nSaved customer interpretation metadata -> {metadata_path}")

    # 6. Ghi file tổng hợp metric
    metrics_path = output_dir / "feature_pipeline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2, ensure_ascii=False)
    print(f"Saved pipeline metrics report -> {metrics_path}")

    print("\n==================================================================")
    print("FEATURE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==================================================================")

    return metrics_report


if __name__ == "__main__":
    run_feature_pipeline()
