"""
Load Module for Chinook Analytical Data Layer.
Persists transformed analytical tables (fact_purchase, dim_*) to data/processed/*.parquet (or CSV).
Calculates SHA256 hashes, row counts, and writes data/processed/manifest.json metadata for lineage.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger("chinook_etl.load")


def compute_file_sha256(filepath: str) -> str:
    """Compute standard SHA256 hash of a file on disk."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_df_sha256(df: pd.DataFrame) -> str:
    """
    Compute a deterministic content-based SHA256 hash for a DataFrame.
    Independent of file format compression timestamps.
    """
    # Sort columns and rows deterministically
    sorted_cols = sorted(df.columns)
    df_sorted = df[sorted_cols].copy()
    
    # Convert datetime to ISO string for deterministic hashing
    for col in df_sorted.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df_sorted[col] = df_sorted[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def save_analytical_tables(
    tables: Dict[str, pd.DataFrame],
    audit_logs: List[Dict[str, Any]],
    output_dir: str = "data/processed",
    file_format: str = "parquet",
    config_version: str = "1.0.0",
    extra_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Save analytical tables to disk and generate manifest.json.
    
    Returns:
        manifest: Dictionary containing full load metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).isoformat()
    
    table_manifests = {}
    
    for table_name, df in tables.items():
        content_hash = compute_df_sha256(df)
        
        if file_format.lower() == "parquet":
            filename = f"{table_name}.parquet"
            filepath = os.path.join(output_dir, filename)
            # Use deterministic parquet write
            df.to_parquet(filepath, index=False, engine="pyarrow")
        else:
            filename = f"{table_name}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False, encoding="utf-8")
            
        file_hash = compute_file_sha256(filepath)
        file_size_bytes = os.path.getsize(filepath)
        
        table_manifests[table_name] = {
            "filename": filename,
            "filepath": filepath.replace("\\", "/"),
            "row_count": int(len(df)),
            "col_count": int(len(df.columns)),
            "columns": list(df.columns),
            "content_hash_sha256": content_hash,
            "file_hash_sha256": file_hash,
            "file_size_bytes": file_size_bytes
        }
        logger.info(f"Saved {table_name} -> {filepath} ({len(df)} rows, {file_size_bytes} bytes).")

    # Save audit logs
    audit_df = pd.DataFrame(audit_logs)
    audit_csv_path = os.path.join(output_dir, "etl_audit_log.csv")
    audit_df.to_csv(audit_csv_path, index=False, encoding="utf-8")
    
    audit_json_path = os.path.join(output_dir, "etl_audit_log.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_logs, f, indent=2, ensure_ascii=False)

    # Build manifest
    manifest = {
        "pipeline_name": "chinook_analytical_etl",
        "config_version": config_version,
        "run_timestamp_utc": run_timestamp,
        "output_format": file_format.lower(),
        "total_tables": len(tables),
        "total_fact_purchases": int(len(tables["fact_purchase"])),
        "total_revenue_usd": float(tables["fact_purchase"]["LineTotal"].sum()),
        "tables": table_manifests,
        "audit_summary": {
            "total_rules_evaluated": len(audit_logs),
            "audit_log_csv": audit_csv_path.replace("\\", "/"),
            "audit_log_json": audit_json_path.replace("\\", "/")
        },
        "extra_metadata": extra_metadata or {}
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Written pipeline manifest -> {manifest_path}")
    return manifest
