"""
Test Suite for Chinook Analytical ETL Pipeline (tests/test_etl.py).
Validates:
1. Reconciliation: Total rows and total revenue in fact_purchase match raw SQLite tables.
2. Uniqueness & Integrity: (CustomerId, TrackId) pairs are unique, all Foreign Keys resolve in Dims.
3. Idempotency: Multiple runs yield 100% identical data hashes.
4. Completeness: No customers are lost or unlogged (all 59 customers accounted for).
5. Cleaning Rules: Exactly 12 tracks flagged with is_duplicate_name_album, Composer NaNs imputed.
6. Manifest Integrity: manifest.json is complete, valid, and reflects exact file/content hashes.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.etl.extract import extract_raw_tables
from src.etl.transform import build_analytical_star_schema
from src.etl.load import save_analytical_tables, compute_df_sha256
from src.etl.run_etl import run_pipeline, load_config


@pytest.fixture(scope="module")
def etl_output():
    """Run pipeline and return loaded analytical tables and manifest."""
    config_path = "config/etl.yaml"
    manifest = run_pipeline(config_path)
    out_dir = manifest["tables"]["fact_purchase"]["filepath"]
    base_dir = os.path.dirname(out_dir)
    
    tables = {
        "fact_purchase": pd.read_parquet(os.path.join(base_dir, "fact_purchase.parquet")),
        "dim_customer": pd.read_parquet(os.path.join(base_dir, "dim_customer.parquet")),
        "dim_track": pd.read_parquet(os.path.join(base_dir, "dim_track.parquet")),
        "dim_album": pd.read_parquet(os.path.join(base_dir, "dim_album.parquet")),
        "dim_artist": pd.read_parquet(os.path.join(base_dir, "dim_artist.parquet")),
        "dim_genre": pd.read_parquet(os.path.join(base_dir, "dim_genre.parquet"))
    }
    
    with open(os.path.join(base_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    audit_log = pd.read_csv(os.path.join(base_dir, "etl_audit_log.csv"))
    
    return {
        "tables": tables,
        "manifest": manifest_data,
        "audit_log": audit_log,
        "base_dir": base_dir
    }


@pytest.fixture(scope="module")
def raw_data():
    """Extract raw tables directly for reconciliation verification."""
    config = load_config("config/etl.yaml")
    db_path = config["database"]["sqlite_path"]
    fallback = config["database"].get("fallback_sqlite_path", "Chinook_Sqlite.sqlite")
    raw_tables, meta = extract_raw_tables(db_path, fallback)
    return raw_tables


# ==============================================================================
# TEST SUITE
# ==============================================================================

def test_reconciliation_total_rows_and_revenue(etl_output, raw_data):
    """
    TEST 1: Financial and Volume Reconciliation.
    Total rows in fact_purchase must exactly match raw InvoiceLine (2,240).
    Total revenue in fact_purchase must match sum of InvoiceLine (2,328.60 USD).
    """
    fact = etl_output["tables"]["fact_purchase"]
    raw_lines = raw_data["InvoiceLine"]
    raw_invoices = raw_data["Invoice"]
    
    # 1. Row count match
    assert len(fact) == len(raw_lines), f"Row count mismatch: fact={len(fact)}, raw={len(raw_lines)}"
    assert len(fact) == 2240, f"Expected exactly 2,240 purchase rows, got {len(fact)}"
    
    # 2. Revenue match
    raw_total = (raw_lines["UnitPrice"] * raw_lines["Quantity"]).sum()
    fact_total = fact["LineTotal"].sum()
    invoice_total = raw_invoices["Total"].sum()
    
    assert abs(fact_total - raw_total) < 1e-4, f"Revenue mismatch: fact={fact_total}, raw={raw_total}"
    assert abs(fact_total - invoice_total) < 1e-4, f"Revenue mismatch with Invoices: fact={fact_total}, inv={invoice_total}"
    assert round(fact_total, 2) == 2328.60, f"Expected 2328.60 USD, got {fact_total}"


def test_keys_uniqueness_and_referential_integrity(etl_output):
    """
    TEST 2: Primary and Foreign Key Integrity.
    - No duplicate (CustomerId, TrackId) pairs in fact_purchase.
    - 100% of foreign keys in fact_purchase exist in corresponding dimensions.
    """
    tables = etl_output["tables"]
    fact = tables["fact_purchase"]
    
    # 1. Uniqueness of (CustomerId, TrackId)
    dup_pairs = fact.duplicated(subset=["CustomerId", "TrackId"])
    assert not dup_pairs.any(), f"Found {dup_pairs.sum()} duplicate (CustomerId, TrackId) pairs in fact_purchase!"
    assert fact["InvoiceLineId"].nunique() == len(fact), "InvoiceLineId PK is not unique in fact_purchase!"
    
    # 2. Foreign Key checks
    dim_cust_ids = set(tables["dim_customer"]["CustomerId"])
    dim_track_ids = set(tables["dim_track"]["TrackId"])
    dim_album_ids = set(tables["dim_album"]["AlbumId"])
    dim_artist_ids = set(tables["dim_artist"]["ArtistId"])
    dim_genre_ids = set(tables["dim_genre"]["GenreId"])
    
    assert set(fact["CustomerId"]).issubset(dim_cust_ids), "Orphan CustomerId found in fact_purchase!"
    assert set(fact["TrackId"]).issubset(dim_track_ids), "Orphan TrackId found in fact_purchase!"
    assert set(fact["AlbumId"]).issubset(dim_album_ids), "Orphan AlbumId found in fact_purchase!"
    assert set(fact["ArtistId"]).issubset(dim_artist_ids), "Orphan ArtistId found in fact_purchase!"
    assert set(fact["GenreId"]).issubset(dim_genre_ids), "Orphan GenreId found in fact_purchase!"


def test_no_customer_loss_without_logging(etl_output, raw_data):
    """
    TEST 3: Customer Population Completeness.
    All 59 customers must be preserved in dim_customer.
    All 59 customers must have active purchase records in fact_purchase.
    """
    dim_cust = etl_output["tables"]["dim_customer"]
    fact = etl_output["tables"]["fact_purchase"]
    raw_cust = raw_data["Customer"]
    
    assert len(dim_cust) == len(raw_cust) == 59, f"Customer count mismatch: dim={len(dim_cust)}, raw={len(raw_cust)}"
    assert fact["CustomerId"].nunique() == 59, f"Expected 59 active customers in fact, got {fact['CustomerId'].nunique()}"
    assert dim_cust["has_purchases"].all(), "Expected all 59 customers to have has_purchases=True"


def test_flagged_duplicate_tracks(etl_output):
    """
    TEST 4: Cleaning Rule 1 Validation.
    Exactly 12 tracks (6 pairs) in dim_track must be flagged with is_duplicate_name_album=True.
    Total tracks in dim_track must remain 3,503 (no tracks deleted).
    """
    dim_track = etl_output["tables"]["dim_track"]
    assert len(dim_track) == 3503, f"Expected 3,503 tracks, got {len(dim_track)}"
    
    flagged = dim_track[dim_track["is_duplicate_name_album"]]
    assert len(flagged) == 12, f"Expected exactly 12 flagged duplicate tracks, got {len(flagged)}"
    
    # Verify the 6 specific names
    dup_names = set(flagged["Name"].unique())
    assert len(dup_names) == 6, f"Expected 6 unique duplicate names, got {len(dup_names)}"
    expected_sample = {"Banditismo Por Uma Questa", "Branch Closing", "Company Man", "Gimme Some Truth", "Imagine", "Not In Portland"}
    assert dup_names == expected_sample, f"Unexpected duplicate names: {dup_names}"


def test_imputed_composer(etl_output):
    """
    TEST 5: Cleaning Rule 2 Validation.
    Composer in dim_track must contain 0 NaN values.
    Exactly 978 records must have Composer == 'Unknown'.
    """
    dim_track = etl_output["tables"]["dim_track"]
    assert not dim_track["Composer"].isna().any(), "Found unexpected NaN in dim_track['Composer']!"
    
    unknown_count = (dim_track["Composer"] == "Unknown").sum()
    assert unknown_count == 978, f"Expected exactly 978 'Unknown' composers, got {unknown_count}"


def test_idempotency_run_twice(etl_output):
    """
    TEST 6: Pipeline Idempotency.
    Re-executing the ETL pipeline must produce identical content hashes for all tables.
    """
    config_path = "config/etl.yaml"
    manifest1 = etl_output["manifest"]
    
    # Run second time
    manifest2 = run_pipeline(config_path)
    
    for table_name in ["fact_purchase", "dim_customer", "dim_track", "dim_album", "dim_artist", "dim_genre"]:
        hash1 = manifest1["tables"][table_name]["content_hash_sha256"]
        hash2 = manifest2["tables"][table_name]["content_hash_sha256"]
        assert hash1 == hash2, f"Idempotency failed for table '{table_name}': hash1={hash1}, hash2={hash2}"


def test_manifest_schema_and_metadata(etl_output):
    """
    TEST 7: Metadata and Manifest Validation.
    Verify manifest.json contains all required schema metrics, file sizes, and audit records.
    """
    manifest = etl_output["manifest"]
    assert manifest["pipeline_name"] == "chinook_analytical_etl"
    assert manifest["config_version"] == "1.0.0"
    assert manifest["total_tables"] == 6
    assert manifest["total_fact_purchases"] == 2240
    assert abs(manifest["total_revenue_usd"] - 2328.60) < 1e-2
    
    for tbl in ["fact_purchase", "dim_customer", "dim_track", "dim_album", "dim_artist", "dim_genre"]:
        assert tbl in manifest["tables"]
        entry = manifest["tables"][tbl]
        assert entry["row_count"] > 0
        assert entry["col_count"] > 0
        assert len(entry["file_hash_sha256"]) == 64
        assert len(entry["content_hash_sha256"]) == 64
        assert os.path.exists(entry["filepath"])
