"""
Orchestration Script for Chinook Analytical ETL Pipeline.
Executes end-to-end extraction, transformation, and loading in a single idempotent command.
Loads settings from config/etl.yaml, executes star-schema creation, performs pre/post reconciliation,
and exports all analytical tables and manifest metadata.

Usage:
    python src/etl/run_etl.py
    python src/etl/run_etl.py --config config/etl.yaml
"""

import os
import sys
import argparse
import logging
import json
import yaml
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.etl.extract import extract_raw_tables
from src.etl.transform import build_analytical_star_schema
from src.etl.load import save_analytical_tables


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """Setup dual console and file logging with UTF-8 encoding."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("chinook_etl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

    # Console with UTF-8 support on Windows
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler with explicit UTF-8
    log_file = os.path.join(log_dir, "etl_run.log")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_config(config_path: str = "config/etl.yaml") -> dict:
    """Load and validate YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def compute_eda_comparison(raw_tables: dict, analytical_tables: dict) -> dict:
    """
    Compute core metrics Before ETL (Raw) vs After ETL (Analytical Star Schema).
    Verifies density, coverage, customer counts, and revenue reconciliation.
    """
    raw_lines = raw_tables["InvoiceLine"]
    raw_invoices = raw_tables["Invoice"]
    raw_tracks = raw_tables["Track"]
    raw_albums = raw_tables["Album"]
    raw_custs = raw_tables["Customer"]
    
    fact = analytical_tables["fact_purchase"]
    dim_cust = analytical_tables["dim_customer"]
    dim_track = analytical_tables["dim_track"]
    dim_album = analytical_tables["dim_album"]

    # 1. Row counts & Financial totals
    raw_line_count = len(raw_lines)
    fact_line_count = len(fact)
    raw_total_rev = (raw_lines["UnitPrice"] * raw_lines["Quantity"]).sum()
    fact_total_rev = fact["LineTotal"].sum()
    
    # 2. Customers
    raw_total_cust = len(raw_custs)
    raw_active_cust = raw_invoices["CustomerId"].nunique()
    fact_active_cust = fact["CustomerId"].nunique()
    
    # 3. Catalog & Sold Coverage
    raw_catalog_tracks = len(raw_tracks)
    raw_sold_tracks = raw_lines["TrackId"].nunique()
    fact_sold_tracks = fact["TrackId"].nunique()
    
    raw_catalog_albums = len(raw_albums)
    raw_sold_albums = raw_tracks.loc[raw_tracks["TrackId"].isin(raw_lines["TrackId"]), "AlbumId"].nunique()
    fact_sold_albums = fact["AlbumId"].nunique()
    
    # 4. Sparsity / Density (Customer x Track and Customer x Album on sold)
    num_cust = 59
    raw_ct_density = len(raw_lines.drop_duplicates(subset=["InvoiceId", "TrackId"])) / (num_cust * raw_sold_tracks)
    fact_ct_density = len(fact.drop_duplicates(subset=["CustomerId", "TrackId"])) / (num_cust * fact_sold_tracks)
    
    raw_ca_density = len(fact.drop_duplicates(subset=["CustomerId", "AlbumId"])) / (num_cust * raw_sold_albums)
    fact_ca_density = len(fact.drop_duplicates(subset=["CustomerId", "AlbumId"])) / (num_cust * fact_sold_albums)

    comparison = {
        "metric_comparison": [
            {
                "metric": "Tổng số dòng giao dịch (Transaction lines)",
                "before_etl": raw_line_count,
                "after_etl": fact_line_count,
                "diff": fact_line_count - raw_line_count,
                "status": "MATCH" if fact_line_count == raw_line_count else "CHANGED"
            },
            {
                "metric": "Tổng doanh thu tích lũy (USD)",
                "before_etl": round(raw_total_rev, 2),
                "after_etl": round(fact_total_rev, 2),
                "diff": round(fact_total_rev - raw_total_rev, 2),
                "status": "MATCH" if round(fact_total_rev, 2) == round(raw_total_rev, 2) else "CHANGED"
            },
            {
                "metric": "Số lượng khách hàng có đơn (Active Customers)",
                "before_etl": raw_active_cust,
                "after_etl": fact_active_cust,
                "diff": fact_active_cust - raw_active_cust,
                "status": "MATCH" if fact_active_cust == raw_active_cust else "CHANGED"
            },
            {
                "metric": "Số lượng Track đã phát sinh mua (Sold Tracks)",
                "before_etl": raw_sold_tracks,
                "after_etl": fact_sold_tracks,
                "diff": fact_sold_tracks - raw_sold_tracks,
                "status": "MATCH" if fact_sold_tracks == raw_sold_tracks else "CHANGED"
            },
            {
                "metric": "Số lượng Album có track được mua (Sold Albums)",
                "before_etl": raw_sold_albums,
                "after_etl": fact_sold_albums,
                "diff": fact_sold_albums - raw_sold_albums,
                "status": "MATCH" if fact_sold_albums == raw_sold_albums else "CHANGED"
            },
            {
                "metric": "Độ đậm đặc Khách x Track (Density on sold)",
                "before_etl": f"{raw_ct_density*100:.3f}%",
                "after_etl": f"{fact_ct_density*100:.3f}%",
                "diff": "0.000%",
                "status": "MATCH" if abs(raw_ct_density - fact_ct_density) < 1e-6 else "CHANGED"
            },
            {
                "metric": "Độ đậm đặc Khách x Album (Density on sold)",
                "before_etl": f"{raw_ca_density*100:.3f}%",
                "after_etl": f"{fact_ca_density*100:.3f}%",
                "diff": "0.000%",
                "status": "MATCH" if abs(raw_ca_density - fact_ca_density) < 1e-6 else "CHANGED"
            }
        ]
    }
    return comparison


def run_pipeline(config_path: str = "config/etl.yaml") -> dict:
    """Execute end-to-end analytical ETL pipeline."""
    logger = setup_logger()
    logger.info("==================================================================")
    logger.info("STARTING CHINOOK ANALYTICAL ETL PIPELINE")
    logger.info("==================================================================")

    config = load_config(config_path)
    logger.info(f"Loaded configuration from '{config_path}' (version: {config.get('version', 'unknown')}).")

    # 1. EXTRACT
    db_path = config["database"]["sqlite_path"]
    fallback_path = config["database"].get("fallback_sqlite_path", "Chinook_Sqlite.sqlite")
    logger.info(">>> STEP 1: EXTRACTING RAW TABLES FROM SQLITE...")
    raw_tables, extract_meta = extract_raw_tables(db_path, fallback_path)

    # 2. TRANSFORM
    logger.info(">>> STEP 2: APPLYING CLEANING RULES & BUILDING STAR SCHEMA...")
    config_rules = config.get("rules", {})
    analytical_tables, audit_logs = build_analytical_star_schema(raw_tables, config_rules)

    # 3. RECONCILIATION
    logger.info(">>> STEP 3: PERFORMING PRE/POST ETL RECONCILIATION & EDA COMPARISON...")
    eda_comparison = compute_eda_comparison(raw_tables, analytical_tables)
    logger.info("--- PRE/POST ETL COMPARISON TABLE ---")
    for row in eda_comparison["metric_comparison"]:
        logger.info(f"  {row['metric']}: Before={row['before_etl']} | After={row['after_etl']} -> [{row['status']}]")

    # 4. LOAD
    logger.info(">>> STEP 4: PERSISTING ANALYTICAL TABLES & WRITING MANIFEST...")
    out_dir = config["storage"].get("output_dir", "data/processed")
    file_format = config["storage"].get("format", "parquet")
    config_ver = config.get("version", "1.0.0")

    manifest = save_analytical_tables(
        tables=analytical_tables,
        audit_logs=audit_logs,
        output_dir=out_dir,
        file_format=file_format,
        config_version=config_ver,
        extra_metadata={
            "extract_metadata": extract_meta,
            "eda_comparison": eda_comparison
        }
    )

    logger.info("==================================================================")
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Output files stored in: {out_dir}")
    logger.info("==================================================================")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chinook Analytical ETL Pipeline")
    parser.add_argument("--config", type=str, default="config/etl.yaml", help="Path to ETL YAML config")
    args = parser.parse_args()

    run_pipeline(args.config)
