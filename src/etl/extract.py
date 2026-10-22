"""
Extraction Module for Chinook ETL Pipeline.
Connects to SQLite database in read-only mode and extracts raw tables into pandas DataFrames.
Tracks extracted row counts, column counts, and execution metrics.
"""

import os
import sqlite3
import logging
from typing import Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger("chinook_etl.extract")

REQUIRED_TABLES = [
    "Customer",
    "Invoice",
    "InvoiceLine",
    "Track",
    "Album",
    "Artist",
    "Genre",
    "MediaType"
]


def resolve_db_path(db_path: str, fallback_path: str = "Chinook_Sqlite.sqlite") -> str:
    """Resolve database path, checking primary and fallback paths."""
    if os.path.exists(db_path):
        return db_path
    if os.path.exists(fallback_path):
        logger.warning(f"Primary DB path '{db_path}' not found. Using fallback '{fallback_path}'.")
        return fallback_path
    raise FileNotFoundError(f"SQLite database not found at '{db_path}' or '{fallback_path}'.")


def get_readonly_connection(db_path: str) -> sqlite3.Connection:
    """Connect to SQLite database strictly in read-only mode using URI syntax."""
    resolved_path = os.path.abspath(db_path)
    # Windows path forward-slash normalization for SQLite URI
    uri_path = resolved_path.replace("\\", "/")
    uri = f"file:///{uri_path}?mode=ro"
    logger.info(f"Connecting to SQLite read-only URI: {uri}")
    return sqlite3.connect(uri, uri=True)


def extract_raw_tables(db_path: str, fallback_path: str = "Chinook_Sqlite.sqlite") -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Extract all required raw tables from SQLite database in read-only mode.
    
    Returns:
        raw_tables: Dictionary of table_name -> pd.DataFrame
        extraction_metadata: Dictionary of extracted table statistics and row counts
    """
    resolved_path = resolve_db_path(db_path, fallback_path)
    conn = get_readonly_connection(resolved_path)
    
    raw_tables = {}
    table_stats = {}
    
    try:
        cursor = conn.cursor()
        for table in REQUIRED_TABLES:
            query = f"SELECT * FROM [{table}];"
            df = pd.read_sql_query(query, conn)
            
            # Specific date parsing for Invoice
            if table == "Invoice" and "InvoiceDate" in df.columns:
                df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
                
            raw_tables[table] = df
            table_stats[table] = {
                "row_count": len(df),
                "col_count": len(df.columns),
                "columns": list(df.columns)
            }
            logger.info(f"Extracted table '{table}': {len(df)} rows, {len(df.columns)} columns.")
            
    finally:
        conn.close()
        logger.info("Closed read-only database connection.")
        
    metadata = {
        "db_path": resolved_path,
        "tables_extracted": len(raw_tables),
        "table_stats": table_stats
    }
    
    return raw_tables, metadata
