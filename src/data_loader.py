"""
Data Loader Module for Chinook Database.
Connects to SQLite in read-only mode, inspects schema, and provides utility functions
to load core entities and joined representations.
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any


DB_PATH = "data/Chinook_Sqlite.sqlite"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite database in read-only mode."""
    # URI read-only connection
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def get_table_names(conn: sqlite3.Connection) -> List[str]:
    """Retrieve all user table names in alphabetical order."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    return [row[0] for row in cursor.fetchall()]


def get_schema_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Extract schema details for all tables:
    - row count
    - columns (name, type, notnull, default_value, pk)
    - foreign keys (id, seq, table, from, to, on_update, on_delete, match)
    """
    cursor = conn.cursor()
    tables = get_table_names(conn)
    schema = {}

    for table in tables:
        # Row count
        cursor.execute(f"SELECT COUNT(*) FROM [{table}];")
        row_count = cursor.fetchone()[0]

        # Columns
        cursor.execute(f"PRAGMA table_info([{table}]);")
        columns = []
        for col in cursor.fetchall():
            columns.append({
                "cid": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default": col[4],
                "pk": bool(col[5])
            })

        # Foreign keys
        cursor.execute(f"PRAGMA foreign_key_list([{table}]);")
        fks = []
        for fk in cursor.fetchall():
            fks.append({
                "id": fk[0],
                "seq": fk[1],
                "to_table": fk[2],
                "from_col": fk[3],
                "to_col": fk[4],
                "on_update": fk[5],
                "on_delete": fk[6],
                "match": fk[7]
            })

        schema[table] = {
            "row_count": row_count,
            "columns": columns,
            "foreign_keys": fks
        }

    return schema


def load_table(table_name: str, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """Load an entire table into a pandas DataFrame."""
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}];", conn)
        return df
    finally:
        if should_close:
            conn.close()


def load_all_tables() -> Dict[str, pd.DataFrame]:
    """Load all tables into a dictionary of DataFrames."""
    conn = get_connection()
    try:
        tables = get_table_names(conn)
        return {t: load_table(t, conn) for t in tables}
    finally:
        conn.close()


def load_master_purchases(conn: sqlite3.Connection = None) -> pd.DataFrame:
    """
    Load the denormalized purchase transaction grain (InvoiceLine joined with
    Invoice, Customer, Track, Album, Artist, Genre, MediaType).
    """
    query = """
    SELECT 
        il.InvoiceLineId,
        il.InvoiceId,
        i.CustomerId,
        i.InvoiceDate,
        i.BillingCountry,
        c.FirstName || ' ' || c.LastName AS CustomerName,
        c.Country AS CustomerCountry,
        c.SupportRepId,
        il.TrackId,
        t.Name AS TrackName,
        t.AlbumId,
        al.Title AS AlbumTitle,
        al.ArtistId,
        ar.Name AS ArtistName,
        t.GenreId,
        g.Name AS GenreName,
        t.MediaTypeId,
        m.Name AS MediaTypeName,
        t.Milliseconds,
        t.Bytes,
        il.UnitPrice,
        il.Quantity,
        (il.UnitPrice * il.Quantity) AS LineTotal
    FROM InvoiceLine il
    JOIN Invoice i      ON il.InvoiceId = i.InvoiceId
    JOIN Customer c     ON i.CustomerId = c.CustomerId
    JOIN Track t        ON il.TrackId = t.TrackId
    JOIN Album al       ON t.AlbumId = al.AlbumId
    JOIN Artist ar      ON al.ArtistId = ar.ArtistId
    JOIN Genre g        ON t.GenreId = g.GenreId
    JOIN MediaType m    ON t.MediaTypeId = m.MediaTypeId;
    """
    should_close = False
    if conn is None:
        conn = get_connection()
        should_close = True
    try:
        df = pd.read_sql_query(query, conn, parse_dates=["InvoiceDate"])
        return df
    finally:
        if should_close:
            conn.close()
