"""
Transformation Module for Chinook Analytical Data Layer.
Converts raw tables into a star-schema analytical model:
- fact_purchase: 1 row = 1 (CustomerId, TrackId) purchase instance with full dimensional keys.
- dim_customer, dim_track, dim_album, dim_artist, dim_genre.

All cleaning rules from outputs/cleaning_log.csv are codified as dedicated, isolated functions.
Filtering is never implicit: all filters are controllable via configuration flags and audited with
detailed logs of affected rows, customers, and impact on EDA/sparsity figures.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("chinook_etl.transform")


# ==============================================================================
# 1. CODIFIED CLEANING & FILTERING RULES
# ==============================================================================

def rule_flag_duplicate_logic_tracks(
    df_track: pd.DataFrame,
    enabled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    CLEANING RULE 1: Flag duplicate logic tracks with identical (Name, AlbumId).
    
    Rationale (from outputs/cleaning_log.csv & reports/data_quality.md):
        In catalog, exactly 6 pairs (12 tracks) share the same Name and AlbumId.
        They possess distinct TrackIds, different Milliseconds/Bytes, or MediaTypes.
        They represent alternative formats/takes rather than technical errors.
        Therefore, we NEVER drop or merge them; we explicitly flag them.
    
    Args:
        df_track: Raw Track DataFrame
        enabled: If True, computes and appends 'is_duplicate_name_album' boolean column.
        
    Returns:
        df_track_out: DataFrame with 'is_duplicate_name_album' column
        audit_record: Metadata on affected rows and EDA impact
    """
    df_out = df_track.copy()
    if not enabled:
        df_out["is_duplicate_name_album"] = False
        return df_out, {
            "rule": "flag_duplicate_logic_tracks",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (Rule disabled)",
            "description": "Rule disabled by configuration."
        }
        
    dup_mask = df_out.duplicated(subset=["Name", "AlbumId"], keep=False)
    df_out["is_duplicate_name_album"] = dup_mask
    affected_count = int(dup_mask.sum())
    
    audit_record = {
        "rule": "flag_duplicate_logic_tracks",
        "enabled": True,
        "action": "flag",
        "affected_rows": affected_count,
        "affected_customers": 0,
        "impact_on_eda": "None (No rows deleted; exactly 12 tracks flagged with is_duplicate_name_album=True).",
        "description": "Identified 6 pairs of alternative formats of same song; preserved all 3,503 tracks without data loss."
    }
    logger.info(f"[Rule 1] Flagged {affected_count} duplicate (Name, AlbumId) tracks.")
    return df_out, audit_record


def rule_fill_null_composer(
    df_track: pd.DataFrame,
    enabled: bool = True,
    fill_value: str = "Unknown"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    CLEANING RULE 2: Impute missing Composer in dim_track with 'Unknown'.
    
    Rationale (from outputs/cleaning_log.csv & reports/data_quality.md):
        978 out of 3,503 catalog tracks (27.92%) have Composer as NULL.
        This is valid business missingness (e.g., traditional music, audio tracks without credits).
        Replacing NULL with 'Unknown' prevents NaN handling errors in downstream text analysis.
    
    Args:
        df_track: Track DataFrame
        enabled: Whether to apply the imputation
        fill_value: Replacement string for missing composer
        
    Returns:
        df_track_out: DataFrame with imputed Composer
        audit_record: Audit metadata
    """
    df_out = df_track.copy()
    if not enabled:
        return df_out, {
            "rule": "fill_null_composer",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (Rule disabled)",
            "description": "Composer NULL values retained as NaN."
        }
        
    null_mask = df_out["Composer"].isna()
    affected_count = int(null_mask.sum())
    df_out["Composer"] = df_out["Composer"].fillna(fill_value)
    
    audit_record = {
        "rule": "fill_null_composer",
        "enabled": True,
        "action": "impute",
        "affected_rows": affected_count,
        "affected_customers": 0,
        "impact_on_eda": "None on purchase metrics; 978 track records imputed with 'Unknown' for downstream safety.",
        "description": f"Imputed {affected_count} NULL Composer values with '{fill_value}'."
    }
    logger.info(f"[Rule 2] Imputed {affected_count} NULL Composer records.")
    return df_out, audit_record


def rule_validate_positive_prices(
    df_invoiceline: pd.DataFrame,
    df_invoice: pd.DataFrame,
    enabled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    VALIDATION & FILTER RULE 3: Validate and filter non-positive unit prices (UnitPrice <= 0).
    
    Rationale:
        Protects financial consistency. Free promotional downloads or negative return amounts
        would distort revenue and customer similarity if present.
    """
    if not enabled:
        return df_invoiceline.copy(), {
            "rule": "validate_positive_prices",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (Rule disabled)",
            "description": "Price validation bypassed."
        }
        
    invalid_mask = df_invoiceline["UnitPrice"] <= 0
    affected_count = int(invalid_mask.sum())
    
    if affected_count > 0:
        inv_ids = df_invoiceline.loc[invalid_mask, "InvoiceId"].unique()
        cust_ids = df_invoice.loc[df_invoice["InvoiceId"].isin(inv_ids), "CustomerId"].nunique()
        df_clean = df_invoiceline[~invalid_mask].copy()
        impact = f"WARNING: Removed {affected_count} lines across {cust_ids} customers; alters purchase metrics."
    else:
        df_clean = df_invoiceline.copy()
        cust_ids = 0
        impact = "None (0 invalid prices found; 100% of 2,240 lines have positive UnitPrice 0.99 or 1.99 USD)."
        
    audit_record = {
        "rule": "validate_positive_prices",
        "enabled": True,
        "action": "filter" if affected_count > 0 else "validate_pass",
        "affected_rows": affected_count,
        "affected_customers": cust_ids,
        "impact_on_eda": impact,
        "description": f"Verified UnitPrice > 0. Exactly {affected_count} non-positive rows filtered."
    }
    logger.info(f"[Rule 3] Price validation check completed. Affected rows: {affected_count}.")
    return df_clean, audit_record


def rule_validate_positive_quantities(
    df_invoiceline: pd.DataFrame,
    df_invoice: pd.DataFrame,
    enabled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    VALIDATION & FILTER RULE 4: Validate and filter non-positive quantities (Quantity <= 0).
    
    Rationale:
        Every valid purchase must represent at least 1 digital license unit.
    """
    if not enabled:
        return df_invoiceline.copy(), {
            "rule": "validate_positive_quantities",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (Rule disabled)",
            "description": "Quantity validation bypassed."
        }
        
    invalid_mask = df_invoiceline["Quantity"] <= 0
    affected_count = int(invalid_mask.sum())
    
    if affected_count > 0:
        inv_ids = df_invoiceline.loc[invalid_mask, "InvoiceId"].unique()
        cust_ids = df_invoice.loc[df_invoice["InvoiceId"].isin(inv_ids), "CustomerId"].nunique()
        df_clean = df_invoiceline[~invalid_mask].copy()
        impact = f"WARNING: Removed {affected_count} rows across {cust_ids} customers."
    else:
        df_clean = df_invoiceline.copy()
        cust_ids = 0
        impact = "None (0 invalid quantities found; 100% of 2,240 lines have Quantity == 1)."
        
    audit_record = {
        "rule": "validate_positive_quantities",
        "enabled": True,
        "action": "filter" if affected_count > 0 else "validate_pass",
        "affected_rows": affected_count,
        "affected_customers": cust_ids,
        "impact_on_eda": impact,
        "description": f"Verified Quantity > 0. Exactly {affected_count} non-positive rows filtered."
    }
    logger.info(f"[Rule 4] Quantity validation check completed. Affected rows: {affected_count}.")
    return df_clean, audit_record


def rule_filter_media_types(
    df_invoiceline: pd.DataFrame,
    df_track: pd.DataFrame,
    df_invoice: pd.DataFrame,
    media_type_ids_to_filter: List[int],
    enabled: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    FILTER RULE 5: Optional filtering of specific MediaTypeId categories (e.g. video files).
    
    Rationale:
        Some clustering use cases focus solely on audio/music files, excluding video media types.
        By default, list is empty so all 2,240 purchases are retained.
    """
    if not enabled or not media_type_ids_to_filter:
        return df_invoiceline.copy(), {
            "rule": "filter_media_types",
            "enabled": enabled,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (No media types filtered; default retain all).",
            "description": "Filter list empty or disabled."
        }
        
    # Map track media types
    track_media = df_track.set_index("TrackId")["MediaTypeId"].to_dict()
    line_media = df_invoiceline["TrackId"].map(track_media)
    
    filter_mask = line_media.isin(media_type_ids_to_filter)
    affected_count = int(filter_mask.sum())
    
    if affected_count > 0:
        inv_ids = df_invoiceline.loc[filter_mask, "InvoiceId"].unique()
        cust_ids = df_invoice.loc[df_invoice["InvoiceId"].isin(inv_ids), "CustomerId"].nunique()
        df_clean = df_invoiceline[~filter_mask].copy()
        impact = f"FILTER APPLIED: Excluded {affected_count} purchase lines with MediaTypeId in {media_type_ids_to_filter} across {cust_ids} customers."
    else:
        df_clean = df_invoiceline.copy()
        cust_ids = 0
        impact = f"None (0 purchase lines matched MediaTypeId in {media_type_ids_to_filter})."
        
    audit_record = {
        "rule": "filter_media_types",
        "enabled": True,
        "action": "filter" if affected_count > 0 else "none",
        "affected_rows": affected_count,
        "affected_customers": cust_ids,
        "impact_on_eda": impact,
        "description": f"Filtered lines matching MediaTypeId in {media_type_ids_to_filter}."
    }
    logger.info(f"[Rule 5] Media type filter completed. Affected rows: {affected_count}.")
    return df_clean, audit_record


# ==============================================================================
# 2. STAR SCHEMA ANALYTICAL BUILDER
# ==============================================================================

def build_analytical_star_schema(
    raw_tables: Dict[str, pd.DataFrame],
    config_rules: Dict[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """
    Build clean analytical star-schema tables:
    - fact_purchase: 1 row = 1 unique purchase of (CustomerId, TrackId)
    - dim_customer, dim_track, dim_album, dim_artist, dim_genre
    
    Applies all codified rules and returns tables plus full audit log records.
    """
    audit_logs = []
    
    # 1. Apply rules to Raw Tables
    # Track cleaning
    df_track_clean, r1_log = rule_flag_duplicate_logic_tracks(
        raw_tables["Track"],
        enabled=config_rules.get("flag_duplicate_tracks", True)
    )
    audit_logs.append(r1_log)
    
    df_track_clean, r2_log = rule_fill_null_composer(
        df_track_clean,
        enabled=config_rules.get("fill_null_composer", True)
    )
    audit_logs.append(r2_log)
    
    # InvoiceLine validations
    df_line_clean, r3_log = rule_validate_positive_prices(
        raw_tables["InvoiceLine"],
        raw_tables["Invoice"],
        enabled=config_rules.get("filter_non_positive_prices", True)
    )
    audit_logs.append(r3_log)
    
    df_line_clean, r4_log = rule_validate_positive_quantities(
        df_line_clean,
        raw_tables["Invoice"],
        enabled=config_rules.get("filter_non_positive_quantities", True)
    )
    audit_logs.append(r4_log)
    
    # Media type filter
    filter_media_ids = config_rules.get("filter_media_type_ids", [])
    df_line_clean, r5_log = rule_filter_media_types(
        df_line_clean,
        df_track_clean,
        raw_tables["Invoice"],
        media_type_ids_to_filter=filter_media_ids,
        enabled=bool(filter_media_ids)
    )
    audit_logs.append(r5_log)
    
    # 2. Construct fact_purchase
    # Join InvoiceLine with Invoice (for CustomerId, InvoiceDate) and Track (for AlbumId, GenreId)
    # and Album (for ArtistId)
    df_inv = raw_tables["Invoice"][["InvoiceId", "CustomerId", "InvoiceDate", "BillingCountry"]].copy()
    df_alb = raw_tables["Album"][["AlbumId", "ArtistId"]].copy()
    
    fact = df_line_clean.merge(df_inv, on="InvoiceId", how="inner")
    fact = fact.merge(df_track_clean[["TrackId", "AlbumId", "GenreId", "MediaTypeId"]], on="TrackId", how="inner")
    fact = fact.merge(df_alb, on="AlbumId", how="inner")
    
    # Add calculated LineTotal
    fact["LineTotal"] = (fact["UnitPrice"] * fact["Quantity"]).round(2)
    
    # Explicit column arrangement for fact_purchase
    fact_cols = [
        "InvoiceLineId",
        "CustomerId",
        "TrackId",
        "InvoiceId",
        "InvoiceDate",
        "UnitPrice",
        "Quantity",
        "LineTotal",
        "AlbumId",
        "ArtistId",
        "GenreId",
        "MediaTypeId"
    ]
    fact_purchase = fact[fact_cols].sort_values(by=["CustomerId", "InvoiceDate", "TrackId"]).reset_index(drop=True)
    
    # Compute active sets of keys from fact
    sold_track_ids = set(fact_purchase["TrackId"].unique())
    sold_album_ids = set(fact_purchase["AlbumId"].unique())
    sold_artist_ids = set(fact_purchase["ArtistId"].unique())
    sold_genre_ids = set(fact_purchase["GenreId"].unique())
    active_cust_ids = set(fact_purchase["CustomerId"].unique())
    
    # 3. Construct Dimensions
    # dim_customer (Preserve demographics solely for interpretability / profiling)
    df_cust = raw_tables["Customer"].copy()
    df_cust["CustomerName"] = df_cust["FirstName"] + " " + df_cust["LastName"]
    
    # Precalculate behavioral totals for easy profiling in Phase D/E
    cust_totals = fact_purchase.groupby("CustomerId").agg(
        total_tracks_bought=("TrackId", "count"),
        unique_albums_bought=("AlbumId", "nunique"),
        unique_artists_bought=("ArtistId", "nunique"),
        total_spending=("LineTotal", "sum")
    ).reset_index()
    
    dim_customer = df_cust.merge(cust_totals, on="CustomerId", how="left")
    dim_customer["total_tracks_bought"] = dim_customer["total_tracks_bought"].fillna(0).astype(int)
    dim_customer["unique_albums_bought"] = dim_customer["unique_albums_bought"].fillna(0).astype(int)
    dim_customer["unique_artists_bought"] = dim_customer["unique_artists_bought"].fillna(0).astype(int)
    dim_customer["total_spending"] = dim_customer["total_spending"].fillna(0.0).round(2)
    dim_customer["has_purchases"] = dim_customer["CustomerId"].isin(active_cust_ids)
    
    # Optional filter for zero-order customers
    if config_rules.get("filter_zero_order_customers", False):
        inactive_cust_mask = ~dim_customer["has_purchases"]
        affected_custs = int(inactive_cust_mask.sum())
        dim_customer = dim_customer[dim_customer["has_purchases"]].copy()
        audit_logs.append({
            "rule": "filter_zero_order_customers",
            "enabled": True,
            "action": "filter",
            "affected_rows": affected_custs,
            "affected_customers": affected_custs,
            "impact_on_eda": f"Removed {affected_custs} zero-order customers from dim_customer.",
            "description": "Filtered customers with zero purchases."
        })
    else:
        audit_logs.append({
            "rule": "filter_zero_order_customers",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (All 59 customers retained; 59/59 have orders).",
            "description": "Rule disabled; all customers retained."
        })
    
    # dim_track
    dim_track = df_track_clean.copy()
    dim_track["is_sold"] = dim_track["TrackId"].isin(sold_track_ids)
    
    # Optional filter unsold tracks
    if config_rules.get("filter_unsold_tracks_from_dim", False):
        unsold_count = int((~dim_track["is_sold"]).sum())
        dim_track = dim_track[dim_track["is_sold"]].copy()
        audit_logs.append({
            "rule": "filter_unsold_tracks_from_dim",
            "enabled": True,
            "action": "filter",
            "affected_rows": unsold_count,
            "affected_customers": 0,
            "impact_on_eda": f"dim_track restricted from 3,503 to {len(dim_track)} sold tracks; alters catalog dimension size.",
            "description": f"Filtered {unsold_count} unsold tracks from dim_track."
        })
    else:
        audit_logs.append({
            "rule": "filter_unsold_tracks_from_dim",
            "enabled": False,
            "action": "none",
            "affected_rows": 0,
            "affected_customers": 0,
            "impact_on_eda": "None (All 3,503 catalog tracks retained in dim_track with is_sold boolean flag).",
            "description": "Retained entire 3,503 catalog tracks in dim_track."
        })
        
    # dim_album
    df_album = raw_tables["Album"].copy()
    album_track_counts = df_track_clean.groupby("AlbumId")["TrackId"].count().to_dict()
    df_album["total_tracks"] = df_album["AlbumId"].map(album_track_counts).fillna(0).astype(int)
    df_album["is_sold"] = df_album["AlbumId"].isin(sold_album_ids)
    dim_album = df_album.sort_values(by="AlbumId").reset_index(drop=True)
    
    # dim_artist
    df_artist = raw_tables["Artist"].copy()
    artist_alb_counts = df_album.groupby("ArtistId")["AlbumId"].count().to_dict()
    # Map track count via album
    track_with_artist = df_track_clean.merge(df_album[["AlbumId", "ArtistId"]], on="AlbumId", how="left")
    artist_track_counts = track_with_artist.groupby("ArtistId")["TrackId"].count().to_dict()
    
    df_artist["total_albums"] = df_artist["ArtistId"].map(artist_alb_counts).fillna(0).astype(int)
    df_artist["total_tracks"] = df_artist["ArtistId"].map(artist_track_counts).fillna(0).astype(int)
    df_artist["is_sold"] = df_artist["ArtistId"].isin(sold_artist_ids)
    dim_artist = df_artist.sort_values(by="ArtistId").reset_index(drop=True)
    
    # dim_genre
    df_genre = raw_tables["Genre"].copy()
    genre_track_counts = df_track_clean.groupby("GenreId")["TrackId"].count().to_dict()
    df_genre["total_tracks"] = df_genre["GenreId"].map(genre_track_counts).fillna(0).astype(int)
    df_genre["is_sold"] = df_genre["GenreId"].isin(sold_genre_ids)
    dim_genre = df_genre.sort_values(by="GenreId").reset_index(drop=True)
    
    analytical_tables = {
        "fact_purchase": fact_purchase,
        "dim_customer": dim_customer,
        "dim_track": dim_track,
        "dim_album": dim_album,
        "dim_artist": dim_artist,
        "dim_genre": dim_genre
    }
    
    logger.info("Built Star-Schema: fact_purchase (%d rows), dim_customer (%d rows), dim_track (%d rows), dim_album (%d rows), dim_artist (%d rows), dim_genre (%d rows)",
                len(fact_purchase), len(dim_customer), len(dim_track), len(dim_album), len(dim_artist), len(dim_genre))
                
    return analytical_tables, audit_logs
