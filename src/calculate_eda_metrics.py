"""
Statistical extraction script for EDA metrics.
Computes exact numbers for customer variance, sparsity, coverage, and co-occurrence.
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_connection, load_all_tables, load_master_purchases

conn = get_connection()
tables = load_all_tables()
df_master = load_master_purchases(conn)

print("=== 1. CUSTOMER BEHAVIOR VARIANCE & CV ===")
cust_summary = df_master.groupby("CustomerId").agg(
    total_spending=("LineTotal", "sum"),
    total_tracks=("TrackId", "count"),
    unique_tracks=("TrackId", "nunique"),
    unique_albums=("AlbumId", "nunique"),
    unique_artists=("ArtistId", "nunique"),
    unique_genres=("GenreId", "nunique"),
    total_invoices=("InvoiceId", "nunique")
).reset_index()

for metric in ["total_spending", "total_tracks", "unique_tracks", "unique_albums", "total_invoices"]:
    vals = cust_summary[metric]
    mean_val = vals.mean()
    std_val = vals.std()
    cv_val = (std_val / mean_val) if mean_val != 0 else 0
    print(f"{metric}: Mean={mean_val:.2f}, Std={std_val:.2f}, Min={vals.min()}, Median={vals.median()}, Max={vals.max()}, CV={cv_val:.4f} ({cv_val*100:.2f}%)")

print("\n=== 2. INVOICE FREQUENCY & INTERVALS ===")
inv_df = tables["Invoice"].copy()
inv_df["InvoiceDate"] = pd.to_datetime(inv_df["InvoiceDate"])
inv_df = inv_df.sort_values(by=["CustomerId", "InvoiceDate"])
inv_df["prev_date"] = inv_df.groupby("CustomerId")["InvoiceDate"].shift(1)
inv_df["days_between"] = (inv_df["InvoiceDate"] - inv_df["prev_date"]).dt.days
intervals = inv_df["days_between"].dropna()
print(f"Days between purchases: Mean={intervals.mean():.1f}, Median={intervals.median():.1f}, Min={intervals.min()}, Max={intervals.max()}, Std={intervals.std():.1f}")

print("\n=== 3. COVERAGE & LONG-TAIL ===")
total_catalog_tracks = len(tables["Track"])
sold_tracks = df_master["TrackId"].nunique()
print(f"Tracks coverage: {sold_tracks}/{total_catalog_tracks} ({sold_tracks/total_catalog_tracks*100:.2f}%)")

total_catalog_albums = len(tables["Album"])
sold_albums = df_master["AlbumId"].nunique()
print(f"Albums coverage: {sold_albums}/{total_catalog_albums} ({sold_albums/total_catalog_albums*100:.2f}%)")

tracks_per_buyer = df_master.groupby("TrackId")["CustomerId"].nunique()
print(f"Tracks bought by >= 2 customers: {(tracks_per_buyer >= 2).sum()} / {sold_tracks} ({(tracks_per_buyer >= 2).mean()*100:.2f}%)")
print(f"Tracks bought by exactly 1 customer: {(tracks_per_buyer == 1).sum()} / {sold_tracks} ({(tracks_per_buyer == 1).mean()*100:.2f}%)")
print(f"Max buyers for a single track: {tracks_per_buyer.max()}")

albums_per_buyer = df_master.groupby("AlbumId")["CustomerId"].nunique()
print(f"Albums bought by >= 2 customers: {(albums_per_buyer >= 2).sum()} / {sold_albums} ({(albums_per_buyer >= 2).mean()*100:.2f}%)")
print(f"Albums bought by exactly 1 customer: {(albums_per_buyer == 1).sum()} / {sold_albums} ({(albums_per_buyer == 1).mean()*100:.2f}%)")
print(f"Max buyers for a single album: {albums_per_buyer.max()}")

print("\n=== 4. SPARSITY & DENSITY ===")
num_customers = 59
# Customer x Track (all catalog)
cust_track_all_density = (len(df_master.drop_duplicates(subset=["CustomerId", "TrackId"]))) / (num_customers * total_catalog_tracks)
# Customer x Track (sold catalog only)
cust_track_sold_density = (len(df_master.drop_duplicates(subset=["CustomerId", "TrackId"]))) / (num_customers * sold_tracks)
# Customer x Album (all catalog)
cust_album_all_density = (len(df_master.drop_duplicates(subset=["CustomerId", "AlbumId"]))) / (num_customers * total_catalog_albums)
# Customer x Album (sold catalog only)
cust_album_sold_density = (len(df_master.drop_duplicates(subset=["CustomerId", "AlbumId"]))) / (num_customers * sold_albums)
# Customer x Genre
total_genres = len(tables["Genre"])
cust_genre_density = (len(df_master.drop_duplicates(subset=["CustomerId", "GenreId"]))) / (num_customers * total_genres)
# Customer x Artist
total_artists = len(tables["Artist"])
cust_artist_density = (len(df_master.drop_duplicates(subset=["CustomerId", "ArtistId"]))) / (num_customers * total_artists)

print(f"Customer x Track (All 3,503 tracks): Density = {cust_track_all_density*100:.3f}% (Sparsity = {100 - cust_track_all_density*100:.3f}%)")
print(f"Customer x Track (Sold {sold_tracks} tracks): Density = {cust_track_sold_density*100:.3f}% (Sparsity = {100 - cust_track_sold_density*100:.3f}%)")
print(f"Customer x Album (All 347 albums): Density = {cust_album_all_density*100:.3f}% (Sparsity = {100 - cust_album_all_density*100:.3f}%)")
print(f"Customer x Album (Sold {sold_albums} albums): Density = {cust_album_sold_density*100:.3f}% (Sparsity = {100 - cust_album_sold_density*100:.3f}%)")
print(f"Customer x Genre (25 genres): Density = {cust_genre_density*100:.3f}%")
print(f"Customer x Artist (275 artists): Density = {cust_artist_density*100:.3f}%")

print("\n=== 5. ALBUM COVERAGE RATIO PER CUSTOMER ===")
# Album track count
album_track_counts = tables["Track"].groupby("AlbumId")["TrackId"].count().to_dict()
df_master["AlbumTotalTracks"] = df_master["AlbumId"].map(album_track_counts)
cust_album_cov = df_master.groupby(["CustomerId", "AlbumId"]).agg(
    tracks_bought=("TrackId", "nunique"),
    album_total_tracks=("AlbumTotalTracks", "first")
).reset_index()
cust_album_cov["coverage_ratio"] = cust_album_cov["tracks_bought"] / cust_album_cov["album_total_tracks"]
print(f"Average album coverage ratio per purchase: {cust_album_cov['coverage_ratio'].mean():.4f} ({cust_album_cov['coverage_ratio'].mean()*100:.2f}%)")
print(f"Purchases of full album (coverage == 1.0): {(cust_album_cov['coverage_ratio'] == 1.0).sum()} / {len(cust_album_cov)} ({(cust_album_cov['coverage_ratio'] == 1.0).mean()*100:.2f}%)")
