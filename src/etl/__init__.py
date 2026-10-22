"""
ETL Package for Chinook Customer Clustering Project.
Modules:
- extract: Connects to SQLite in read-only mode and extracts raw tables.
- transform: Implements business cleaning rules, builds Fact and Dimension tables, logs all filters.
- load: Persists analytical tables (Parquet) and writes manifest.json metadata.
- run_etl: Command-line entry point to orchestrate end-to-end pipeline execution.
"""
