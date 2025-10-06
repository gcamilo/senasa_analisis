"""Helpers for loading SISALRIL SFS historical datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

RAW_SERIES_DIR = Path("data/raw/sisalril/series_historicas")


def _ensure_raw_exists(filename: str) -> Path:
    path = RAW_SERIES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required SISALRIL dataset not found: {path}")
    return path


def load_sfs_coverage(data_root: Path) -> pl.DataFrame:
    processed = data_root / "sfs_coverage_historical.parquet"
    if processed.exists():
        return pl.read_parquet(processed)

    raw = _ensure_raw_exists("h_indicadores_sfs_04.xlsx")
    df = pd.read_excel(raw, skiprows=6)
    df = df.rename(
        columns={
            "Unnamed: 0": "period",
            "Unnamed: 1": "projected_population",
            "Unnamed: 2": "coverage_ratio",
            "Unnamed: 3": "affiliates_total",
            "Unnamed: 4": "affiliates_subsidiado",
            "Unnamed: 5": "affiliates_contributivo",
            "Unnamed: 6": "affiliates_pensionados",
        }
    )
    df["period_numeric"] = pd.to_numeric(df["period"], errors="coerce")
    df = df.dropna(subset=["period_numeric"])
    df["period"] = df["period_numeric"].astype(int).astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["period"], format="%Y%m")
    df["coverage_pct"] = df["coverage_ratio"] * 100
    df = df.drop(columns=["period", "coverage_ratio", "period_numeric"])
    pl_df = pl.from_pandas(df)
    pl_df = pl_df.select(
        "date",
        pl.col("projected_population").alias("population_total"),
        pl.col("affiliates_total"),
        pl.col("affiliates_subsidiado"),
        pl.col("affiliates_contributivo"),
        pl.col("affiliates_pensionados"),
        pl.col("coverage_pct"),
    ).sort("date")
    pl_df.write_parquet(processed)
    return pl_df


def load_sfs_financing(data_root: Path) -> pl.DataFrame:
    processed = data_root / "sfs_financing_gdp.parquet"
    if processed.exists():
        return pl.read_parquet(processed)

    raw = _ensure_raw_exists("h_financiamiento_sfs_03.xlsx")
    df = pd.read_excel(raw, skiprows=8)
    df = df.rename(
        columns={
            "Unnamed: 0": "year",
            "Unnamed: 1": "spend_total",
            "Subsidiado": "spend_pdss_subsidiado",
            "Contributivo": "spend_pdss_contributivo",
            "Unnamed: 4": "spend_other_plans",
            "Unnamed: 5": "gdp_current",
            "Unnamed: 6": "ratio_total",
            "Subsidiado.1": "ratio_subsidiado",
            "Contributivo.1": "ratio_contributivo",
            "Unnamed: 9": "ratio_other",
        }
    )
    df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_numeric"])
    df["year"] = df["year_numeric"].astype(int)
    numeric_cols = [
        "spend_total",
        "spend_pdss_subsidiado",
        "spend_pdss_contributivo",
        "spend_other_plans",
        "gdp_current",
        "ratio_total",
        "ratio_subsidiado",
        "ratio_contributivo",
        "ratio_other",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    pl_df = pl.from_pandas(df.drop(columns=["year_numeric"]))
    pl_df = pl_df.select(
        "date",
        "year",
        "spend_total",
        "spend_pdss_subsidiado",
        "spend_pdss_contributivo",
        "spend_other_plans",
        "gdp_current",
        "ratio_total",
        "ratio_subsidiado",
        "ratio_contributivo",
        "ratio_other",
    ).sort("date")
    pl_df.write_parquet(processed)
    return pl_df


def load_sfs_prestaciones(data_root: Path) -> pl.DataFrame:
    processed = data_root / "sfs_prestaciones_totals.parquet"
    if processed.exists():
        return pl.read_parquet(processed)

    raw = _ensure_raw_exists("h_prestaciones_sfs_01.xlsx")
    df = pd.read_excel(raw, header=5)
    df = df.rename(columns={"Grupo Número/2": "group_id", "Grupo Descripción ": "group_name"})
    df = df.dropna(subset=["group_id", "group_name"])
    df["group_id"] = df["group_id"].astype(str).str.strip()
    df["group_name"] = df["group_name"].astype(str).str.strip()

    # Keep only numeric group identifiers (drop totals/footnotes)
    df = df[df["group_id"].str.fullmatch(r"\d+")]

    # Remove any residual summary rows
    df = df[~df["group_name"].str.contains("Total", case=False, na=False)]
    df = df[~df["group_name"].str.contains("Distribución", case=False, na=False)]
    df = df[~df["group_name"].str.contains("Notas", case=False, na=False)]
    df_melt = df.melt(id_vars=["group_id", "group_name"], var_name="year", value_name="amount")
    df_melt["year_numeric"] = pd.to_numeric(df_melt["year"], errors="coerce")
    df_melt = df_melt.dropna(subset=["year_numeric"])
    df_melt["year"] = df_melt["year_numeric"].astype(int)
    df_melt["amount"] = pd.to_numeric(df_melt["amount"], errors="coerce").fillna(0.0)
    df_melt["group_id"] = df_melt["group_id"].astype(str)
    df_melt["group_name"] = df_melt["group_name"].astype(str).str.strip()
    df_melt["date"] = pd.to_datetime(df_melt["year"].astype(str) + "-01-01")
    pl_df = pl.from_pandas(df_melt.drop(columns=["year_numeric"]))
    pl_df = pl_df.select("date", "year", "group_id", "group_name", "amount")
    pl_df = pl_df.sort(["group_id", "date"])
    pl_df.write_parquet(processed)
    return pl_df
