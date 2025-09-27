"""Data preparation helpers for Senasa dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

ENTITY_SENASA: Literal["ARS SENASA"] = "ARS SENASA"
ENTITY_TOTAL: Literal["TOTAL GENERAL"] = "TOTAL GENERAL"
ENTITY_REST: Literal["RESTO INDUSTRIA"] = "RESTO INDUSTRIA"


def _read_grouped_dataset(data_root: Path | str) -> pl.DataFrame:
    """Load the EF2 grouped metrics parquet file with consistency checks."""

    data_root = Path(data_root)
    path = data_root / "ef2_metrics_grouped.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {path}")
    df = pl.read_parquet(path)
    required = {
        "entity",
        "date",
        "total_income",
        "net_income_total",
        "technical_reserves",
        "invested_reserves",
        "reserve_gap",
        "retained_earnings",
        "accounts_payable_pss",
        "siniestrality_total",
        "siniestrality_contributivo",
        "siniestrality_subsidiado",
    }
    missing = required.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Dataset {path} is missing required columns: {missing_cols}")
    return df


def _read_derived_dataset(data_root: Path | str) -> pl.DataFrame:
    """Load pre-aggregated Senasa vs resto metrics derived from EF2 workbooks."""

    data_root = Path(data_root)
    path = data_root / "senasa_market_metrics_derived.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset not found: {path}")
    return pl.read_parquet(path)


def _prepare_entity_panel(df: pl.DataFrame, entity: str) -> pl.DataFrame:
    """Compute monthly flows for a given entity and normalise to millions of DOP."""

    subset = df.filter(pl.col("entity") == entity).sort("date")
    subset = subset.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.when(pl.col("total_income") <= 0)
        .then(None)
        .otherwise(pl.col("total_income"))
        .alias("_total_income_clean"),
    )

    subset = subset.with_columns(
        pl.col("_total_income_clean").cum_max().over("year").alias("_total_income_adj")
    )

    subset = subset.with_columns(
        pl.col("_total_income_adj").fill_null(strategy="forward").shift(1).over("year").alias(
            "_total_income_prev"
        ),
        pl.col("net_income_total").fill_null(strategy="forward").shift(1).over("year").alias(
            "_net_income_total_prev"
        ),
        pl.col("net_income_contributivo").fill_null(strategy="forward").shift(1).over("year").alias(
            "_net_income_contrib_prev"
        ),
        pl.col("net_income_subsidiado").fill_null(strategy="forward").shift(1).over("year").alias(
            "_net_income_subsid_prev"
        ),
        pl.col("net_income_plan").fill_null(strategy="forward").shift(1).over("year").alias(
            "_net_income_plan_prev"
        ),
    )

    subset = subset.with_columns(
        (pl.col("_total_income_adj") - pl.col("_total_income_prev")).alias("_monthly_income_raw"),
        (pl.col("net_income_total") - pl.col("_net_income_total_prev")).alias(
            "_net_income_monthly_raw"
        ),
        (pl.col("net_income_contributivo") - pl.col("_net_income_contrib_prev")).alias(
            "_net_income_contrib_monthly_raw"
        ),
        (pl.col("net_income_subsidiado") - pl.col("_net_income_subsid_prev")).alias(
            "_net_income_subsid_monthly_raw"
        ),
        (pl.col("net_income_plan") - pl.col("_net_income_plan_prev")).alias(
            "_net_income_plan_monthly_raw"
        ),
    )

    subset = subset.with_columns(
        pl.when(pl.col("_monthly_income_raw").is_null() & pl.col("_total_income_adj").is_not_null())
        .then(pl.col("_total_income_adj"))
        .otherwise(pl.col("_monthly_income_raw"))
        .alias("_monthly_income_raw"),
        pl.when(pl.col("_net_income_monthly_raw").is_null() & pl.col("net_income_total").is_not_null())
        .then(pl.col("net_income_total"))
        .otherwise(pl.col("_net_income_monthly_raw"))
        .alias("_net_income_monthly_raw"),
        pl.when(pl.col("_net_income_contrib_monthly_raw").is_null() & pl.col("net_income_contributivo").is_not_null())
        .then(pl.col("net_income_contributivo"))
        .otherwise(pl.col("_net_income_contrib_monthly_raw"))
        .alias("_net_income_contrib_monthly_raw"),
        pl.when(pl.col("_net_income_subsid_monthly_raw").is_null() & pl.col("net_income_subsidiado").is_not_null())
        .then(pl.col("net_income_subsidiado"))
        .otherwise(pl.col("_net_income_subsid_monthly_raw"))
        .alias("_net_income_subsid_monthly_raw"),
        pl.when(pl.col("_net_income_plan_monthly_raw").is_null() & pl.col("net_income_plan").is_not_null())
        .then(pl.col("net_income_plan"))
        .otherwise(pl.col("_net_income_plan_monthly_raw"))
        .alias("_net_income_plan_monthly_raw"),
    )

    subset = subset.with_columns(
        pl.when((pl.col("_monthly_income_raw") <= 0) | pl.col("_monthly_income_raw").is_null())
        .then(None)
        .otherwise(pl.col("_monthly_income_raw"))
        .alias("monthly_income"),
    )

    subset = subset.with_columns(
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("_net_income_monthly_raw"))
        .alias("net_income_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("_net_income_contrib_monthly_raw"))
        .alias("net_income_contrib_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("_net_income_subsid_monthly_raw"))
        .alias("net_income_subsid_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("_net_income_plan_monthly_raw"))
        .alias("net_income_plan_monthly"),
    )

    subset = subset.with_columns(
        (pl.col("monthly_income") * pl.col("siniestrality_total") / 100).alias("monthly_claims"),
        (pl.col("monthly_income") - pl.col("monthly_income") * pl.col("siniestrality_total") / 100).alias(
            "monthly_margin"
        ),
    )

    subset = subset.with_columns(
        pl.when(pl.col("monthly_income").is_null() | (pl.col("monthly_income") == 0))
        .then(None)
        .otherwise(pl.col("monthly_claims") / pl.col("monthly_income") * 100)
        .alias("monthly_claims_pct"),
        pl.when(pl.col("technical_reserves").is_null() | (pl.col("technical_reserves") == 0))
        .then(None)
        .otherwise(pl.col("reserve_gap") / pl.col("technical_reserves") * 100)
        .alias("reserve_gap_pct"),
        pl.when(pl.col("accounts_payable_pss").is_null() | (pl.col("accounts_payable_pss") == 0))
        .then(None)
        .otherwise(pl.col("technical_reserves") / pl.col("accounts_payable_pss"))
        .alias("reserves_to_payables"),
    )

    subset = subset.with_columns(
        (pl.col("monthly_income") / 1e6).alias("monthly_income_mm"),
        (pl.col("monthly_claims") / 1e6).alias("monthly_claims_mm"),
        (pl.col("monthly_margin") / 1e6).alias("monthly_margin_mm"),
        (pl.col("net_income_total") / 1e6).alias("net_income_total_mm"),
        (pl.col("net_income_monthly") / 1e6).alias("net_income_monthly_mm"),
        (pl.col("net_income_contributivo") / 1e6).alias("net_income_contributivo_mm"),
        (pl.col("net_income_subsidiado") / 1e6).alias("net_income_subsidiado_mm"),
        (pl.col("net_income_plan") / 1e6).alias("net_income_plan_mm"),
        (pl.col("net_income_contrib_monthly") / 1e6).alias("net_income_contrib_monthly_mm"),
        (pl.col("net_income_subsid_monthly") / 1e6).alias("net_income_subsid_monthly_mm"),
        (pl.col("net_income_plan_monthly") / 1e6).alias("net_income_plan_monthly_mm"),
        (pl.col("technical_reserves") / 1e6).alias("technical_reserves_mm"),
        (pl.col("invested_reserves") / 1e6).alias("invested_mm"),
        (pl.col("reserve_gap") / 1e6).alias("reserve_gap_mm"),
        (pl.col("retained_earnings") / 1e6).alias("retained_mm"),
        (pl.col("accounts_payable_pss") / 1e6).alias("accounts_payable_mm"),
    )

    subset = subset.drop([
        "year",
        "_total_income_clean",
        "_total_income_adj",
        "_total_income_prev",
        "_monthly_income_raw",
        "_net_income_monthly_raw",
        "_net_income_contrib_monthly_raw",
        "_net_income_subsid_monthly_raw",
        "_net_income_plan_monthly_raw",
        "_net_income_total_prev",
        "_net_income_contrib_prev",
        "_net_income_subsid_prev",
        "_net_income_plan_prev",
    ])

    return subset


def build_monthly_metrics(data_root: Path | str = Path("data/processed")) -> pl.DataFrame:
    """Return monthly Senasa vs rest-of-industry metrics for dashboard consumption."""

    grouped = _read_grouped_dataset(data_root)
    derived = _read_derived_dataset(data_root)

    senasa = _prepare_entity_panel(grouped, ENTITY_SENASA)
    rest = _prepare_entity_panel(derived, ENTITY_REST)

    senasa_final = senasa.select(
        pl.lit(ENTITY_SENASA).alias("entity"),
        "date",
        "monthly_income",
        "monthly_income_mm",
        "monthly_claims",
        "monthly_claims_mm",
        "monthly_margin",
        "monthly_margin_mm",
        "monthly_claims_pct",
        "net_income_monthly",
        "net_income_monthly_mm",
        "net_income_total",
        "net_income_total_mm",
        "siniestrality_total",
        "siniestrality_contributivo",
        "siniestrality_subsidiado",
        "net_income_contrib_monthly",
        "net_income_contrib_monthly_mm",
        "net_income_subsid_monthly",
        "net_income_subsid_monthly_mm",
        "net_income_plan_monthly",
        "net_income_plan_monthly_mm",
        "technical_reserves",
        "technical_reserves_mm",
        "invested_reserves",
        "invested_mm",
        "reserve_gap",
        "reserve_gap_mm",
        "reserve_gap_pct",
        "accounts_payable_pss",
        "accounts_payable_mm",
        "reserves_to_payables",
        "retained_earnings",
        "retained_mm",
    )

    rest_final = rest.select(
        "entity",
        "date",
        "monthly_income",
        "monthly_income_mm",
        "monthly_claims",
        "monthly_claims_mm",
        "monthly_margin",
        "monthly_margin_mm",
        "monthly_claims_pct",
        "net_income_monthly",
        "net_income_monthly_mm",
        "net_income_total",
        "net_income_total_mm",
        "siniestrality_total",
        "siniestrality_contributivo",
        "siniestrality_subsidiado",
        "net_income_contrib_monthly",
        "net_income_contrib_monthly_mm",
        "net_income_subsid_monthly",
        "net_income_subsid_monthly_mm",
        "net_income_plan_monthly",
        "net_income_plan_monthly_mm",
        "technical_reserves",
        "technical_reserves_mm",
        "invested_reserves",
        "invested_mm",
        "reserve_gap",
        "reserve_gap_mm",
        "reserve_gap_pct",
        "accounts_payable_pss",
        "accounts_payable_mm",
        "reserves_to_payables",
        "retained_earnings",
        "retained_mm",
    )

    combined = pl.concat([senasa_final, rest_final]).sort(["entity", "date"])
    return combined
