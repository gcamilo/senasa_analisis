"""Data preparation helpers for Senasa dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl
import polars.selectors as cs

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


def _rename_with_prefix(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    mapping = {col: f"{prefix}{col}" for col in df.columns if col != "date"}
    return df.rename(mapping)


def _densify_and_interpolate(panel: pl.DataFrame, *, entity: str, base: pl.DataFrame | None = None) -> pl.DataFrame:
    if base is not None:
        valid = base.filter(pl.col("monthly_income").is_not_null())
        if valid.height == 0:
            valid = panel.filter(pl.col("monthly_income").is_not_null())
    else:
        valid = panel.filter(pl.col("monthly_income").is_not_null())
    if valid.height == 0:
        return panel

    min_date = valid.select(pl.col("date").min().alias("min"))["min"][0]
    max_date = valid.select(pl.col("date").max().alias("max"))["max"][0]
    all_dates = pl.date_range(min_date, max_date, interval="1mo", eager=True)

    dense = pl.DataFrame({"date": all_dates.cast(pl.Datetime("ns"))})
    panel = dense.join(panel, on="date", how="left")

    numeric_cols = panel.select(cs.numeric()).columns

    panel = panel.with_columns(
        [pl.col(col).interpolate().alias(col) for col in numeric_cols]
    )

    return panel.with_columns(pl.lit(entity).alias("entity"))


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
        pl.col("_total_income_clean").cummax().over("year").alias("_total_income_adj")
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
        pl.col("_total_income_prev").is_null().alias("_prev_income_missing"),
        pl.col("_net_income_total_prev").is_null().alias("_prev_net_missing"),
        pl.col("_net_income_contrib_prev").is_null().alias("_prev_net_contrib_missing"),
        pl.col("_net_income_subsid_prev").is_null().alias("_prev_net_subsid_missing"),
        pl.col("_net_income_plan_prev").is_null().alias("_prev_net_plan_missing"),
    )

    subset = subset.with_columns(
        pl.when(pl.col("_monthly_income_raw").is_null() & pl.col("_prev_income_missing"))
        .then(pl.col("_total_income_adj"))
        .otherwise(pl.col("_monthly_income_raw"))
        .alias("_monthly_income_raw"),
        pl.when(pl.col("_net_income_monthly_raw").is_null() & pl.col("_prev_net_missing"))
        .then(pl.col("net_income_total"))
        .otherwise(pl.col("_net_income_monthly_raw"))
        .alias("_net_income_monthly_raw"),
        pl.when(pl.col("_net_income_contrib_monthly_raw").is_null() & pl.col("_prev_net_contrib_missing"))
        .then(pl.col("net_income_contributivo"))
        .otherwise(pl.col("_net_income_contrib_monthly_raw"))
        .alias("_net_income_contrib_monthly_raw"),
        pl.when(pl.col("_net_income_subsid_monthly_raw").is_null() & pl.col("_prev_net_subsid_missing"))
        .then(pl.col("net_income_subsidiado"))
        .otherwise(pl.col("_net_income_subsid_monthly_raw"))
        .alias("_net_income_subsid_monthly_raw"),
        pl.when(pl.col("_net_income_plan_monthly_raw").is_null() & pl.col("_prev_net_plan_missing"))
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

    null_guard_cols = [
        "monthly_claims",
        "monthly_claims_mm",
        "monthly_margin",
        "monthly_margin_mm",
        "net_income_monthly",
        "net_income_monthly_mm",
        "net_income_contrib_monthly",
        "net_income_contrib_monthly_mm",
        "net_income_subsid_monthly",
        "net_income_subsid_monthly_mm",
        "net_income_plan_monthly",
        "net_income_plan_monthly_mm",
        "monthly_claims_pct",
        "siniestrality_total",
        "reserves_to_payables",
        "technical_reserves",
        "technical_reserves_mm",
        "invested_reserves",
        "invested_mm",
        "reserve_gap",
        "reserve_gap_mm",
        "retained_earnings",
        "retained_mm",
        "accounts_payable_pss",
        "accounts_payable_mm",
    ]

    subset = subset.with_columns(
        [
            pl.when(pl.col("monthly_income").is_null()).then(None).otherwise(pl.col(col)).alias(col)
            for col in null_guard_cols
        ]
    )

    subset = subset.with_columns(
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("monthly_claims")).alias(
            "monthly_claims"
        ),
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("monthly_claims_mm")).alias(
            "monthly_claims_mm"
        ),
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("monthly_margin")).alias(
            "monthly_margin"
        ),
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("monthly_margin_mm")).alias(
            "monthly_margin_mm"
        ),
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("monthly_claims_pct")).alias(
            "monthly_claims_pct"
        ),
        pl.when(pl.col("_prev_income_missing")).then(None).otherwise(pl.col("siniestrality_total")).alias(
            "siniestrality_total"
        ),
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
        "_prev_income_missing",
        "_prev_net_missing",
        "_prev_net_contrib_missing",
        "_prev_net_subsid_missing",
        "_prev_net_plan_missing",
    ])

    return subset


def _compute_rest_from_totals(senasa: pl.DataFrame, total: pl.DataFrame) -> pl.DataFrame:
    total_prefixed = _rename_with_prefix(total, "total_")
    senasa_prefixed = _rename_with_prefix(senasa, "senasa_")

    joined = total_prefixed.join(senasa_prefixed, on="date", how="inner", validate="1:1")

    rest = joined.with_columns(
        pl.lit(ENTITY_REST).alias("entity"),
        (pl.col("total_monthly_income") - pl.col("senasa_monthly_income")).alias("monthly_income"),
        (pl.col("total_monthly_claims") - pl.col("senasa_monthly_claims")).alias("monthly_claims"),
        (pl.col("total_monthly_margin") - pl.col("senasa_monthly_margin")).alias("monthly_margin"),
        (pl.col("total_net_income_monthly") - pl.col("senasa_net_income_monthly")).alias("net_income_monthly"),
        (pl.col("total_net_income_total") - pl.col("senasa_net_income_total")).alias("net_income_total"),
        (pl.col("total_net_income_contrib_monthly") - pl.col("senasa_net_income_contrib_monthly")).alias(
            "net_income_contrib_monthly"
        ),
        (pl.col("total_net_income_subsid_monthly") - pl.col("senasa_net_income_subsid_monthly")).alias(
            "net_income_subsid_monthly"
        ),
        (pl.col("total_net_income_plan_monthly") - pl.col("senasa_net_income_plan_monthly")).alias(
            "net_income_plan_monthly"
        ),
        (pl.col("total_technical_reserves") - pl.col("senasa_technical_reserves")).alias(
            "technical_reserves"
        ),
        (pl.col("total_invested_reserves") - pl.col("senasa_invested_reserves")).alias("invested_reserves"),
        (pl.col("total_reserve_gap") - pl.col("senasa_reserve_gap")).alias("reserve_gap"),
        (pl.col("total_accounts_payable_pss") - pl.col("senasa_accounts_payable_pss")).alias(
            "accounts_payable_pss"
        ),
        (pl.col("total_retained_earnings") - pl.col("senasa_retained_earnings")).alias("retained_earnings"),
    )

    rest = rest.with_columns(
        pl.when(pl.col("monthly_income") <= 0)
        .then(None)
        .otherwise(pl.col("monthly_income"))
        .alias("monthly_income"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("net_income_monthly"))
        .alias("net_income_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("net_income_total"))
        .alias("net_income_total"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("net_income_contrib_monthly"))
        .alias("net_income_contrib_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("net_income_subsid_monthly"))
        .alias("net_income_subsid_monthly"),
        pl.when(pl.col("monthly_income").is_null())
        .then(None)
        .otherwise(pl.col("net_income_plan_monthly"))
        .alias("net_income_plan_monthly"),
    )

    rest = rest.with_columns(
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
        pl.when(pl.col("monthly_income").is_null() | (pl.col("monthly_income") == 0))
        .then(None)
        .otherwise(pl.col("monthly_claims") / pl.col("monthly_income") * 100)
        .alias("siniestrality_total"),
        pl.lit(None).cast(pl.Float64).alias("siniestrality_contributivo"),
        pl.lit(None).cast(pl.Float64).alias("siniestrality_subsidiado"),
    )

    rest = rest.with_columns(
        (pl.col("monthly_income") / 1e6).alias("monthly_income_mm"),
        (pl.col("monthly_claims") / 1e6).alias("monthly_claims_mm"),
        (pl.col("monthly_margin") / 1e6).alias("monthly_margin_mm"),
        (pl.col("net_income_total") / 1e6).alias("net_income_total_mm"),
        (pl.col("net_income_monthly") / 1e6).alias("net_income_monthly_mm"),
        (pl.col("net_income_contrib_monthly") / 1e6).alias("net_income_contrib_monthly_mm"),
        (pl.col("net_income_subsid_monthly") / 1e6).alias("net_income_subsid_monthly_mm"),
        (pl.col("net_income_plan_monthly") / 1e6).alias("net_income_plan_monthly_mm"),
        (pl.col("technical_reserves") / 1e6).alias("technical_reserves_mm"),
        (pl.col("invested_reserves") / 1e6).alias("invested_mm"),
        (pl.col("reserve_gap") / 1e6).alias("reserve_gap_mm"),
        (pl.col("retained_earnings") / 1e6).alias("retained_mm"),
        (pl.col("accounts_payable_pss") / 1e6).alias("accounts_payable_mm"),
    )

    rest = rest.with_columns(
        pl.when(pl.col("monthly_claims").is_null()).then(None).otherwise(pl.col("reserves_to_payables")).alias(
            "reserves_to_payables"
        )
    )

    rest = rest.select(
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

    return rest


def build_monthly_metrics(data_root: Path | str = Path("data/processed")) -> pl.DataFrame:
    """Return monthly Senasa vs rest-of-industry metrics for dashboard consumption."""

    grouped = _read_grouped_dataset(data_root)
    derived = _read_derived_dataset(data_root)

    senasa_grouped = _prepare_entity_panel(grouped, ENTITY_SENASA)
    senasa_derived = _prepare_entity_panel(derived, ENTITY_SENASA)

    senasa_columns = senasa_grouped.columns
    senasa_schema = senasa_derived.schema
    for column in senasa_columns:
        if column not in senasa_derived.columns:
            senasa_derived = senasa_derived.with_columns(
                pl.lit(None, dtype=senasa_schema.get(column, pl.Null)).alias(column)
            )

    senasa_grouped = senasa_grouped.select(senasa_columns)
    senasa_derived = senasa_derived.select(senasa_columns)

    senasa_candidates = pl.concat(
        [
            senasa_grouped.with_columns(
                pl.when(pl.col("monthly_income").is_null()).then(2).otherwise(0).alias("priority")
            ),
            senasa_derived.with_columns(pl.lit(1).alias("priority")),
        ]
    )

    senasa = (
        senasa_candidates.sort(["date", "priority"])
        .unique(subset=["date"], keep="first")
        .drop("priority")
        .sort("date")
    )

    senasa_for_rest = _densify_and_interpolate(senasa, entity=ENTITY_SENASA)

    total = _prepare_entity_panel(grouped, ENTITY_TOTAL)
    rest_from_totals = _compute_rest_from_totals(senasa_for_rest, total)
    rest_from_derived = _prepare_entity_panel(derived, ENTITY_REST)

    target_columns = rest_from_derived.columns
    rest_schema = rest_from_derived.schema
    for column in target_columns:
        if column not in rest_from_totals.columns:
            rest_from_totals = rest_from_totals.with_columns(
                pl.lit(None, dtype=rest_schema.get(column, pl.Null)).alias(column)
            )

    rest_from_totals = rest_from_totals.select(target_columns)
    rest_from_derived = rest_from_derived.select(target_columns)

    fallback_lookup = rest_from_totals.select(
        pl.col("date"), pl.col("monthly_income").alias("__fallback_income")
    )

    rest_derived_with_quality = rest_from_derived.join(
        fallback_lookup, on="date", how="left"
    ).with_columns(
        pl.when(pl.col("monthly_income").is_null())
        .then(2)
        .when(
            (pl.col("__fallback_income").is_not_null())
            & (pl.col("monthly_income") > pl.col("__fallback_income") * 3)
        )
        .then(2)
        .otherwise(0)
        .alias("priority")
    ).drop("__fallback_income")

    rest_candidates = pl.concat(
        [
            rest_derived_with_quality,
            rest_from_totals.with_columns(
                pl.when(pl.col("monthly_income").is_null()).then(2).otherwise(1).alias("priority")
            ),
        ]
    )

    rest = (
        rest_candidates.sort(["date", "priority"])
        .unique(subset=["date"], keep="first")
        .drop("priority")
        .sort("date")
    )

    rest = _densify_and_interpolate(rest, entity=ENTITY_REST, base=rest_from_derived)

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
