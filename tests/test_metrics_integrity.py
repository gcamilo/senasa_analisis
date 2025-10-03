from datetime import datetime
import polars as pl
import pytest

from senasa_dashboard import build_monthly_metrics
from senasa_dashboard.data import ENTITY_REST, ENTITY_SENASA

DATA_ROOT = "data/processed"

@pytest.fixture(scope="session")
def metrics() -> pl.DataFrame:
    return build_monthly_metrics(DATA_ROOT)


def _single_value(df: pl.DataFrame, entity: str, date: datetime, column: str) -> float | None:
    row = df.filter((pl.col("entity") == entity) & (pl.col("date") == date)).select(column)
    if row.height == 0:
        raise AssertionError(f"No data for entity={entity}, date={date}")
    return row.get_column(column)[0]


def test_rest_income_feb_2024(metrics: pl.DataFrame) -> None:
    value = _single_value(metrics, ENTITY_REST, datetime(2024, 2, 1), "monthly_income_mm")
    assert value is not None
    assert value == pytest.approx(6251.408, rel=1e-3)


def test_rest_income_jun_2021(metrics: pl.DataFrame) -> None:
    value = _single_value(metrics, ENTITY_REST, datetime(2021, 6, 1), "monthly_income_mm")
    assert value is not None
    assert value == pytest.approx(1480.756, rel=1e-3)


def test_senasa_trailing_months_are_null(metrics: pl.DataFrame) -> None:
    future = metrics.filter(
        (pl.col("entity") == ENTITY_SENASA) & (pl.col("date") >= datetime(2024, 12, 1))
    )
    assert future.select(pl.col("monthly_income").is_null().all()).item()
    assert future.select(pl.col("monthly_claims").is_null().all()).item()
    assert future.select(pl.col("technical_reserves").is_null().all()).item()


def test_siniestrality_has_no_gaps(metrics: pl.DataFrame) -> None:
    missing = metrics.filter(
        (pl.col("monthly_income").is_not_null()) & pl.col("siniestrality_total").is_null()
    )
    assert missing.is_empty(), "Siniestralidad missing for some months with income data"


def test_rest_income_outlier_guardrail(metrics: pl.DataFrame) -> None:
    rest = (
        metrics.filter(pl.col("entity") == ENTITY_REST)
        .sort("date")
        .with_columns(
            pl.col("monthly_income_mm").rolling_median(window_size=4, min_samples=2).alias(
                "rolling_median"
            )
        )
        .filter(pl.col("monthly_income_mm").is_not_null() & pl.col("rolling_median").is_not_null())
        .with_columns(
            pl.when(pl.col("rolling_median") <= 0)
            .then(1.0)
            .when(pl.col("monthly_income_mm") == 0)
            .then(0.0)
            .otherwise(pl.col("monthly_income_mm") / pl.col("rolling_median"))
            .alias("ratio")
        )
        .with_columns(
            pl.when(pl.col("ratio") <= 0)
            .then(1.0)
            .when(pl.col("ratio") < 1)
            .then(1 / pl.col("ratio"))
            .otherwise(pl.col("ratio"))
            .alias("symmetric_ratio")
        )
    )
    max_ratio = rest.select(pl.col("symmetric_ratio").max()).item()
    assert max_ratio <= 1.75, f"Rest of ARS income spike detected (ratio={max_ratio:.2f})"
