from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from scripts.generate_dashboard import (
    _make_coverage_chart,
    _make_financing_chart,
    _make_prestaciones_chart,
    load_dispersion,
    load_solvency,
)
from senasa_dashboard.sfs_series import (
    load_sfs_coverage,
    load_sfs_financing,
    load_sfs_prestaciones,
)


DATA_ROOT = Path("data/processed")


def test_load_solvency_contains_system_metrics() -> None:
    df = load_solvency(DATA_ROOT)
    system = df.filter(pl.col("entity") == "TOTAL GENERAL")
    assert system.height > 0
    assert "capital_gap_bn" in df.columns
    assert system.select(pl.col("capital_multiple").drop_nulls()).height > 0


def test_load_dispersion_has_scaled_columns() -> None:
    df = load_dispersion(DATA_ROOT)
    assert df.select(pl.col("capitas_millones").max()).item() > 0
    assert df.select(pl.col("monto_bn").max()).item() > 0


def test_load_sfs_coverage_structure() -> None:
    df = load_sfs_coverage(DATA_ROOT)
    assert df.height > 0
    assert "coverage_pct" in df.columns
    assert df.select(pl.col("affiliates_total").drop_nulls()).height > 0


def test_load_sfs_financing_structure() -> None:
    df = load_sfs_financing(DATA_ROOT)
    assert df.height > 0
    assert df.select(pl.col("ratio_total").drop_nulls()).height > 0


def test_load_sfs_prestaciones_structure() -> None:
    df = load_sfs_prestaciones(DATA_ROOT)
    assert df.height > 0
    assert df.select(pl.col("group_name").drop_nulls()).height > 0


def test_coverage_chart_generates_traces() -> None:
    df = load_sfs_coverage(DATA_ROOT)
    fig = _make_coverage_chart(df)
    assert len(fig.data) >= 4  # stacked bars + population + coverage line


def test_financing_chart_generates_traces() -> None:
    df = load_sfs_financing(DATA_ROOT)
    fig = _make_financing_chart(df)
    assert len(fig.data) == 3  # spend bar, GDP line, ratio line


def test_prestaciones_chart_uses_top_groups() -> None:
    df = load_sfs_prestaciones(DATA_ROOT)
    fig = _make_prestaciones_chart(df)
    assert len(fig.data) > 0
