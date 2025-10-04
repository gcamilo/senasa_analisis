from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from scripts.generate_dashboard import load_dispersion, load_solvency


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
