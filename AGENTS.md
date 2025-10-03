# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Python source modules (package entry lives here).
- `tests/`: Unit/integration tests mirroring `src/` layout.
- `notebooks/`: Exploratory analysis; keep outputs light or cleared.
- `data/`: Local datasets (untracked by default). Use subfolders like `raw/`, `processed/`.
- `scripts/`: One‑off utilities or CLI helpers.
- `AGENTS.md`: This contributor guide.

Tip: Keep notebooks thin and move reusable logic into `src/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps:
  - If `requirements.txt`: `pip install -r requirements.txt`
  - If package: `pip install -e .`
- Run tests: `pytest -q` (from repo root).
- Lint/format (if configured): `ruff check . && ruff format .`
- Run a module: `python -m package_or_module` (from `src/`).

Use `make help` if a `Makefile` is present for shortcuts.

## Coding Style & Naming Conventions
- Python: 4‑space indentation, type hints required on public functions.
- Naming: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`.
- Imports: standard lib, third‑party, local (each group separated).
- Formatting: `ruff format` (Black‑compatible). Lint with `ruff` and fix warnings before commit.

## Testing Guidelines
- Framework: `pytest` with `tests/` mirroring `src/` paths.
- Names: files `test_*.py`; tests use AAA pattern and clear arrange data.
- Coverage: target ≥85% lines for core modules. Run `pytest --cov=src` if coverage is configured.

## Commit & Pull Request Guidelines
- Commits: small, focused, imperative mood. Example: `feat: add loader for CSV inputs`.
- Include context: why the change, not just what changed.
- PRs: concise description, screenshots for user‑facing changes, link issues (`Closes #123`), checklist for tests, docs, and lint passing.

## Security & Configuration Tips
- Do not commit data with PII or credentials. Keep `.env`, `data/`, and large artifacts in `.gitignore`.
- Use environment variables for secrets; access via `os.getenv` and document required keys in `README`.
- Validate inputs and prefer pure functions in `src/` for reproducibility.


## Recent Analyst Notes
- Pulled SISALRIL Estadísticas (Estados Financieros, Afiliación, Dispersión) into `data/raw/sisalril/…` and tidied monthly tables under `data/processed/` (`sfs_affiliation_totals.parquet`, `sfs_plan_especial_breakdown.parquet`, `sfs_dispersion_capitas.parquet`).
- Parsed EF2 states for ARS Senasa; derived liquidity metrics (`senasa_dashboard_metrics.parquet`, `senasa_liquidity_metrics.parquet`) including stress-scenario cash margins, reserve coverage, and reserve gap ratios.
- Delivered dashboards in `reports/` (`senasa_dashboard.html`, `senasa_liquidity_dashboard.html`) with stacked-bar views of ingresos vs gastos, siniestralidad por régimen, reservas técnicas vs invertidas, and retained earnings trends.
- Assumptions flagged: post-Nov-2024 EF2 rows are zero pending audit; claims are estimated via siniestralidad until EF1 claim detail is integrated.
- Rebuilt EF2 ingestion to capture every monthly workbook (2020–2024), including Senasa and “resto industria” aggregates. Clean outputs: `ef2_metrics_clean.parquet`, `ef2_metrics_grouped.parquet`, and comparative `senasa_market_metrics*.parquet` (with derived ratios).
- Updated `reports/senasa_dashboard.html` with Senasa vs. rest-of-market comparison (no stress scenario), regime-specific siniestralidad, reserve/payable ratios, and dropdown-enabled net-income drill-down. Added captions noting audit gaps (e.g., Oct-2023 missing data).
- Recomputed EF2 ingestion via position-based parser to capture Senasa, total system, and rest-of-industry monthly data (see `ef2_metrics_clean.parquet`, `ef2_metrics_grouped.parquet`).
- Derived monthly (non-cumulative) incomes/claims in `senasa_market_metrics_monthly.parquet`, dropping audit placeholders (e.g., oct-2023). Dashboard now compares Senasa vs Rest with regime-level siniestralidad and reserve ratios, no stress scenario.

## Dashboard Styling & Data Decisions (2025-XX)
- Metrics pipeline merges grouped EF2 dataset with derived monthly parquet; when grouped data has gaps, fallback/interpolated values from the derived series fill them, and single-month gaps interpolate between surrounding observations.
- `RESTO INDUSTRIA` monthly flows come from derived parquet when available; `TOTAL GENERAL - ARS SENASA` fallback is used only where derived data is missing; we densify/interpolate to keep timelines contiguous.
- KPI cards replace the summary table, showing Jan–Nov 2024 aggregated ingreso totals, average siniestralidad, latest net income by plan (Senasa + Rest), gap de reservas pct, and Senasa patrimonio peak-to-trough drawdown.
- Drawdown computation only considers dates up to the last month with non-null Senasa monthly income, excluding zero placeholders beyond Nov-2024.
- Line chart styling uses solid Senasa vs dashed Rest traces for all Senasa vs Rest comparisons; reserve gap now charted as % of reserves.
- Sector net income chart now includes entity/regime dropdowns; net income monthly chart hides redundant legend.
- Senasa-only siniestralidad por régimen moved to end of report, rest-of-industry data not available.
