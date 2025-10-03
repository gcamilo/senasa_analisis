"""Generate the Senasa vs. resto del sistema dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from senasa_dashboard import build_monthly_metrics
from senasa_dashboard.data import ENTITY_REST, ENTITY_SENASA

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"
DATA_ROOT = Path("data/processed")
REPORT_DIR = Path("reports")
DOCS_DIR = Path("docs")
OVERVIEW_SLUG: Literal["index"] = "index"
METRICS_PATH = DATA_ROOT / "senasa_dashboard_metrics.parquet"
GITHUB_REPO_URL = "https://github.com/gcamilo/senasa_analisis"
PAGES_URL = "https://gcamilo.github.io/senasa_analisis/"

PAGES: list[dict[str, object]] = [
    {
        "slug": OVERVIEW_SLUG,
        "title": "Visión general",
        "include_summary": True,
        "sections": [
            "income",
            "claims",
            "sector_net",
            "sini_total",
        ],
    },
    {
        "slug": "sistema",
        "title": "Sistema SFS",
        "include_summary": False,
        "sections": [
            "affiliations",
            "reserves",
            "payables_ratio",
            "reserve_gap",
        ],
    },
    {
        "slug": "senasa",
        "title": "Foco Senasa",
        "include_summary": False,
        "sections": [
            "net_income",
            "net_income_cumulative",
            "net_income_per_benef",
            "senasa_regimen",
        ],
    },
]

ENTITY_COLORS = {
    "ARS SENASA": "#1f77b4",
    "RESTO INDUSTRIA": "#ff7f0e",
}

REGIMEN_COLORS = {
    "Contributivo": "#1f77b4",
    "Subsidiado": "#ff7f0e",
}

ENTITY_LINE_STYLE = {
    "ARS SENASA": "solid",
    "RESTO INDUSTRIA": "dash",
}

PER_BENEF_COLORS = {
    "Senasa (Subsidiado)": "#1f77b4",
    "Senasa (Contributivo)": "#2ca02c",
    "Resto (Contributivo)": "#ff7f0e",
}

ENTITY_DISPLAY = {
    ENTITY_SENASA: "Sector público (ARS Senasa)",
    ENTITY_REST: "Sector privado",
}

SECTOR_COMPONENTS = [
    ("net_income_contrib_monthly_mm", "Contributivo"),
    ("net_income_subsid_monthly_mm", "Subsidiado"),
    ("net_income_plan_monthly_mm", "Planes especiales"),
]

SECTOR_COMPONENT_COLORS = {
    (ENTITY_SENASA, "Contributivo"): "#1f77b4",
    (ENTITY_SENASA, "Subsidiado"): "#6baed6",
    (ENTITY_SENASA, "Planes especiales"): "#9ecae1",
    (ENTITY_REST, "Contributivo"): "#ff7f0e",
    (ENTITY_REST, "Subsidiado"): "#ffb366",
    (ENTITY_REST, "Planes especiales"): "#ffa64d",
}

def _latest_metric(metrics: pl.DataFrame, entity: str, column: str) -> tuple[str | None, float | None]:
    df = (
        metrics.filter(pl.col("entity") == entity)
        .select("date", column)
        .drop_nulls(column)
        .sort("date")
    )
    if df.height == 0:
        return None, None
    tail = df.tail(1)
    date = tail.get_column("date")[0]
    value = tail.get_column(column)[0]
    return date, value


def _format_number(value: float, decimals: int) -> str:
    formatted = f"{value:,.{decimals}f}"
    if decimals > 0:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_value(value: float | None, fmt: str) -> str:
    if value is None:
        return "—"
    if fmt == "bn":
        return _format_number(value, 2)
    if fmt == "mm":
        return _format_number(value, 2)
    if fmt == "pct":
        return f"{_format_number(value, 2)}%"
    if fmt == "ratio":
        return f"{_format_number(value, 2)}x"
    return _format_number(value, 2)


def _format_date(date: str | None) -> str:
    if date is None:
        return ""
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m")
    return str(date)


def _format_bn_from_mm(value_mm: float | None) -> str:
    if value_mm is None:
        return "—"
    return f"RD$ {_format_number(value_mm / 1_000, 2)} Bn"


def _format_mm_currency(value_mm: float | None) -> str:
    if value_mm is None:
        return "—"
    return f"RD$ {_format_number(value_mm, 2)} MM"


def _build_kpi_list(items: list[dict[str, str]]) -> str:
    lines = []
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "—")
        date = item.get("date")
        date_html = f"<small>{date}</small>" if date else ""
        lines.append(f"<li><span>{label}</span><strong>{value}</strong>{date_html}</li>")
    return "".join(lines)


def _compute_peak_to_trough(metrics: pl.DataFrame) -> tuple[float | None, str | None, str | None]:
    income_valid = (
        metrics.filter((pl.col("entity") == ENTITY_SENASA) & pl.col("monthly_income").is_not_null())
        .select(pl.col("date").max().alias("max_date"))
    )

    if income_valid.height == 0 or income_valid.get_column("max_date")[0] is None:
        return None, None, None

    max_valid_date = income_valid.get_column("max_date")[0]

    series = (
        metrics.filter(pl.col("entity") == ENTITY_SENASA)
        .filter(pl.col("date") <= max_valid_date)
        .select("date", "retained_earnings")
        .drop_nulls("retained_earnings")
        .sort("date")
    )
    if series.height == 0:
        return None, None, None

    max_so_far = None
    peak_date = trough_date = None
    peak_value = trough_value = None
    drawdown = 0.0

    for row in series.iter_rows(named=True):
        date = row["date"]
        value = row["retained_earnings"]

        if max_so_far is None or value > max_so_far:
            max_so_far = value
            peak_value = value
            peak_date = date

        if max_so_far is not None:
            current_drawdown = max_so_far - value
            if current_drawdown > drawdown:
                drawdown = current_drawdown
                trough_value = value
                trough_date = date

    if drawdown <= 0 or peak_date is None or trough_date is None:
        return None, None, None

    return drawdown, _format_date(peak_date), _format_date(trough_date)


def build_kpi_cards_html(metrics: pl.DataFrame) -> str:
    def _sum_income_2024(entity: str) -> float | None:
        subset = metrics.filter(
            (pl.col("entity") == entity)
            & (pl.col("date").dt.year() == 2024)
            & (pl.col("date").dt.month() <= 11)
            & pl.col("monthly_income_mm").is_not_null()
        )
        if subset.height == 0:
            return None
        return subset.get_column("monthly_income_mm").sum()

    def _avg_sini_2024(entity: str) -> float | None:
        subset = metrics.filter(
            (pl.col("entity") == entity)
            & (pl.col("date").dt.year() == 2024)
            & (pl.col("date").dt.month() <= 11)
            & pl.col("siniestrality_total").is_not_null()
        )
        if subset.height == 0:
            return None
        return subset.get_column("siniestrality_total").mean()

    senasa_income_sum = _sum_income_2024(ENTITY_SENASA)
    rest_income_sum = _sum_income_2024(ENTITY_REST)
    income_period_label = "2024 Ene-Nov"

    senasa_sini_avg = _avg_sini_2024(ENTITY_SENASA)
    rest_sini_avg = _avg_sini_2024(ENTITY_REST)

    senasa_contrib_date, senasa_contrib = _latest_metric(metrics, ENTITY_SENASA, "net_income_contrib_monthly_mm")
    senasa_subsid_date, senasa_subsid = _latest_metric(metrics, ENTITY_SENASA, "net_income_subsid_monthly_mm")
    rest_contrib_date, rest_contrib = _latest_metric(metrics, ENTITY_REST, "net_income_contrib_monthly_mm")
    rest_subsid_date, rest_subsid = _latest_metric(metrics, ENTITY_REST, "net_income_subsid_monthly_mm")

    senasa_gap_date, senasa_gap = _latest_metric(metrics, ENTITY_SENASA, "reserve_gap_pct")
    rest_gap_date, rest_gap = _latest_metric(metrics, ENTITY_REST, "reserve_gap_pct")

    drawdown, peak_date, trough_date = _compute_peak_to_trough(metrics)

    cards: list[str] = []

    # Ingreso mensual
    cards.append(
        "<div class=\"kpi-card\">"
        "<h3>Ingreso mensual</h3>"
        "<p class=\"kpi-caption\">Suma enero-noviembre 2024 (RD$ Billones).</p>"
        "<ul class=\"kpi-list\">"
        + _build_kpi_list(
            [
                {
                    "label": "ARS Senasa",
                    "value": _format_bn_from_mm(senasa_income_sum),
                    "date": income_period_label,
                },
                {
                    "label": "Resto industria",
                    "value": _format_bn_from_mm(rest_income_sum),
                    "date": income_period_label,
                },
            ]
        )
        + "</ul></div>"
    )

    # Siniestralidad total
    cards.append(
        "<div class=\"kpi-card\">"
        "<h3>Siniestralidad total</h3>"
        "<p class=\"kpi-caption\">Promedio enero-noviembre 2024.</p>"
        "<ul class=\"kpi-list\">"
        + _build_kpi_list(
            [
                {
                    "label": "ARS Senasa",
                    "value": f"{_format_number(senasa_sini_avg, 2)}%" if senasa_sini_avg is not None else "—",
                    "date": income_period_label,
                },
                {
                    "label": "Resto industria",
                    "value": f"{_format_number(rest_sini_avg, 2)}%" if rest_sini_avg is not None else "—",
                    "date": income_period_label,
                },
            ]
        )
        + "</ul></div>"
    )

    # Resultado neto por plan
    cards.append(
        "<div class=\"kpi-card\">"
        "<h3>Resultado neto por plan</h3>"
        "<p class=\"kpi-caption\">Valores mensuales en RD$ MM.</p>"
        "<ul class=\"kpi-list\">"
        + _build_kpi_list(
            [
                {
                    "label": "Senasa · Contributivo",
                    "value": _format_mm_currency(senasa_contrib),
                    "date": _format_date(senasa_contrib_date),
                },
                {
                    "label": "Senasa · Subsidiado",
                    "value": _format_mm_currency(senasa_subsid),
                    "date": _format_date(senasa_subsid_date),
                },
                {
                    "label": "Resto · Contributivo",
                    "value": _format_mm_currency(rest_contrib),
                    "date": _format_date(rest_contrib_date),
                },
                {
                    "label": "Resto · Subsidiado",
                    "value": _format_mm_currency(rest_subsid),
                    "date": _format_date(rest_subsid_date),
                },
            ]
        )
        + "</ul></div>"
    )

    # Gap de reservas sobre reservas totales
    cards.append(
        "<div class=\"kpi-card\">"
        "<h3>Gap de reservas / reservas técnicas</h3>"
        "<p class=\"kpi-caption\">Brecha como porcentaje de reservas técnicas.</p>"
        "<ul class=\"kpi-list\">"
        + _build_kpi_list(
            [
                {
                    "label": "ARS Senasa",
                    "value": f"{_format_number(senasa_gap, 2)}%" if senasa_gap is not None else "—",
                    "date": _format_date(senasa_gap_date),
                },
                {
                    "label": "Resto industria",
                    "value": f"{_format_number(rest_gap, 2)}%" if rest_gap is not None else "—",
                    "date": _format_date(rest_gap_date),
                },
            ]
        )
        + "</ul></div>"
    )

    # Peak to trough patrimonio Senasa
    cards.append(
        "<div class=\"kpi-card\">"
        "<h3>Drawdown patrimonio Senasa</h3>"
        "<p class=\"kpi-caption\">Máxima caída histórica de patrimonio (retained earnings).</p>"
        + (
            "<div class=\"kpi-highlight\">"
            f"<strong>{'RD$ ' + _format_number(drawdown / 1e9, 2) + ' Bn' if drawdown else '—'}</strong>"
            f"<small>{'Pico ' + peak_date + ' · Valle ' + trough_date if drawdown and peak_date and trough_date else ''}</small>"
            "</div>"
        )
        + "</div>"
    )

    if not cards:
        return ""

    return "<section><h2>Indicadores clave</h2><div class=\"kpi-grid\">" + "".join(cards) + "</div></section>"


def load_affiliations(data_root: Path) -> pl.DataFrame:
    path = data_root / "sfs_affiliation_totals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing affiliation dataset: {path}")
    df = (
        pl.read_parquet(path)
        .select(
            "date",
            "total_sfs",
            "regimen_subsidiado",
            "regimen_contributivo",
        )
        .sort("date")
        .with_columns(
            pl.col("regimen_subsidiado").cast(pl.Float64),
            pl.col("regimen_contributivo").cast(pl.Float64),
            pl.col("total_sfs").cast(pl.Float64),
        )
        .with_columns(
            pl.col("regimen_subsidiado").fill_null(strategy="forward"),
            pl.col("regimen_contributivo").fill_null(strategy="forward"),
            pl.col("total_sfs").fill_null(strategy="forward"),
        )
    )
    return df


def build_net_income_per_beneficiary(
    metrics: pl.DataFrame, affiliations: pl.DataFrame
) -> pl.DataFrame:
    joined = metrics.join(affiliations, on="date", how="left")

    senasa = (
        joined.filter(pl.col("entity") == "ARS SENASA")
        .with_columns(
            pl.when(pl.col("regimen_subsidiado") > 0)
            .then(pl.col("net_income_subsid_monthly") / pl.col("regimen_subsidiado"))
            .otherwise(None)
            .alias("value"),
            pl.lit("Senasa (Subsidiado)").alias("category"),
        )
        .select("date", "value", "category")
    )

    senasa_contrib = (
        joined.filter(pl.col("entity") == "ARS SENASA")
        .with_columns(
            pl.when(pl.col("regimen_contributivo") > 0)
            .then(pl.col("net_income_contrib_monthly") / pl.col("regimen_contributivo"))
            .otherwise(None)
            .alias("value"),
            pl.lit("Senasa (Contributivo)").alias("category"),
        )
        .select("date", "value", "category")
    )

    resto = (
        joined.filter(pl.col("entity") == "RESTO INDUSTRIA")
        .with_columns(
            pl.when(pl.col("regimen_contributivo") > 0)
            .then(pl.col("net_income_contrib_monthly") / pl.col("regimen_contributivo"))
            .otherwise(None)
            .alias("value"),
            pl.lit("Resto (Contributivo)").alias("category"),
        )
        .select("date", "value", "category")
    )

    return pl.concat([senasa, senasa_contrib, resto]).sort(["date", "category"])


def _format_figure(fig: go.Figure, *, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=60, l=60, r=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    if y_title:
        fig.update_yaxes(title=y_title)
    fig.update_xaxes(title="Fecha", tickformat="%Y-%m")
    fig.update_yaxes(zeroline=True, zerolinecolor="#000000", zerolinewidth=1.5)
    return fig


def _make_grouped_bar(df: pl.DataFrame, value_col: str, *, title: str, y_label: str) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        entity_df = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Bar(
                x=entity_df.get_column("date").to_list(),
                y=entity_df.get_column(value_col).to_list(),
                name=entity,
                marker_color=color,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} " + y_label + "<extra></extra>"
    )
    fig.update_layout(barmode="group", title=title)
    return _format_figure(fig, y_title=y_label)


def _make_sector_net_income_chart(df: pl.DataFrame) -> go.Figure:
    filtered = (
        df.filter(pl.col("entity").is_in([ENTITY_SENASA, ENTITY_REST]))
        .select(["entity", "date"] + [col for col, _ in SECTOR_COMPONENTS])
        .sort("date")
    )

    fig = go.Figure()
    trace_meta: list[dict[str, str]] = []
    original_y: list[list[float]] = []

    for entity in (ENTITY_SENASA, ENTITY_REST):
        entity_df = filtered.filter(pl.col("entity") == entity)
        if entity_df.height == 0:
            continue
        display_name = ENTITY_DISPLAY.get(entity, entity)
        offsetgroup = "public" if entity == ENTITY_SENASA else "private"

        dates = entity_df.get_column("date").to_list()

        for column, label in SECTOR_COMPONENTS:
            if column not in entity_df.columns:
                continue
            series = entity_df.get_column(column)
            if series.null_count() == series.len():
                continue
            y_values = series.fill_null(0).to_list()
            color = SECTOR_COMPONENT_COLORS.get((entity, label), ENTITY_COLORS.get(entity, "#888888"))
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=y_values,
                    name=f"{display_name} · {label}",
                    legendgroup=display_name,
                    offsetgroup=offsetgroup,
                    marker_color=color,
                    hovertemplate=(
                        f"{display_name}<br>%{{x|%Y-%m}}<br>{label}: %{{y:,.2f}} RD$ MM<extra></extra>"
                    ),
                )
            )
            trace_meta.append({"entity": entity, "regime": label})
            original_y.append(y_values)

    if not trace_meta:
        return _format_figure(fig, y_title="RD$ MM")

    n_traces = len(trace_meta)

    entity_masks = {
        "Todos": [True] * n_traces,
        ENTITY_SENASA: [meta["entity"] == ENTITY_SENASA for meta in trace_meta],
        ENTITY_REST: [meta["entity"] == ENTITY_REST for meta in trace_meta],
    }

    regime_arrays: dict[str, list[list[float]]] = {"Todos": original_y}
    zeros = [[0.0] * len(y) for y in original_y]

    for _, regime in SECTOR_COMPONENTS:
        regime_arrays[regime] = [
            y if meta["regime"] == regime else zeros[idx]
            for idx, (meta, y) in enumerate(zip(trace_meta, original_y))
        ]

    fig.update_layout(
        barmode="relative",
        title="Resultado neto mensual por sector",
        bargap=0.25,
        updatemenus=[
            {
                "buttons": [
                {
                    "label": "Todos",
                    "method": "update",
                    "args": [{"visible": entity_masks["Todos"]}, {}],
                },
                {
                    "label": "ARS Senasa",
                    "method": "update",
                    "args": [{"visible": entity_masks[ENTITY_SENASA]}, {}],
                },
                {
                    "label": "Resto industria",
                    "method": "update",
                    "args": [{"visible": entity_masks[ENTITY_REST]}, {}],
                },
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.18,
            },
            {
                "buttons": [
                    {
                        "label": "Todos",
                        "method": "update",
                        "args": [{"y": regime_arrays["Todos"]}, {}],
                    },
                    {
                        "label": "Contributivo",
                        "method": "update",
                        "args": [{"y": regime_arrays["Contributivo"]}, {}],
                    },
                    {
                        "label": "Subsidiado",
                        "method": "update",
                        "args": [{"y": regime_arrays["Subsidiado"]}, {}],
                    },
                    {
                        "label": "Planes especiales",
                        "method": "update",
                        "args": [{"y": regime_arrays["Planes especiales"]}, {}],
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.25,
                "y": 1.18,
            },
        ],
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Entidad",
                x=0.0,
                xref="paper",
                y=1.22,
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                text="Régimen",
                x=0.25,
                xref="paper",
                y=1.22,
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            ),
        ]
    )

    return _format_figure(fig, y_title="RD$ MM")


def _make_net_income_chart(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("ARS Senasa", "Resto industria"),
    )

    for row, (entity, color) in enumerate(ENTITY_COLORS.items(), start=1):
        entity_df = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Bar(
                x=entity_df.get_column("date").to_list(),
                y=entity_df.get_column("net_income_monthly_mm").to_list(),
                marker_color=color,
                name=entity,
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} Millones DOP<extra></extra>"
    )
    fig.update_layout(
        title="Resultado neto mensual (Millones DOP)",
        bargap=0.15,
        showlegend=False,
    )
    fig.update_yaxes(title="Millones DOP", row=1, col=1)
    fig.update_yaxes(title="Millones DOP", row=2, col=1)
    fig.update_xaxes(title="Fecha", tickformat="%Y-%m", row=2, col=1)
    return fig


def _make_net_income_cumulative(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        entity_df = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Scatter(
                x=entity_df.get_column("date").to_list(),
                y=entity_df.get_column("net_income_total_mm").to_list(),
                mode="lines+markers",
                line=dict(color=color, dash=ENTITY_LINE_STYLE.get(entity, "solid")),
                name=entity,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} Millones DOP<extra></extra>"
    )
    fig.update_layout(title="Resultado neto acumulado (Millones DOP)")
    return _format_figure(fig, y_title="Millones DOP")


def _make_siniestralidad_total(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        entity_df = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Scatter(
                x=entity_df.get_column("date").to_list(),
                y=entity_df.get_column("monthly_claims_pct").to_list(),
                mode="lines+markers",
                line=dict(color=color, dash=ENTITY_LINE_STYLE.get(entity, "solid")),
                name=entity,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f}%<extra></extra>"
    )
    fig.update_layout(title="Siniestralidad total (%)")
    return _format_figure(fig, y_title="%")


def _make_siniestralidad_regimen(df: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    senasa = df.filter(pl.col("entity") == ENTITY_SENASA).sort("date")
    if senasa.height == 0:
        return fig
    fig.add_trace(
        go.Scatter(
            x=senasa.get_column("date").to_list(),
            y=senasa.get_column("siniestrality_contributivo").to_list(),
            mode="lines+markers",
            line=dict(color=REGIMEN_COLORS["Contributivo"], dash="solid"),
            name="Contributivo",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=senasa.get_column("date").to_list(),
            y=senasa.get_column("siniestrality_subsidiado").to_list(),
            mode="lines+markers",
            line=dict(color=REGIMEN_COLORS["Subsidiado"], dash="dash"),
            name="Subsidiado",
        )
    )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f}%<extra></extra>"
    )
    fig.update_layout(title="Siniestralidad por régimen (%)")
    return _format_figure(fig, y_title="%")


def _make_reserve_levels(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        subset = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("technical_reserves_mm").to_list(),
                mode="lines",
                line=dict(color=color, dash=ENTITY_LINE_STYLE.get(entity, "solid")),
                name=f"Reservas técnicas - {entity}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("invested_mm").to_list(),
                mode="lines",
                line=dict(
                    color=color,
                    dash="dot" if entity == ENTITY_SENASA else "dashdot",
                ),
                name=f"Reservas invertidas - {entity}",
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} Millones DOP<extra></extra>"
    )
    fig.update_layout(title="Reservas técnicas vs invertidas (Millones DOP)")
    return _format_figure(fig, y_title="Millones DOP")


def _make_payables_ratio(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        subset = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("reserves_to_payables").to_list(),
                mode="lines+markers",
                line=dict(color=color, dash=ENTITY_LINE_STYLE.get(entity, "solid")),
                name=entity,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f}x<extra></extra>"
    )
    fig.update_layout(title="Cobertura de reservas sobre cuentas por pagar")
    return _format_figure(fig, y_title="Ratio")


def _make_reserve_gap(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date")
    fig = go.Figure()
    for entity, color in ENTITY_COLORS.items():
        subset = ordered.filter(pl.col("entity") == entity)
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("reserve_gap_pct").to_list(),
                mode="lines+markers",
                line=dict(color=color, dash=ENTITY_LINE_STYLE.get(entity, "solid")),
                name=entity,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f}%<extra></extra>"
    )
    fig.update_layout(title="Gap de reservas (% de reservas técnicas)")
    return _format_figure(fig, y_title="%")


def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, default_height="520px")


def _make_enrollment_chart(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort("date").with_columns(
        (pl.col("regimen_subsidiado") / 1e6).alias("subs_mill"),
        (pl.col("regimen_contributivo") / 1e6).alias("contrib_mill"),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ordered.get_column("date").to_list(),
            y=ordered.get_column("subs_mill").to_list(),
            mode="lines",
            line=dict(color="#1f77b4"),
            name="Subsidiado",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ordered.get_column("date").to_list(),
            y=ordered.get_column("contrib_mill").to_list(),
            mode="lines",
            line=dict(color="#ff7f0e"),
            name="Contributivo",
        )
    )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} Millones de afiliados<extra></extra>"
    )
    fig.update_layout(title="Afiliados al SFS (millones)")
    return _format_figure(fig, y_title="Millones de afiliados")


def _make_net_income_per_benef_chart(df: pl.DataFrame) -> go.Figure:
    ordered = df.sort(["category", "date"])
    fig = go.Figure()
    for category, color in PER_BENEF_COLORS.items():
        subset = ordered.filter(pl.col("category") == category)
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("value").to_list(),
                mode="lines",
                line=dict(
                    color=color,
                    dash="dash" if "Resto" in category else "solid",
                ),
                name=category,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>RD$ %{y:,.2f}<extra></extra>"
    )
    fig.update_layout(title="Resultado neto mensual por afiliado")
    return _format_figure(fig, y_title="RD$ por afiliado")



def _render_html(
    sections: Iterable[tuple[str, str | None, go.Figure]],
    *,
    summary_html: str | None = None,
    nav_links: list[tuple[str, str, bool]] | None = None,
    page_title: str = "Visión general",
) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html lang=\"es\">",
        "<head>",
        "<meta charset=\"utf-8\" />",
        f"<script src=\"{PLOTLY_CDN}\"></script>",
        "<style>",
        "body { font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 32px; }",
        "h1 { font-size: 2.4rem; margin-bottom: 0.25rem; }",
        "h2 { font-size: 1.8rem; margin: 2.5rem 0 0.75rem; }",
        "p.caption { color: #444; max-width: 960px; margin-bottom: 1rem; }",
        "section { margin-bottom: 3rem; }",
        "table.summary { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; }",
        "table.summary th, table.summary td { border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; font-size: 0.95rem; }",
        "table.summary th { background: #f5f7fa; font-weight: 600; }",
        "table.summary td.metric { width: 40%; }",
        ".kpi-grid { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 1rem; }",
        ".kpi-card { background: #f5f7fa; border-radius: 12px; padding: 18px; box-shadow: inset 0 0 0 1px #e2e8f0; }",
        ".kpi-card h3 { margin: 0 0 8px; font-size: 1.05rem; color: #1a202c; }",
        ".kpi-caption { font-size: 0.8rem; color: #4a5568; margin: 0 0 12px; }",
        ".kpi-list { list-style: none; padding: 0; margin: 0; }",
        ".kpi-list li { display: flex; flex-direction: column; margin-bottom: 10px; }",
        ".kpi-list li span { font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; color: #4a5568; }",
        ".kpi-list li strong { font-size: 1.2rem; color: #1a202c; }",
        ".kpi-list li small { font-size: 0.7rem; color: #718096; margin-top: 2px; }",
        ".kpi-card .kpi-highlight { display: flex; flex-direction: column; gap: 4px; }",
        ".kpi-card .kpi-highlight strong { font-size: 1.4rem; color: #1a202c; }",
        ".kpi-card .kpi-highlight small { font-size: 0.75rem; color: #4a5568; }",
        ".top-nav { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 24px; }",
        ".top-nav a { text-decoration: none; color: #2d3748; padding: 6px 12px; border-radius: 6px; background: #edf2f7; font-weight: 500; }",
        ".top-nav a.active { background: #1a202c; color: #fff; }",
        ".top-nav a:hover { background: #2d3748; color: #fff; }",
        "</style>",
        "<title>ARS Senasa vs resto del sistema</title>",
        "</head>",
        "<body>",
    ]

    if nav_links:
        nav_html = ["<nav class=\"top-nav\">"]
        for href, label, active in nav_links:
            class_attr = " class=\"active\"" if active else ""
            nav_html.append(f"<a href=\"{href}\"{class_attr}>{label}</a>")
        nav_html.append("</nav>")
        parts.append("".join(nav_html))

    header_text = "ARS Senasa vs resto del sistema"
    if page_title and page_title != "Visión general":
        header_text = f"{header_text} · {page_title}"

    parts.extend(
        [
            f"<h1>{header_text}</h1>",
            "<p class=\"caption\">Series mensuales reconstruidas a partir de los EF2 (Situación Financiera) publicados por SISALRIL. Senasa se toma de su propio estado y el resto de la industria agrupa todas las ARS privadas. Las cuentas de resultado se expresan como flujo mensual, mientras que las partidas patrimoniales permanecen como stocks. Meses en cero indican periodos aún no auditados.</p>",
        ]
    )

    if summary_html:
        parts.append(summary_html)

    for title, caption, fig in sections:
        parts.append("<section>")
        parts.append(f"<h2>{title}</h2>")
        if caption:
            parts.append(f"<p class=\"caption\">{caption}</p>")
        parts.append(_fig_to_html(fig))
        parts.append("</section>")

    parts.append(
        "<footer style=\"margin-top:48px;font-size:0.85rem;color:#4a5568;\">"
        f"Fuente: SISALRIL. Código y datos: <a href=\"{GITHUB_REPO_URL}\" target=\"_blank\">GitHub</a>."
        "</footer>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    metrics = build_monthly_metrics(DATA_ROOT)
    metrics.write_parquet(METRICS_PATH)

    affiliations = load_affiliations(DATA_ROOT)
    per_beneficiary = build_net_income_per_beneficiary(metrics, affiliations)
    summary_html = build_kpi_cards_html(metrics)
    section_map = {
        "income": {
            "title": "Ingresos mensuales (Bn DOP)",
            "caption": "Flujo mensual de ingresos por capitación tomado de los EF2. El agregado privado suma todas las ARS y corrige el rezago que mezclaba enero y febrero de 2024.",
            "figure": _make_grouped_bar(metrics, "monthly_income_mm", title="Ingresos mensuales (Bn DOP)", y_label="Bn DOP"),
        },
        "claims": {
            "title": "Siniestros mensuales (Bn DOP)",
            "caption": "Costo mensual estimado como ingresos × siniestralidad declarada. Permite contrastar la presión asistencial frente al flujo de primas en cada bloque.",
            "figure": _make_grouped_bar(metrics, "monthly_claims_mm", title="Siniestros mensuales (Bn DOP)", y_label="Bn DOP"),
        },
        "sector_net": {
            "title": "Resultado neto mensual por sector",
            "caption": "Fuentes: REPÚBLICA DOMINICANA: INDICADORES FINANCIEROS DE LAS ARS PÚBLICAS/1 y Régimen Contributivo. El apilado muestra cómo cada régimen explica las ganancias o pérdidas de Senasa y del agregado privado en cada mes.",
            "figure": _make_sector_net_income_chart(metrics),
        },
        "affiliations": {
            "title": "Afiliados al SFS",
            "caption": "Padrones oficiales del Seguro Familiar de Salud; se presentan los millones de beneficiarios en los regímenes subsidiado y contributivo.",
            "figure": _make_enrollment_chart(affiliations),
        },
        "net_income": {
            "title": "Resultado neto mensual (Millones DOP)",
            "caption": "Resultado del mes según EF2. Útil para reconocer episodios con pérdidas o ganancias extraordinarias antes de acumulaciones.",
            "figure": _make_net_income_chart(metrics),
        },
        "net_income_cumulative": {
            "title": "Resultado neto acumulado (Millones DOP)",
            "caption": "Saldo acumulado de utilidades reportado en EF2; ilustra la trayectoria anual al sumar cada mes transcurrido.",
            "figure": _make_net_income_cumulative(metrics),
        },
        "net_income_per_benef": {
            "title": "Resultado neto por afiliado",
            "caption": "Resultado mensual dividido entre la base de afiliados del régimen correspondiente (SISALRIL). Permite medir rentabilidad relativa por beneficiario.",
            "figure": _make_net_income_per_benef_chart(per_beneficiary),
        },
        "sini_total": {
            "title": "Siniestralidad total (%)",
            "caption": "Relación siniestros/ingresos calculada con los flujos anteriores. El trazo punteado identifica al resto de la industria.",
            "figure": _make_siniestralidad_total(metrics),
        },
        "reserves": {
            "title": "Reservas técnicas vs invertidas (Millones DOP)",
            "caption": "Stocks de reservas técnicas y el monto efectivamente invertido según EF2; dimensiona el respaldo financiero de cada bloque.",
            "figure": _make_reserve_levels(metrics),
        },
        "payables_ratio": {
            "title": "Cobertura de reservas sobre cuentas por pagar",
            "caption": "Indicador de liquidez estructural: un valor superior a 1 implica que las reservas técnicas cubren las cuentas por pagar a prestadores (PSS).",
            "figure": _make_payables_ratio(metrics),
        },
        "reserve_gap": {
            "title": "Gap de reservas (% de reservas técnicas)",
            "caption": "Brecha entre reservas técnicas requeridas e invertidas expresada como % de reservas técnicas.",
            "figure": _make_reserve_gap(metrics),
        },
        "senasa_regimen": {
            "title": "Senasa: siniestralidad por régimen (%)",
            "caption": "Serie mensual publicada por Senasa para los regímenes contributivo y subsidiado. Las ARS privadas no reportan este detalle.",
            "figure": _make_siniestralidad_regimen(metrics),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    nav_structure = [(page["slug"], page["title"]) for page in PAGES]
    overview_html = None

    for page in PAGES:
        slug = str(page["slug"])
        sections_payload = []
        for key in page["sections"]:
            data = section_map[key]
            sections_payload.append((data["title"], data["caption"], data["figure"]))

        nav_links = []
        for nav_slug, nav_title in nav_structure:
            href = "index.html" if nav_slug == OVERVIEW_SLUG else f"{nav_slug}.html"
            nav_links.append((href, nav_title, nav_slug == slug))

        page_html = _render_html(
            sections_payload,
            summary_html=summary_html if page.get("include_summary") else None,
            nav_links=nav_links,
            page_title=str(page["title"]),
        )

        filename = "index.html" if slug == OVERVIEW_SLUG else f"{slug}.html"
        (REPORT_DIR / filename).write_text(page_html, encoding="utf-8")
        (DOCS_DIR / filename).write_text(page_html, encoding="utf-8")

        if slug == OVERVIEW_SLUG:
            overview_html = page_html

    if overview_html is not None:
        (REPORT_DIR / "senasa_dashboard.html").write_text(overview_html, encoding="utf-8")


if __name__ == "__main__":
    main()
