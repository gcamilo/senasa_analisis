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
REPORT_PATH = Path("reports/senasa_dashboard.html")
METRICS_PATH = DATA_ROOT / "senasa_dashboard_metrics.parquet"

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

SUMMARY_METRICS = [
    ("monthly_income_mm", "Ingresos mensuales (Millones DOP)", "mm"),
    ("monthly_margin_mm", "Margen mensual (Millones DOP)", "mm"),
    ("monthly_claims_pct", "Siniestros / Ingresos (%)", "pct"),
    ("net_income_monthly_mm", "Resultado neto mensual (Millones DOP)", "mm"),
    ("net_income_contrib_monthly_mm", "Resultado neto contrib. (Millones DOP)", "mm"),
    ("net_income_subsid_monthly_mm", "Resultado neto subsidiado (Millones DOP)", "mm"),
    ("siniestrality_total", "Siniestralidad total (%)", "pct"),
    ("siniestrality_contributivo", "Siniestralidad contrib. (%)", "pct"),
    ("siniestrality_subsidiado", "Siniestralidad subsidiado (%)", "pct"),
    ("technical_reserves_mm", "Reservas técnicas (Millones DOP)", "mm"),
    ("invested_mm", "Reservas invertidas (Millones DOP)", "mm"),
    ("reserve_gap_mm", "Gap de reservas (Millones DOP)", "mm"),
    ("reserves_to_payables", "Reservas / CxP", "ratio"),
]


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


def _format_table_cell(value: float | None) -> str:
    return "—" if value is None else _format_number(value, 2)


def _format_date(date: str | None) -> str:
    if date is None:
        return ""
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m")
    return str(date)


def build_summary_html(metrics: pl.DataFrame) -> str:
    comparative_rows: list[dict[str, str]] = []
    senasa_rows: list[dict[str, str]] = []

    for column, label, fmt in SUMMARY_METRICS:
        senasa_date, senasa_val = _latest_metric(metrics, ENTITY_SENASA, column)
        rest_date, rest_val = _latest_metric(metrics, ENTITY_REST, column)

        if senasa_val is None and rest_val is None:
            continue

        if senasa_val is None:
            continue

        senasa_display = f"{_format_value(senasa_val, fmt)} ({_format_date(senasa_date)})"

        if rest_val is not None:
            rest_display = f"{_format_value(rest_val, fmt)} ({_format_date(rest_date)})"
            comparative_rows.append(
                {
                    "metric": label,
                    "senasa": senasa_display,
                    "rest": rest_display,
                }
            )
        else:
            senasa_rows.append(
                {
                    "metric": label,
                    "senasa": senasa_display,
                }
            )

    if not comparative_rows and not senasa_rows:
        return ""

    parts: list[str] = ["<section>", "<h2>Resumen de indicadores</h2>"]

    if comparative_rows:
        parts.extend([
            "<p class=\"caption\">Métricas con cobertura para Senasa y el resto del sistema (último dato disponible).</p>",
            "<table class=\"summary\"><thead><tr><th>Métrica</th><th>Senasa</th><th>Resto industria</th></tr></thead><tbody>",
        ])
        for row in comparative_rows:
            parts.append(
                f"<tr><td class=\"metric\">{row['metric']}</td><td>{row['senasa']}</td><td>{row['rest']}</td></tr>"
            )
        parts.append("</tbody></table>")

    if senasa_rows:
        parts.extend([
            "<p class=\"caption\">Indicadores reportados únicamente por Senasa.</p>",
            "<table class=\"summary\"><thead><tr><th>Métrica</th><th>Senasa</th></tr></thead><tbody>",
        ])
        for row in senasa_rows:
            parts.append(
                f"<tr><td class=\"metric\">{row['metric']}</td><td>{row['senasa']}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</section>")
    return "".join(parts)


def build_sector_net_income_html(metrics: pl.DataFrame) -> str:
    subset = (
        metrics.select("date", "entity", "net_income_monthly_mm")
        .filter(pl.col("net_income_monthly_mm").is_not_null())
        .pivot(index="date", columns="entity", values="net_income_monthly_mm")
        .sort("date", descending=True)
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("period"))
    )

    if subset.height == 0:
        return ""

    rename_map = {
        ENTITY_SENASA: "Pública (ARS Senasa)",
        ENTITY_REST: "Privada (resto del sistema)",
    }

    subset = subset.rename(rename_map)

    public_col = rename_map[ENTITY_SENASA]
    private_col = rename_map[ENTITY_REST]

    parts: list[str] = [
        "<section>",
        "<h2>Resultado neto mensual por sector</h2>",
        "<p class=\"caption\">Montos en millones de pesos dominicanos (RD$ MM). Sector público corresponde a ARS Senasa; sector privado agrupa el resto del sistema.</p>",
        "<table class=\"timeseries\"><thead><tr><th>Periodo</th><th>Sector público</th><th>Sector privado</th></tr></thead><tbody>",
    ]

    for row in subset.select("period", public_col, private_col).iter_rows(named=True):
        parts.append(
            "<tr><td>"
            + row["period"]
            + "</td><td>"
            + _format_table_cell(row[public_col])
            + "</td><td>"
            + _format_table_cell(row[private_col])
            + "</td></tr>"
        )

    parts.append("</tbody></table>")
    parts.append("</section>")
    return "".join(parts)


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
    joined = metrics.join(affiliations, on="date", how="left", coalesce=True)

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
                line=dict(color=color),
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
            line=dict(color=REGIMEN_COLORS["Subsidiado"], dash="solid"),
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
                line=dict(color=color),
                name=f"Reservas técnicas - {entity}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=subset.get_column("date").to_list(),
                y=subset.get_column("invested_mm").to_list(),
                mode="lines",
                line=dict(color=color, dash="dot"),
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
                line=dict(color=color),
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
                y=subset.get_column("reserve_gap_mm").to_list(),
                mode="lines+markers",
                line=dict(color=color),
                name=entity,
            )
        )
    fig.update_traces(
        hovertemplate="%{fullData.name}<br>%{x|%Y-%m}<br>%{y:,.2f} Millones DOP<extra></extra>"
    )
    fig.update_layout(title="Gap de reservas (Millones DOP)")
    return _format_figure(fig, y_title="Millones DOP")


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
                line=dict(color=color),
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
    summary_html: str | None = None,
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
        "table.timeseries { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.92rem; }",
        "table.timeseries th, table.timeseries td { border: 1px solid #e4e6eb; padding: 6px 10px; text-align: right; }",
        "table.timeseries th:first-child, table.timeseries td:first-child { text-align: left; }",
        "table.timeseries tbody tr:nth-child(even) { background: #f8f9fb; }",
        "</style>",
        "<title>ARS Senasa vs resto del sistema</title>",
        "</head>",
        "<body>",
        "<h1>ARS Senasa vs resto del sistema</h1>",
        "<p class=\"caption\">Series mensuales derivadas de EF2 (Situación Financiera). Las cuentas de resultado se presentan en flujo mensual; las de balance permanecen como stocks.</p>",
    ]

    if summary_html:
        parts.append(summary_html)

    for title, caption, fig in sections:
        parts.append("<section>")
        parts.append(f"<h2>{title}</h2>")
        if caption:
            parts.append(f"<p class=\"caption\">{caption}</p>")
        parts.append(_fig_to_html(fig))
        parts.append("</section>")

    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    metrics = build_monthly_metrics(DATA_ROOT)
    metrics.write_parquet(METRICS_PATH)

    affiliations = load_affiliations(DATA_ROOT)
    per_beneficiary = build_net_income_per_beneficiary(metrics, affiliations)
    summary_html = build_summary_html(metrics)
    sector_html = build_sector_net_income_html(metrics)
    summary_block = "".join(part for part in (summary_html, sector_html) if part)

    sections = [
        (
            "Ingresos mensuales (Bn DOP)",
            "Ingresos convertidos a flujo mensual en miles de millones de pesos.",
            _make_grouped_bar(metrics, "monthly_income_mm", title="Ingresos mensuales (Bn DOP)", y_label="Bn DOP"),
        ),
        (
            "Siniestros mensuales (Bn DOP)",
            "Corrección de signos para el resto de la industria asegura valores positivos.",
            _make_grouped_bar(metrics, "monthly_claims_mm", title="Siniestros mensuales (Bn DOP)", y_label="Bn DOP"),
        ),
        (
            "Afiliados al SFS",
            "Serie oficial SISALRIL: millones de beneficiarios en los regímenes subsidiado y contributivo.",
            _make_enrollment_chart(affiliations),
        ),
        (
            "Resultado neto mensual (Millones DOP)",
            "Resultado neto expresado como flujo del mes. El gráfico siguiente conserva la trayectoria acumulada para referencia.",
            _make_net_income_chart(metrics),
        ),
        (
            "Resultado neto acumulado (Millones DOP)",
            "Serie acumulada directamente del EF2 para seguir la trayectoria histórica.",
            _make_net_income_cumulative(metrics),
        ),
        (
            "Resultado neto por afiliado",
            "Senasa se compara contra el resto del sistema usando la base de afiliados de cada régimen.",
            _make_net_income_per_benef_chart(per_beneficiary),
        ),
        (
            "Siniestralidad total (%)",
            "Promedio ponderado Senasa vs resto del sistema. Los valores del resto ya no colapsan a cero.",
            _make_siniestralidad_total(metrics),
        ),
        (
            "Siniestralidad por régimen (%)",
            "Serie mensual de Senasa por régimen; el resto del sistema se referencia en el panel de siniestralidad total.",
            _make_siniestralidad_regimen(metrics),
        ),
        (
            "Reservas técnicas vs invertidas (Millones DOP)",
            "Comparación de reservas constituidas y las invertidas para cada grupo.",
            _make_reserve_levels(metrics),
        ),
        (
            "Cobertura de reservas sobre cuentas por pagar",
            "Un ratio por encima de 1 indica reservas suficientes para cubrir obligaciones con prestadores.",
            _make_payables_ratio(metrics),
        ),
        (
            "Gap de reservas (Millones DOP)",
            "Brecha entre reservas técnicas requeridas e invertidas.",
            _make_reserve_gap(metrics),
        ),
    ]

    html = _render_html(sections, summary_block)
    REPORT_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
