# Análisis ARS Senasa vs Resto del Sistema

Este repositorio contiene la canalización de datos y el tablero estático que resumen los estados financieros mensuales publicados por la SISALRIL para ARS Senasa y el agregado de ARS privadas ("resto de la industria").

## ¿Qué hace?
- **Procesamiento EF2/EF3**: fusiona los libros EF2 históricos con los derivados y llena huecos mensuales para ambas series.
- **Indicadores mensuales**: calcula ingresos, siniestralidad, resultado neto (total y por plan), reservas y brechas.
- **Dashboard estático**: `docs/index.html` se genera con `scripts/generate_dashboard.py` e incluye KPI cards, comparativos Senasa vs Resto e interacciones por régimen.

## Cómo regenerarlo
1. Crear entorno y dependencias (Python 3.11+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # o pip install polars plotly si no hay requirements
   ```
2. Ejecutar el script de generación:
   ```bash
   PYTHONPATH=src python scripts/generate_dashboard.py
   ```
   Esto actualiza `reports/senasa_dashboard.html` y `data/processed/senasa_dashboard_metrics.parquet`.
3. Copiar el HTML a `docs/index.html` si se desea publicar mediante GitHub Pages.

## Publicación automática
- El workflow de GitHub Actions (`.github/workflows/deploy.yml`) reconstruye el tablero en cada push a `main` y despliega `docs/index.html` en GitHub Pages.
- Los datos necesarios para la generación se almacenan en `data/processed/` (agregados sin información sensible).

## Estructura relevante
- `src/senasa_dashboard/data.py`: lógica de ingestión, interpolación y combinación de series (Senasa + resto).
- `scripts/generate_dashboard.py`: script que prepara los KPI, gráficos y HTML final.
- `data/processed/`: conjuntos procesados (parquet y CSV) requeridos para regenerar el tablero.
- `docs/index.html`: salida estática lista para hospedarse.

## Consideraciones de datos
- Se usan únicamente series agregadas publicadas por la SISALRIL (sin PII).
- Los meses faltantes se interpolan para preservar continuidad, documentado en `AGENTS.md`.

## Licencia y uso
Los datos provienen de fuentes públicas de la SISALRIL. Verifica las condiciones de uso de estos datasets antes de redistribuirlos. El código puede adaptarse para otros análisis financieros dentro del sistema dominicano de salud.
