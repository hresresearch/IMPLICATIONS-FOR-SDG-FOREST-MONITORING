#!/bin/bash
set -e

# LADEE MapBiomas Amazonas - V3 Pipeline Runner (Dual VIIRS metrics)
# Keeps the original V3 runner intact; writes results to separate outputs/cache.
#
# Español:
# Esta variante de la tubería V3 vuelve a ejecutar los pasos principales
# utilizando dos métricas de VIIRS (conteo y píxel‑día único) y escribe los
# resultados en archivos de salida y cachés separados. Sirve para evaluar
# la sensibilidad de los resultados a la definición de la métrica de fuego
# sin modificar los productos de la ejecución estándar.

echo "========================================================"
echo "Starting V3 Pipeline Execution (Dual VIIRS Metrics)"
echo "========================================================"

# 1. Prepare Protected Areas (WDPA)
echo "[1/3] Running prep_wdpa_v3.py (Geometry Repair & Filtering)..."
python prep_wdpa_v3.py

# 2. Prepare FRA Forest Data (Interpolation/Extrapolation)
echo "[2/3] Running etl_fra_forest_pchip_v2.py (FRA Data Prep)..."
python etl_fra_forest_pchip_v2.py

# 3. Process Hansen Data & Merge All Sources (writes separate outputs/cache)
echo "[3/3] Running process_hansen_and_merge_v3.py (Dual VIIRS outputs)..."
python process_hansen_and_merge_v3.py \
  --cache-dir Data/processed/hansen_tile_cache_v3_dualviirs \
  --output final_paper_dataset_v3_dualviirs.csv \
  --all-thresholds-output final_paper_dataset_v3_dualviirs_all_thresholds.csv \
  --workers 4

echo "========================================================"
echo "Pipeline Finished Successfully (Dual VIIRS)."
echo "Results are in 'final_paper_dataset_v3_dualviirs.csv' and corresponding all-threshold file."
echo "========================================================"
