#!/bin/bash
set -e

# LADEE MapBiomas Amazonas - V3 Pipeline Runner
# Automates the execution of the refactored scientific pipeline.
#
# Español:
# Este script ejecuta de forma secuencial la tubería V3 completa:
# (1) prepara las áreas protegidas (WDPA) con reparación geométrica y filtro
#     temporal; (2) interpola la superficie forestal de FRA con PCHIP;
# (3) procesa las teselas de Hansen y combina Hansen, FRA, WDPA, VIIRS y
#     variables socioeconómicas en el panel país‑año; y
# (4) genera todas las figuras y tablas utilizadas en el artículo.

echo "========================================================"
echo "Starting V3 Pipeline Execution"
echo "========================================================"

# 1. Prepare Protected Areas (WDPA)
# Input: Data/WDPA.../WDPA...shp
# Output: Data/processed/amazon_wdpa_filtered_102033_v3.geojson
echo "[1/4] Running prep_wdpa_v3.py (Geometry Repair & Filtering)..."
python prep_wdpa_v3.py

# 2. Prepare FRA Forest Data (Interpolation/Extrapolation)
# Input: Data/bulk-download_fra_2025/...
# Output: Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv
echo "[2/4] Running etl_fra_forest_pchip_v2.py (FRA Data Prep)..."
python etl_fra_forest_pchip_v2.py

# 3. Process Hansen Data & Merge All Sources
# Input: Hansen Rasters, WDPA, FRA, VIIRS, Socio-Economic Data
# Output: final_paper_dataset_v3.csv
echo "[3/4] Running process_hansen_and_merge_v3.py (Main Pipeline)..."
python process_hansen_and_merge_v3.py \
  --all-thresholds-output final_paper_dataset_all_thresholds_v3.csv

# 4. Generate Figures & Statistics
# Input: final_paper_dataset_v3.csv
# Output: Figures/*.png, Figures/statistical_summary.txt
echo "[4/4] Running build_figures.py (Analysis & Visualization)..."
python build_figures.py

echo "========================================================"
echo "Pipeline Finished Successfully."
echo "Results are in 'final_paper_dataset_v3.csv' and 'Figures/'."
echo "========================================================"
