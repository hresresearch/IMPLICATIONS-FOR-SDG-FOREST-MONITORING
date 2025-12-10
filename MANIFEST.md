# Replication Package MANIFEST

This document outlines the specific files from the main project that need to be copied into this replication package. This manifest ensures that only essential files for reproducibility are included.

## 1. `src/` directory (Scripts)

The following Python and shell scripts should be copied into the `Replication_Package/src/` directory:

* `process_hansen_and_merge_v3.py` (main Hansen/FRA/WDPA/VIIRS processing pipeline)
* `etl_fra_forest_pchip_v2.py` (FRA 2025 forest area interpolation and net change)
* `prep_wdpa_v3.py` (WDPA geometry repair, ISO filtering, STATUS_YR ≤ 2015, reprojection)
* `prep_geo_regions_v2.py` (Amazon/Southern Cone regional boundaries, reprojection)
* `viirs_loader.py` (VIIRS active fire data loading and filtering)
* `build_figures.py` (core figures and tables for the manuscript)
* `build_figures_intro.py` (conceptual schematic and study area map for the Introduction)
* `build_figures_dualviirs.py` (supplementary figures using dual VIIRS metrics)
* `build_stat_summary_dualviirs.py` (regression statistical summaries)
* `run_pipeline_v3.sh` (main end‑to‑end pipeline runner)
* `run_pipeline_v3_viirsdual.sh` (end‑to‑end runner for dual VIIRS metrics)
* `data_ingestor.py` (utility for handling compressed data; optional helper)
* `config.py` (project configuration; optional helper if environment variables are used)

## 2. `Data/` directory (Input Data)

The following input data files (or instructions/links for larger datasets) are
located in the `Replication_Package/Data/` directory:

* `amazon_regions.geojson` (original geographic regions definitions in geographic coordinates; stored at the package root for compatibility with V1 scripts).
* `Data/processed/amazon_regions_102033.geojson` (regional boundaries in ESRI:102033).
* `amazon_wdpa_filtered.geojson` (original filtered WDPA boundaries; stored at the package root).
* `Data/processed/amazon_wdpa_filtered_102033_v3.geojson` (processed WDPA boundaries in ESRI:102033).
* `Data/bulk-download_fra_2025/FRA_Years_variables/1a_forestArea_2025_11_27.csv` (raw FRA 2025 forest area data).
* `Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv` (interpolated FRA forest area and net change, 2015–2023).
* `lossyear_urls.txt`, `treecover2000.txt`, `datamask.txt`, `amazon_datamask.txt`, `amazon_treecover2000.txt` (lists of Hansen GFC v1.12 tiles and/or URLs used in this study; stored at the package root).
* `Data/viirs_download_instructions.md` (instructions and links for downloading VIIRS VNP14IMG active fire CSVs; these are not stored in Git).

Large global source datasets such as Hansen GFC rasters, WDPA shapefiles, and
VIIRS fire archives are not duplicated in this package; instead, tile lists
and download instructions are provided.

## 3. `results/` directory (Output Data & Figures)

The following output files should be copied into the `Replication_Package/results/` directory:

* `final_paper_dataset_v3_dualviirs.csv` (primary aggregated dataset for the analysis, threshold 30 percent)
* `final_paper_dataset_v3_dualviirs_all_thresholds.csv` (aggregated dataset for thresholds 10/30/50 percent)
* `fao_forest_interpolated_2015_2023_pchip_v2.csv` (duplicated here for convenience, as intermediate data)
* `Tables.xlsx` (country summaries, yearly snapshots, protected‑loss statistics, fire–loss correlations, interpolation diagnostics)
* All generated figures used in the manuscript and supplementary material from the `Figures/` directory (e.g., `figure1_total_loss_a.png`, `figure2_ratio_timeseries_a.png`, `fig_viirs_comparison.png`, etc.).

## 4. Root Level (Configuration & Documentation)

*   `requirements.txt` (List of all Python package dependencies, generated from the project's virtual environment)
*   `README.md` (The main README for the replication package)
*   `.gitignore` (To ensure only relevant files are committed if this package is pushed to Git)
