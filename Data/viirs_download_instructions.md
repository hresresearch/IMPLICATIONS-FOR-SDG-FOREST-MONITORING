# VIIRS Active Fire Data: Download Instructions

This study uses VIIRS VNP14IMG (375 m) active fire detections for 2015–2023
covering South America. The raw CSV files are not redistributed in this
replication package because of size constraints, but they can be obtained
from publicly accessible archives.

## 1. Source

- NASA / NOAA VIIRS 375 m active fire product (VNP14IMG)
- Access via NASA FIRMS or related data distribution services that provide
  per‑day or per‑orbit CSV exports of active fire detections.

## 2. Spatial and Temporal Filters

When exporting data, apply the following filters as closely as the portal
allows:

- Region: South America (bounding box approximately −85° to −30° longitude,
  −60° to 15° latitude).
- Time period: 2015‑01‑01 to 2023‑12‑31.
- Satellites: Suomi‑NPP VIIRS; if available, also NOAA‑20 VIIRS to match the
  dual‑satellite configuration used in the analysis.
- Output format: CSV with at least the following fields:
  - latitude
  - longitude
  - acquisition date and time
  - satellite identifier
  - confidence or quality flags

## 3. File Organisation Expected by `viirs_loader.py`

The helper script `viirs_loader.py` expects VIIRS CSV files to be organised in
one or more directories at the project root (for example:
`Data/viirs-snpp`, `Data/viirs-snpp-2`, …). All CSV files in these folders are
loaded and concatenated.

For replication, place your downloaded CSV files into one or more directories
under `Data/` and update the directory list at the top of `viirs_loader.py`
if your folder names differ from the original project layout.

## 4. Quality Filtering

During processing, the pipeline:

- Filters to detections classified as land (excluding permanent water) using
  the Hansen datamask.
- Uses only detections with non‑missing coordinates and valid acquisition
  timestamps.
- Aggregates detections to yearly counts and to a second metric based on
  unique pixel‑days (`viirs_fire_unique_pxday`).

Replicators may optionally apply additional filtering (for example, minimum
confidence thresholds) before running the pipeline, provided that any changes
are clearly documented in the Methods section of derivative work.

