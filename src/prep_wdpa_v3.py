"""Filter WDPA polygons for Amazon countries and reproject to ESRI:102033 (V3).

V3 Update:
- Strict Time Filter: Only include Protected Areas with STATUS_YR <= 2015.
  This eliminates "Look-Ahead Bias" for the analysis period (2015-2023).

Spanish / Español:
Este script filtra los polígonos de la base de datos mundial de áreas
protegidas (WDPA) para los doce países analizados, repara geometrías
inválidas, aplica un filtro temporal que conserva únicamente áreas con
STATUS_YR ≤ 2015 (evitando sesgos de “mirar hacia el futuro” respecto al
período 2015–2023) y reproyecta el resultado a la proyección equal‑area
ESRI:102033. El archivo de salida se utiliza para calcular la pérdida de
cobertura arbórea dentro de áreas protegidas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_ISOS = ["BRA", "BOL", "PER", "COL", "ECU", "VEN", "GUY", "SUR", "ARG", "CHL", "PRY", "URY"]
TARGET_CRS = "ESRI:102033"
RAW_DEFAULT = Path(
    "Data/WDPA_WDOECM_Nov2025_Public_SA_shp/WDPA_WDOECM_Nov2025_Public_SA_shp-polygons.shp"
)
OUTPUT_DEFAULT = Path("Data/processed/amazon_wdpa_filtered_102033_v3.geojson")



def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Checks and repairs geometries using make_valid.
    Explodes collections to ensure consistent polygon types.
    """
    # 1. Identify invalid geometries
    invalid_count = (~gdf.is_valid).sum()
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid geometries. Starting repair...")
        
        # 2. Apply make_valid (requires Geopandas >= 0.10.0)
        gdf["geometry"] = gdf.geometry.make_valid()
        
        # 3. Explode Collections (make_valid can return GeometryCollections)
        # We are only interested in Polygons/MultiPolygons
        gdf = gdf.explode(index_parts=False)
        
        # Filter for Polygon/MultiPolygon
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        
        logger.info("Geometry repair completed.")
    else:
        logger.info("All geometries are valid.")
        
    return gdf

def filter_wdpa(
    input_path: Path = RAW_DEFAULT,
    output_path: Path = OUTPUT_DEFAULT,
    iso_list: Iterable[str] = TARGET_ISOS,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"WDPA input not found: {input_path}")

    logger.info("Loading WDPA polygons from %s", input_path)
    gdf = gpd.read_file(input_path)
    
    # Initial Repair
    gdf = repair_geometries(gdf)

    iso_col = None
    for candidate in ["PARENT_ISO", "PRNT_ISO3", "ISO3"]:
        if candidate in gdf.columns:
            iso_col = candidate
            break
    if iso_col is None:
        raise RuntimeError("Could not find ISO column in WDPA shapefile.")

    # 1. ISO Filter
    iso_set = {iso.upper() for iso in iso_list}
    mask_iso = gdf[iso_col].apply(lambda x: any(iso in str(x) for iso in iso_set))
    filtered = gdf[mask_iso].copy()
    filtered["iso3"] = filtered[iso_col].astype(str).str[:3].str.upper()
    
    if filtered.empty:
        raise RuntimeError("WDPA filter resulted in zero features; check ISO codes.")

    # 2. Time Filter (V3 Scientific Fix)
    # Ensure STATUS_YR is numeric
    if "STATUS_YR" in filtered.columns:
        filtered["STATUS_YR"] = pd.to_numeric(filtered["STATUS_YR"], errors="coerce").fillna(0).astype(int)
        
        # Filter: Keep only PAs established <= 2015.
        mask_time = (filtered["STATUS_YR"] <= 2015)
        
        before_count = len(filtered)
        filtered = filtered[mask_time].copy()
        after_count = len(filtered)
        logger.info("Applied Time Filter (<= 2015): Dropped %d polygons (New PAs)", before_count - after_count)
    else:
        logger.warning("STATUS_YR column not found. Skipping time filter.")

    logger.info("Filtered to %d protected polygons for target countries", len(filtered))

    cols_keep = [
        "WDPAID",
        "NAME",
        "DESIG",
        "IUCN_CAT",
        "STATUS",
        "STATUS_YR",
        "REP_AREA",
        "GIS_AREA",
        "iso3",
        "geometry",
    ]
    filtered = filtered[[c for c in cols_keep if c in filtered.columns]]

    filtered = filtered.to_crs(TARGET_CRS)
    
    # Final Repair after reprojection (reprojection can introduce artifacts)
    filtered = repair_geometries(filtered)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(output_path, driver="GeoJSON")
    logger.info("Wrote filtered WDPA to %s (CRS=%s)", output_path, TARGET_CRS)


if __name__ == "__main__":
    filter_wdpa()
