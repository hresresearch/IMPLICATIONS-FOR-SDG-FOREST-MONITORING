"""Build Amazon country polygons reprojected to ESRI:102033 for V2 pipeline.

This script keeps the V1 artifacts untouched and writes a new GeoJSON with:
- ISO3 country codes for the eight Amazon countries
- geometries clipped to a generous Amazon basin bounding box
- CRS set to South America Albers Equal Area (ESRI:102033) for area-safe math

Spanish / Español:
Este script descarga los límites administrativos de Natural Earth, selecciona
los países de interés en la Amazonia y el Cono Sur y recorta sus geometrías a
una caja envolvente que cubre la región. A continuación reproyecta las
geometrías a la proyección equal‑area ESRI:102033 para que los cálculos de
superficie (por ejemplo, hectáreas de pérdida de bosque) sean comparables a
lo largo de la gradiente latitudinal. El resultado se guarda como
``Data/processed/amazon_regions_102033.geojson``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AMAZON_ISO3 = ["BRA", "BOL", "PER", "COL", "ECU", "VEN", "GUY", "SUR", "ARG", "CHL", "PRY", "URY"]
# Expanded bbox to cover full South America (down to Tierra del Fuego)
AMAZON_BBOX = box(-85.0, -60.0, -30.0, 15.0)
TARGET_CRS = "ESRI:102033"
OUTPUT_DEFAULT = Path("Data/processed/amazon_regions_102033.geojson")
NE_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


def load_base_world() -> gpd.GeoDataFrame:
    # Always download fresh Natural Earth data to ensure we have all countries
    # (V1 amazon_regions.geojson might be incomplete).
    logger.info("Downloading Natural Earth 110m admin boundaries from %s", NE_URL)
    world = gpd.read_file(NE_URL)
    return world.to_crs("EPSG:4326")


def build_regions(output_path: Path = OUTPUT_DEFAULT) -> None:
    world = load_base_world()
    iso_col = None
    for candidate in ["iso_a3", "ISO_A3", "ADM0_A3", "iso3", "ISO3", "su_a3"]:
        if candidate in world.columns:
            iso_col = candidate
            break
            
    if iso_col is None:
        logger.error("Available columns: %s", world.columns.tolist())
        raise RuntimeError("Base layer lacks iso_a3/iso3 column.")

    subset = world[world[iso_col].isin(AMAZON_ISO3)].copy()
    if subset.empty:
        raise RuntimeError("No matching Amazon ISO3 codes found in base layer.")

    logger.info("Selected %d countries; clipping to Amazon bbox", len(subset))
    subset["geometry"] = subset.geometry.intersection(AMAZON_BBOX)
    subset = subset[subset.geometry.notnull() & ~subset.is_empty].copy()

    if iso_col != "iso3":
        subset = subset.rename(columns={iso_col: "iso3"})
    subset["iso3"] = subset["iso3"].str.upper()

    # Reproject to equal-area CRS for downstream area computations.
    subset = subset.to_crs(TARGET_CRS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_file(output_path, driver="GeoJSON")
    logger.info("Wrote amazon regions to %s (CRS=%s)", output_path, TARGET_CRS)


if __name__ == "__main__":
    build_regions()
