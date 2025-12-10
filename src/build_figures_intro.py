"""Generate introductory figures: conceptual schematic and study area map.

Figure 1: Conceptual schematic of the analytical framework.
Figure 2: Study area map for the Amazon and Southern Cone countries.

Spanish / Español:
Este módulo genera las figuras introductorias del artículo. La Figura 1 es
un esquema conceptual en formato vertical que resume el flujo de datos y
procesos (insumos, preprocesamiento, panel país‑año, análisis descriptivo y
regresión). La Figura 2 muestra el mapa de la región de estudio con los
doce países analizados y la extensión aproximada de la cobertura forestal
inicial en el año 2000.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
from shapely.geometry import box

from build_figures_base import plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "Figures"
FIG_DIR.mkdir(exist_ok=True)

REGIONS_EQAREA = ROOT / "Data/processed/amazon_regions_102033.geojson"


def make_conceptual_schematic() -> None:
    """Create a vertical box-and-arrow schematic of the analysis workflow."""
    logger.info("Building conceptual schematic (Figure 1)...")
    # Hochformat für ein klareres, vertikales Layout
    fig, ax = plt.subplots(figsize=(5.5, 7.5))
    ax.axis("off")

    # Colors from the scientific palette (no pastels, no purple)
    blue = "#4169e1"   # royal blue
    green = "#228b22"  # forest green
    gray = "#555555"   # neutral gray

    # Box positions: (x, y, width, height) im [0,1]‑Koordinatensystem
    # Vertikale Kette: Inputs -> Geoprocessing -> Panel
    # Unten zwei parallele Boxen: Descriptive, Regression
    boxes = {
        "inputs": (0.20, 0.78, 0.60, 0.12),
        "preproc": (0.20, 0.58, 0.60, 0.12),
        "panel": (0.20, 0.38, 0.60, 0.12),
        "descriptive": (0.08, 0.16, 0.36, 0.14),
        "regression": (0.56, 0.16, 0.36, 0.14),
    }

    def add_box(key: str, label: str, facecolor: str) -> None:
        x, y, w, h = boxes[key]
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04",
            linewidth=1.2,
            edgecolor="black",
            facecolor=facecolor,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            linespacing=1.2,
        )

    add_box(
        "inputs",
        "Input data\nHansen GFC\nVIIRS active fires\nFRA forest area\nWDPA protected areas\nMacro indicators",
        facecolor="#e0e0e0",
    )
    add_box(
        "preproc",
        "Geoprocessing and harmonization\nEqual-area projection (ESRI:102033),\nforest and protected-area masks,\ncanopy thresholds, FRA interpolation",
        facecolor="#d9f2d9",
    )
    add_box(
        "panel",
        "Country-year panel 2015–2023\nTree-cover loss, fire metrics,\nFRA net change, protected loss,\nmacro variables",
        facecolor="#d9e1ff",
    )
    add_box(
        "descriptive",
        "Descriptive analysis\nLoss trajectories, discrepancy ratios,\nprotected-loss shares, fire statistics",
        facecolor="#f0f0f0",
    )
    add_box(
        "regression",
        "Fixed-effects regression\nlog(loss) ~ log(fire)\n+ agriculture share + log(GDP)\n+ country effects",
        facecolor="#f0f0f0",
    )

    # Helper to draw an arrow between box centers
    def arrow(from_key: str, to_key: str, color: str = gray) -> None:
        x1, y1, w1, h1 = boxes[from_key]
        x2, y2, w2, h2 = boxes[to_key]
        ax.annotate(
            "",
            xy=(x2, y2 + h2 / 2),
            xytext=(x1 + w1, y1 + h1 / 2),
            arrowprops=dict(arrowstyle="->", linewidth=1.0, color=color),
        )

    arrow("inputs", "preproc", color=blue)
    arrow("preproc", "panel", color=blue)
    arrow("panel", "descriptive", color=green)
    arrow("panel", "regression", color=green)

    ax.set_title("Analytical framework: data, processing, and analysis", pad=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_intro1_conceptual_framework.png", dpi=300)
    plt.close(fig)


def make_study_area_map() -> None:
    """Create a simple map of the study countries and forested region."""
    logger.info("Building study area map (Figure 2)...")
    if not REGIONS_EQAREA.exists():
        logger.warning("Regions file %s not found; study area map skipped.", REGIONS_EQAREA)
        return

    regions = gpd.read_file(REGIONS_EQAREA)
    if regions.crs is None:
        regions.set_crs("ESRI:102033", inplace=True)

    # Reproject to geographic for easier interpretation
    regions_geo = regions.to_crs("EPSG:4326")

    # Construct a loose bounding box for plot extent
    xmin, ymin, xmax, ymax = regions_geo.total_bounds
    padding_lon = 5.0
    padding_lat = 5.0
    extent = box(xmin - padding_lon, ymin - padding_lat, xmax + padding_lon, ymax + padding_lat)
    extent_gdf = gpd.GeoDataFrame(geometry=[extent], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    extent_gdf.boundary.plot(ax=ax, color="#aaaaaa", linewidth=0.5)

    regions_geo.plot(ax=ax, color="#228b22", edgecolor="black", linewidth=0.6, alpha=0.8)

    for _, row in regions_geo.iterrows():
        if row.geometry.is_empty:
            continue
        x, y = row.geometry.representative_point().coords[0]
        ax.text(x, y, row.get("iso3", ""), ha="center", va="center", fontsize=8, color="black")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Study area: Amazon and Southern Cone countries", pad=14)
    ax.set_xlim(extent.bounds[0], extent.bounds[2])
    ax.set_ylim(extent.bounds[1], extent.bounds[3])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_intro2_study_area.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_conceptual_schematic()
    make_study_area_map()
