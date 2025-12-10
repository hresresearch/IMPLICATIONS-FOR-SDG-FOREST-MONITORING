"""Generate supplemental figures using dual VIIRS metrics (count vs. unique px/day).

Outputs (Figures/):
    - fig_viirs_comparison.png: count vs. unique pixel/day scatter with 1:1 line.
    - fig_fire_elasticity_heatmap.png: Fire elasticity (beta) across thresholds and metrics.
    - fig_loss_choropleth_2019.png: Loss rate (% of forest area) in 2019 by country.
    - fig_fire_loss_timeseries_panel.png: Time series (Loss vs. Fire) for selected countries.

Spanish / Español:
Este script construye figuras complementarias basadas en las dos métricas
de VIIRS (conteo ponderado y píxel‑día único) para evaluar la sensibilidad
de los resultados. Produce un diagrama de dispersión entre ambas métricas,
un mapa de calor de la elasticidad estimada fuego–pérdida por umbral de
cobertura y métrica de fuego, un mapa coroplético de la tasa de pérdida en
2019 y paneles de series de tiempo que comparan la pérdida anual con la
intensidad de detecciones de fuego para países seleccionados.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT.parent
FIG_DIR = PKG_ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)

DATA_ALL = PKG_ROOT / "results/final_paper_dataset_v3_dualviirs_all_thresholds.csv"
REGIONS = PKG_ROOT / "Data/processed/amazon_regions_102033.geojson"

# Style: Times New Roman, muted scientific palette (no pastels)
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)
COLORS = {
    "count": "#4169e1",
    "unique": "#b03030",
    "loss": "#4169e1",
    "fire": "#b03030",
    "rate_low": "#e2e8f2",
    "rate_high": "#0b3d91",
}
PALETTE_ISO = [
    "#4169e1",  # Royal Blue
    "#0b6fa4",  # Teal-Blue
    "#1b7837",  # Dark Green
    "#7fbf7b",  # Light Green
    "#d4a017",  # Gold
    "#c67c2c",  # Orange-Brown
    "#b03030",  # Dark Red
    "#8c510a",  # Brown
    "#4d4d4d",  # Dark Gray
    "#000000",  # Black
    "#6f4e37",  # Coffee
    "#2f4f4f",  # Dark Slate Gray
]


def load_data() -> pd.DataFrame:
    if not DATA_ALL.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ALL}")
    df = pd.read_csv(DATA_ALL)
    df["iso3"] = df["iso3"].str.upper()
    df = df[df["year"].between(2015, 2023)]
    return df


def fig_viirs_comparison(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    palette = {iso: PALETTE_ISO[i % len(PALETTE_ISO)] for i, iso in enumerate(sorted(df["iso3"].unique()))}
    sns.scatterplot(
        data=df,
        x="viirs_fire_count",
        y="viirs_fire_unique_pxday",
        hue="iso3",
        palette=palette,
        s=36,
        alpha=0.7,
        ax=ax,
    )
    max_val = max(df["viirs_fire_count"].max(), df["viirs_fire_unique_pxday"].max())
    ax.plot([0, max_val], [0, max_val], color="black", linestyle="--", linewidth=1, label="1:1")
    ax.set_xlabel("VIIRS fire detections (count)")
    ax.set_ylabel("VIIRS fire detections (unique pixel/day)")
    ax.set_title("VIIRS Intensity vs. Unique Pixel/Day")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.tight_layout()
    out = FIG_DIR / "fig_viirs_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Saved %s", out)


def compute_elasticity(df: pd.DataFrame, fire_col: str, threshold: int) -> float:
    sub = df[df["threshold"] == threshold].copy()
    sub = sub.dropna(subset=[fire_col, "hansen_loss_ha"])
    if sub.empty:
        return np.nan
    sub["log_loss"] = np.log(sub["hansen_loss_ha"] + 1)
    sub["log_fire"] = np.log(sub[fire_col] + 1)
    sub["log_gdp"] = np.log(sub["gdp_per_capita_ppp_const2017"] + 1)
    # z-score agriculture share
    ag = sub["agriculture_value_added_share_gdp_pct"]
    sub["z_agri"] = (ag - ag.mean()) / ag.std(ddof=0) if ag.std(ddof=0) > 0 else 0
    model = smf.ols("log_loss ~ log_fire + z_agri + log_gdp + C(iso3)", data=sub).fit()
    return model.params.get("log_fire", np.nan)


def fig_fire_elasticity_heatmap(df: pd.DataFrame) -> None:
    thresholds = [10, 30, 50]
    metrics = [("viirs_fire_count", "Count"), ("viirs_fire_unique_pxday", "Unique px/day")]
    data = []
    for t in thresholds:
        row = {"threshold": t}
        for col, label in metrics:
            beta = compute_elasticity(df, col, t)
            row[label] = beta
        data.append(row)
    heat = pd.DataFrame(data).set_index("threshold")

    plt.figure(figsize=(5.5, 3.8))
    ax = sns.heatmap(
        heat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Fire elasticity (β1)"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("Tree cover threshold (%)")
    ax.set_title("Elasticity across thresholds and fire metrics")
    plt.tight_layout()
    out = FIG_DIR / "fig_fire_elasticity_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Saved %s", out)


def fig_loss_choropleth_2019(df: pd.DataFrame) -> None:
    if not REGIONS.exists():
        logger.warning("Regions file missing: %s", REGIONS)
        return
    gdf = gpd.read_file(REGIONS).to_crs("EPSG:4326")
    sub = df[(df["year"] == 2019) & (df["threshold"] == 30)].copy()
    sub["loss_rate_pct"] = 100 * sub["hansen_loss_ha"] / sub["fao_forest_area_ha"].replace(0, np.nan)
    merged = gdf.merge(sub[["iso3", "loss_rate_pct"]], on="iso3", how="left")
    vmin, vmax = 0, merged["loss_rate_pct"].quantile(0.95)
    plt.figure(figsize=(7.5, 7))
    ax = plt.gca()
    merged.plot(
        column="loss_rate_pct",
        cmap="Blues",
        linewidth=0.5,
        edgecolor="black",
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={"label": "Loss rate 2019 (% of FRA forest area)", "shrink": 0.7},
        ax=ax,
    )
    ax.set_axis_off()
    ax.set_title("2019 Forest Loss Rate (Threshold 30%)", pad=10)
    plt.tight_layout()
    out = FIG_DIR / "fig_loss_choropleth_2019.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_fire_loss_timeseries(df: pd.DataFrame) -> None:
    countries = ["BRA", "BOL", "CHL", "PRY"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    axes = axes.flatten()
    for ax, iso in zip(axes, countries):
        sub = df[(df["iso3"] == iso) & (df["threshold"] == 30)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub["hansen_loss_ha"] / 1e3, label="Loss (kha)", color=COLORS["loss"], linewidth=1.8)
        ax_t = ax.twinx()
        ax_t.plot(sub["year"], sub["viirs_fire_unique_pxday"] / 1e3, label="Fire unique (k)", color=COLORS["unique"], linewidth=1.4, linestyle="--")
        ax.set_title(iso)
        ax.set_ylabel("Loss (kha)")
        ax_t.set_ylabel("Fire detections (k)")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_t.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    out = FIG_DIR / "fig_fire_loss_timeseries_panel.png"
    plt.savefig(out, dpi=300)
    plt.close()
    logger.info("Saved %s", out)


def main() -> None:
    df = load_data()
    fig_viirs_comparison(df[df["threshold"] == 30])
    fig_fire_elasticity_heatmap(df)
    fig_loss_choropleth_2019(df)
    fig_fire_loss_timeseries(df)


if __name__ == "__main__":
    main()
