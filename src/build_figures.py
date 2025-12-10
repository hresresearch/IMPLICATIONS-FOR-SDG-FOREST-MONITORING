"""Generate tables and figures for the manuscript with consistent styling.

Spanish / Español:
Este script genera todas las tablas y figuras utilizadas en el artículo,
aplicando un estilo gráfico coherente. A partir del conjunto de datos
final (panel país‑año), calcula resúmenes estadísticos, crea las tablas
en ``Tables.xlsx`` y produce las figuras principales para la pérdida de
cobertura arbórea, la razón de discrepancia Hansen–FRA, la proporción de
pérdida en áreas protegidas, la relación fuego–pérdida y el análisis de
sensibilidad de los umbrales de cobertura de copa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import logging

from build_figures_base import plt  # Ensures shared style

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT.parent
FIG_DIR = PKG_ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)

DATA_PATH = PKG_ROOT / "results/final_paper_dataset_v3_dualviirs.csv"
FRA_RAW = PKG_ROOT / "Data/bulk-download_fra_2025/FRA_Years_variables/1a_forestArea_2025_11_27.csv"
FRA_PCHIP = PKG_ROOT / "Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv"


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # df is already loaded and filtered

    summary = (
        df.groupby("iso3")
        .agg(
            total_hansen_loss_Mha=("hansen_loss_ha", lambda x: x.sum() / 1e6),
            mean_annual_loss_kha=("hansen_loss_ha", lambda x: x.mean() / 1e3),
            mean_discrepancy_ratio=("discrepancy_ratio", "mean"),
            mean_protected_share=("protected_loss_share", "mean"),
            total_viirs_M=("viirs_fire_count", lambda x: x.sum() / 1e6),
        )
        .reset_index()
    )

    snapshot_years = [2016, 2019, 2023]
    snapshot = df[df["year"].isin(snapshot_years)].copy()
    snapshot["hansen_loss_kha"] = snapshot["hansen_loss_ha"] / 1e3
    snapshot["fao_net_change_kha"] = snapshot["fao_net_change_ha"] / 1e3
    snapshot = snapshot[["iso3", "year", "hansen_loss_kha", "fao_net_change_kha", "discrepancy_ratio"]]
    snapshot = snapshot.sort_values(["iso3", "year"])

    prot = df.dropna(subset=["protected_loss_share"]).copy()
    prot_stats = (
        prot.groupby("iso3")
        .agg(
            protected_loss_share_mean=("protected_loss_share", "mean"),
            protected_loss_share_max=("protected_loss_share", "max"),
        )
        .reset_index()
    )
    idx = prot.groupby("iso3")["protected_loss_share"].idxmax()
    prot_stats["year_of_max"] = prot.loc[idx, "year"].values

    corr_records = []
    for iso, sub in df.groupby("iso3"):
        if sub["viirs_fire_count"].notna().sum() >= 3 and sub["hansen_loss_ha"].notna().sum() >= 3:
            corr = sub["hansen_loss_ha"].corr(sub["viirs_fire_count"])
            lr = linregress(sub["viirs_fire_count"], sub["hansen_loss_ha"])
            corr_records.append(
                {
                    "iso3": iso,
                    "pearson_corr": corr,
                    "slope_ha_per_fire": lr.slope,
                    "intercept": lr.intercept,
                    "p_value": lr.pvalue,
                }
            )
    fire_corr = pd.DataFrame(corr_records)

    fra_raw_df = pd.read_csv(FRA_RAW, encoding="latin-1").rename(columns={"ï»¿regions": "region"})
    fra_raw_df["iso3"] = fra_raw_df["iso3"].str.upper()
    fra_pchip_df = pd.read_csv(FRA_PCHIP)

    years_interp = np.arange(2015, 2024)
    records = []
    for iso in fra_pchip_df["iso3"].unique():
        series = fra_pchip_df[fra_pchip_df["iso3"] == iso].set_index("year")["forest_area_kha"]
        raw_row = fra_raw_df[fra_raw_df["iso3"] == iso]
        if raw_row.empty:
            continue
        year_cols = [c for c in raw_row.columns if c.isdigit()]
        obs = {int(y): raw_row.iloc[0][y] for y in year_cols if pd.notna(raw_row.iloc[0][y])}
        obs = {y: v for y, v in obs.items() if 2015 <= y <= 2023}
        if len(obs) < 2:
            continue
        obs_years = sorted(obs.keys())
        obs_vals = [obs[y] for y in obs_years]
        lin = pd.Series(index=years_interp, dtype=float)
        lin.loc[obs_years] = obs_vals
        lin = lin.interpolate(method="linear")
        diffs = (series - lin).reindex(years_interp)
        records.append(
            {
                "iso3": iso,
                "max_abs_diff_kha": diffs.abs().max(),
                "mean_abs_diff_kha": diffs.abs().mean(),
            }
        )
    fra_interp_cmp = pd.DataFrame(records)

    return {
        "summary": summary,
        "snapshot": snapshot,
        "protected": prot_stats,
        "fire_corr": fire_corr,
        "interp_check": fra_interp_cmp,
    }


def write_tables(tables: dict[str, pd.DataFrame]) -> None:
    excel_path = PKG_ROOT / "results/Tables.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Table 1: Country summary (clean, US English headers)
        summary = tables["summary"].rename(
            columns={
                "iso3": "Country (ISO3)",
                "total_hansen_loss_Mha": "Total Hansen loss 2015–2023 (Mha)",
                "mean_annual_loss_kha": "Mean annual Hansen loss (kha)",
                "mean_discrepancy_ratio": "Mean discrepancy ratio (Hansen/FRA)",
                "mean_protected_share": "Mean protected-loss share",
                "total_viirs_M": "Total VIIRS fire detections (millions)",
            }
        )
        summary.to_excel(writer, sheet_name="Table1_country_summary", index=False)

        # Table 2: Yearly snapshot for selected years
        snapshot = tables["snapshot"].rename(
            columns={
                "iso3": "Country (ISO3)",
                "year": "Year",
                "hansen_loss_kha": "Hansen loss (kha)",
                "fao_net_change_kha": "FRA net forest area change (kha)",
                "discrepancy_ratio": "Discrepancy ratio (Hansen/FRA)",
            }
        )
        snapshot.to_excel(writer, sheet_name="Table2_yearly_snapshot", index=False)

        # Table 3: Protected-area loss statistics
        protected = tables["protected"].rename(
            columns={
                "iso3": "Country (ISO3)",
                "protected_loss_share_mean": "Mean protected-loss share",
                "protected_loss_share_max": "Maximum protected-loss share",
                "year_of_max": "Year of maximum protected-loss share",
            }
        )
        protected.to_excel(writer, sheet_name="Table3_protected_share", index=False)

        # Table 4: Fire–loss correlation diagnostics
        fire_corr = tables["fire_corr"].rename(
            columns={
                "iso3": "Country (ISO3)",
                "pearson_corr": "Pearson correlation (Hansen vs VIIRS)",
                "slope_ha_per_fire": "Slope (ha per detection)",
                "intercept": "Intercept (ha)",
                "p_value": "p-value",
            }
        )
        fire_corr.to_excel(writer, sheet_name="Table4_fire_loss_corr", index=False)

        # Table 5: FRA interpolation diagnostics
        interp = tables["interp_check"].rename(
            columns={
                "iso3": "Country (ISO3)",
                "max_abs_diff_kha": "Max absolute difference (kha)",
                "mean_abs_diff_kha": "Mean absolute difference (kha)",
            }
        )
        interp.to_excel(writer, sheet_name="Table5_interp_check", index=False)

        # Appendix Table A1: Discrepancy ratios for selected years (wide format)
        snapshot_wide = tables["snapshot"].pivot(
            index="iso3", columns="year", values="discrepancy_ratio"
        ).reset_index()
        snapshot_wide = snapshot_wide.rename(columns={"iso3": "Country (ISO3)"})
        snapshot_wide.to_excel(writer, sheet_name="TableA1_discrepancy_ratios", index=False)

        # Appendix Table A2: Fire–loss correlation diagnostics sorted by Pearson r
        fire_corr_sorted = fire_corr.sort_values("Pearson correlation (Hansen vs VIIRS)", ascending=False).reset_index(drop=True)
        fire_corr_sorted.to_excel(writer, sheet_name="TableA2_fire_loss_corr_sorted", index=False)


def plot_figures(df: pd.DataFrame) -> None:
    # Distinct, non-pastel palette; no purple.
    colors = [
        "#4169e1",  # Royal Blue (standard)
        "#6baed6",  # Light Blue
        "#1b7837",  # Dark Green
        "#7fbf7b",  # Light Green
        "#d4a017",  # Gold
        "#c67c2c",  # Orange-Brown
        "#b03030",  # Dark Red
        "#8c510a",  # Brown
        "#4d4d4d",  # Dark Gray
        "#b0b0b0",  # Light Gray
        "#000000",  # Black
    ]
    
    # Annual totals per country to derive cumulative loss and interannual variability
    annual_loss = (
        df.groupby(["iso3", "year"])["hansen_loss_ha"]
        .sum()
        .reset_index()
    )
    fig1_stats = (
        annual_loss.groupby("iso3")["hansen_loss_ha"]
        .agg(
            total_hansen_loss_Mha=lambda x: x.sum() / 1e6,
            sd_annual_loss_Mha=lambda x: x.std(ddof=1) / 1e6,
        )
        .reset_index()
    )
    fig1_stats = fig1_stats.sort_values("total_hansen_loss_Mha", ascending=False)
    # Use Royal Blue and Light Blue for contrast (single-color focus).
    for suffix, color in [("a", "#4169e1"), ("b", "#6baed6")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            fig1_stats["iso3"],
            fig1_stats["total_hansen_loss_Mha"],
            color=color,
            yerr=fig1_stats["sd_annual_loss_Mha"],
            ecolor="black",
            capsize=3,
        )
        ax.set_ylabel("Total tree cover loss (Mha)")
        ax.set_xlabel("")
        ax.set_title("Gross tree cover loss by country (2015–2023)", pad=12)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"figure1_total_loss_{suffix}.png", dpi=300)
        plt.close(fig)

    ratio_data = df.sort_values("year")
    iso_order = sorted(df["iso3"].unique())
    # Cycle through the scientific colors for the countries
    for suffix, linestyle in [("a", "solid"), ("b", "dashed")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, iso in enumerate(iso_order):
            color = colors[i % len(colors)]
            subset = ratio_data[ratio_data["iso3"] == iso]
            ax.plot(subset["year"], subset["discrepancy_ratio"], label=iso, color=color, linestyle=linestyle, linewidth=2)
        ax.set_ylabel("Discrepancy ratio (Hansen Gross / |FAO Net|)")
        ax.set_xlabel("Year")
        ax.set_title("Discrepancy ratio (Gross vs Net)", pad=12)
        ax.legend(title="ISO3", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"figure2_ratio_timeseries_{suffix}.png", dpi=300)
        plt.close(fig)

    # Use dark red for fire plots
    for suffix, log_scale in [("a", False), ("b", True)]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["viirs_fire_count"], df["hansen_loss_ha"], color="#b03030", alpha=0.6)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("Fire–loss relationship (log-log)", pad=12)
        else:
            ax.set_title("Fire–loss relationship", pad=12)
        ax.set_xlabel("Weighted Annual VIIRS fire count (Lag-adjusted)")
        ax.set_ylabel("Annual Hansen tree cover loss (ha)")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"figure3_fire_loss_scatter_{suffix}.png", dpi=300)
        plt.close(fig)

    prot_stats = df.dropna(subset=["protected_loss_share"]).copy()
    prot_stats = (
        prot_stats.groupby("iso3")["protected_loss_share"]
        .agg(
            mean_share="mean",
            sd_share=lambda x: x.std(ddof=1),
        )
        .reset_index()
    )
    prot_stats = prot_stats.sort_values("mean_share", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        prot_stats["iso3"],
        prot_stats["mean_share"],
        color="#4169e1",
        yerr=prot_stats["sd_share"],
        ecolor="black",
        capsize=3,
    )
    ax.set_ylabel("Mean protected loss share")
    ax.set_xlabel("ISO3")
    ax.set_title("Share of loss inside protected areas", pad=12)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure4_protected_share.png", dpi=300)
    plt.close(fig)

    examples = ["BRA", "BOL"]
    raw = pd.read_csv(FRA_RAW, encoding="latin-1").rename(columns={"ï»¿regions": "region"})
    raw["iso3"] = raw["iso3"].str.upper()
    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 4), sharey=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, iso in zip(axes, examples):
        if iso not in raw["iso3"].values:
            continue
        obs_row = raw[raw["iso3"] == iso].iloc[0]
        year_cols = [int(c) for c in raw.columns if c.isdigit()]
        obs = {y: obs_row[str(y)] for y in year_cols if pd.notna(obs_row[str(y)]) and 2015 <= y <= 2023}
        obs_years = sorted(obs.keys())
        obs_vals = [obs[y] for y in obs_years]
        pchip_series = pd.read_csv(FRA_PCHIP)
        pchip_series = pchip_series[pchip_series["iso3"] == iso].set_index("year")["forest_area_kha"]
        lin = pd.Series(index=pchip_series.index, dtype=float)
        lin.loc[obs_years] = obs_vals
        lin = lin.interpolate(method="linear")
        # PCHIP in Royal Blue, Linear in Dark Red
        ax.plot(pchip_series.index, pchip_series.values, color="#4169e1", label="PCHIP", linewidth=2)
        ax.plot(lin.index, lin.values, color="#b03030", linestyle="--", label="Linear", linewidth=2)
        ax.scatter(obs_years, obs_vals, color="black", label="Observed", zorder=5)
        ax.set_title(f"{iso} forest area interpolation", pad=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Forest area (kha)")
        ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure5_interpolation_comparison.png", dpi=300)
    plt.close(fig)


def run_panel_regression(df: pd.DataFrame) -> None:
    """
    Perform a Fixed Effects (LSDV) Panel Regression to assess scientific stability.
    Model: log(Loss) ~ log(Fire) + z(AgShare) + log(GDP) + CountryEffects
    """
    logger.info("Running Panel Regression (Fixed Effects) with Controls...")
    
    # Prepare data: Drop NaNs
    cols_needed = [
        "hansen_loss_ha", "viirs_fire_count", "protected_loss_ha", "iso3",
        "gdp_per_capita_ppp_const2017", "agriculture_value_added_share_gdp_pct"
    ]
    
    # Check if columns exist
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns for advanced regression: {missing}. Falling back to simple model.")
        reg_df = df.dropna(subset=["hansen_loss_ha", "viirs_fire_count", "protected_loss_ha", "iso3"]).copy()
        formula = "hansen_loss_ha ~ viirs_fire_count + protected_loss_ha + C(iso3)"
    else:
        reg_df = df.dropna(subset=cols_needed).copy()
        
        # Transformations
        # Log-Log for Fire and Loss (handle zeros)
        reg_df["log_loss"] = np.log(reg_df["hansen_loss_ha"] + 1)
        reg_df["log_fire"] = np.log(reg_df["viirs_fire_count"] + 1)
        reg_df["log_gdp"] = np.log(reg_df["gdp_per_capita_ppp_const2017"])
        
        # Standardize Agriculture Share
        reg_df["z_agri"] = (
            reg_df["agriculture_value_added_share_gdp_pct"] - reg_df["agriculture_value_added_share_gdp_pct"].mean()
        ) / reg_df["agriculture_value_added_share_gdp_pct"].std()
        
    # Model 1: Full Model
    formula_1 = "log_loss ~ log_fire + z_agri + log_gdp + C(iso3)"
    logger.info(f"Running Model 1: {formula_1}")
    model_1 = smf.ols(formula=formula_1, data=reg_df)
    results_1 = model_1.fit()
    
    # Model 2: No GDP (to check z_agri significance)
    formula_2 = "log_loss ~ log_fire + z_agri + C(iso3)"
    logger.info(f"Running Model 2 (No GDP): {formula_2}")
    model_2 = smf.ols(formula=formula_2, data=reg_df)
    results_2 = model_2.fit()

    summary_1 = results_1.summary().as_text()
    summary_2 = results_2.summary().as_text()
    
    print("Model 1 (Full):")
    print(summary_1)
    print("\nModel 2 (No GDP):")
    print(summary_2)
    
    # VIF Calculation (Model 1)
    vif_text = "\n\nVariance Inflation Factors (Model 1):\n"
    vif_text += "=====================================\n"
    try:
        y, X = dmatrices(formula_1, reg_df, return_type="dataframe")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_text += vif_data.to_string(index=False)
    except Exception as e:
        vif_text += f"Could not calculate VIF: {e}"
        
    # Save to file
    with open(FIG_DIR / "statistical_summary.txt", "w") as f:
        f.write("Scientific Validity Check: Panel Regression (Fixed Effects)\n")
        f.write("=========================================================\n\n")
        f.write("Model 1: Full (log_loss ~ log_fire + z_agri + log_gdp + FixedEffects)\n")
        f.write(summary_1)
        f.write(vif_text)
        f.write("\n\n---------------------------------------------------------\n")
        f.write("Model 2: No GDP (log_loss ~ log_fire + z_agri + FixedEffects)\n")
        f.write("Hypothesis: Dropping GDP improves z_agri significance due to multicollinearity.\n")
        f.write(summary_2)
    
    logger.info("Saved statistical summary to Figures/statistical_summary.txt")

def plot_sensitivity_analysis(df: pd.DataFrame) -> None:
    """
    Compare total loss across different tree cover thresholds (10, 30, 50).
    Addresses 'Savanna Blindness' by showing impact of threshold choice.
    """
    if "threshold" not in df.columns:
        return

    # Annual loss by country, year, and threshold to summarize totals and variability
    annual = (
        df.groupby(["iso3", "threshold", "year"])["hansen_loss_ha"]
        .sum()
        .reset_index()
    )
    sens_stats = (
        annual.groupby(["iso3", "threshold"])["hansen_loss_ha"]
        .agg(
            total_loss_Mha=lambda x: x.sum() / 1e6,
            sd_annual_loss_Mha=lambda x: x.std(ddof=1) / 1e6,
        )
        .reset_index()
    )

    iso_order = sorted(sens_stats["iso3"].unique())
    thresholds = sorted(sens_stats["threshold"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#4169e1", "#1b7837", "#c67c2c"]

    x = np.arange(len(iso_order))
    width = 0.8 / max(len(thresholds), 1)

    for i, thr in enumerate(thresholds):
        sub = sens_stats[sens_stats["threshold"] == thr].set_index("iso3").reindex(iso_order)
        ax.bar(
            x + i * width - 0.5 * width * (len(thresholds) - 1),
            sub["total_loss_Mha"],
            width=width,
            label=f"{thr}%",
            color=palette[i % len(palette)],
            yerr=sub["sd_annual_loss_Mha"],
            ecolor="black",
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(iso_order)
    ax.set_ylabel("Total tree cover loss (Mha)")
    ax.set_title("Sensitivity analysis: Loss by tree cover threshold", pad=12)
    ax.legend(title="Tree cover threshold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "figure6_sensitivity_analysis.png", dpi=300)
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        logger.error("Input file %s not found. Run processing script first.", DATA_PATH)
        return

    # Load full dataset (all thresholds)
    df_full = pd.read_csv(DATA_PATH)

    # 1. Sensitivity Analysis (using full data)
    plot_sensitivity_analysis(df_full)

    # 2. Filter for Main Analysis (Standard: Threshold 30)
    # This prevents "Triple Counting" of loss.
    target_threshold = 30
    if "threshold" in df_full.columns:
        df_30 = df_full[df_full["threshold"] == target_threshold].copy()
        logger.info("Filtered dataset to threshold=%d (Rows: %d/%d)", target_threshold, len(df_30), len(df_full))
    else:
        df_30 = df_full.copy()
        logger.warning("No 'threshold' column found. Using full dataset.")

    # 3. Build Tables (using df_30)
    tables = build_summary_tables(df_30)
    write_tables(tables)

    # 4. Plot Figures (using df_30)
    plot_figures(df_30)
    
    # 5. Statistical Analysis (using df_30)
    run_panel_regression(df_30)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
