"""PCHIP-interpolate FRA forest area to annual series and derive net change (V2).

Inputs:
    Data/bulk-download_fra_2025/FRA_Years_variables/1a_forestArea_2025_11_27.csv
Outputs:
    Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv

Columns:
    iso3, year, forest_area_kha (1000 ha), forest_area_ha, fao_source, net_change_ha
Net change is the year-over-year difference of forest_area_ha (negative = loss).

Spanish / Español:
Este script toma los datos de superficie forestal de FRA 2025 para los años
1990, 2000, 2010, 2015, 2020 y 2025 y aplica interpolación PCHIP para
obtener una serie anual suave y monótona para 2015–2023. La columna
``net_change_ha`` representa el cambio neto año a año de la superficie
forestal (valores negativos = pérdida neta). El resultado se utiliza para
calcular la razón de discrepancia entre la pérdida bruta de Hansen y el
cambio neto de FRA en el panel principal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parent
FRA_INPUT = ROOT / "Data/bulk-download_fra_2025/FRA_Years_variables/1a_forestArea_2025_11_27.csv"
OUTPUT = ROOT / "Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv"
TARGET_ISOS = ["BRA", "BOL", "PER", "COL", "ECU", "VEN", "GUY", "SUR", "ARG", "CHL", "PRY", "URY"]
YEARS_OUT = np.arange(2015, 2024)


def interpolate_country(sub: pd.DataFrame, iso: str, year_cols: list[str]) -> pd.DataFrame:
    observed_years = []
    observed_vals = []
    for yc in year_cols:
        val = sub[yc].values
        if len(val) and pd.notna(val[0]):
            observed_years.append(int(yc))
            observed_vals.append(float(val[0]))
    if len(observed_years) < 2:
        return pd.DataFrame(columns=["iso3", "year", "forest_area_kha", "forest_area_ha"])

    observed_years = np.array(observed_years)
    observed_vals = np.array(observed_vals)
    
    # Sort by year
    idx = np.argsort(observed_years)
    observed_years = observed_years[idx]
    observed_vals = observed_vals[idx]

    # PCHIP preserves monotonicity; avoids artificial overshoot.
    interpolator = PchipInterpolator(observed_years, observed_vals, extrapolate=False)
    
    interp_vals = []
    max_obs_year = observed_years.max()
    last_val = observed_vals[np.argmax(observed_years)]
    
    for y in YEARS_OUT:
        if y <= max_obs_year:
            # Interpolate
            val = float(interpolator(y))
        else:
            # Extrapolate: Hold Last Value (Conservative)
            val = float(last_val)
        interp_vals.append(val)
        
    df = pd.DataFrame(
        {
            "iso3": iso,
            "year": YEARS_OUT.astype(int),
            "forest_area_kha": interp_vals,
        }
    )
    df["forest_area_ha"] = df["forest_area_kha"] * 1_000.0
    return df


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FRA_INPUT, encoding="latin-1").rename(columns={"ï»¿regions": "region"})
    if "iso3" not in df.columns:
        raise RuntimeError("FRA file missing iso3 column.")

    year_cols = [c for c in df.columns if c.isdigit()]
    df = df[df["iso3"].isin(TARGET_ISOS)].copy()
    df["iso3"] = df["iso3"].str.upper()

    frames = []
    for iso, sub in df.groupby("iso3"):
        frames.append(interpolate_country(sub, iso, year_cols))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["iso3", "year"])
    # Derive net change (ha) year-over-year; NaN for first year.
    out["net_change_ha"] = (
        out.sort_values(["iso3", "year"])
        .groupby("iso3")["forest_area_ha"]
        .diff()
    )
    out["fao_source"] = "FRA2025_PCHIP"
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
