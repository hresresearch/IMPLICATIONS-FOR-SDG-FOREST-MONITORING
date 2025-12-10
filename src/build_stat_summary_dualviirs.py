"""Recompute statistical summary for dual VIIRS metrics and all thresholds.

Spanish / Español:
Este módulo vuelve a estimar los modelos de regresión de panel para ambas
métricas de VIIRS (conteo y píxel‑día único) y para todos los umbrales de
cobertura arbórea (10, 30 y 50 %). Para cada combinación se ajusta un
modelo log‑log con efectos fijos por país y se escribe un resumen textual
completo en ``Figures/statistical_summary.txt`` para documentar la
robustez de la elasticidad del fuego.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_ALL = ROOT / "final_paper_dataset_v3_dualviirs_all_thresholds.csv"
OUT = ROOT / "Figures/statistical_summary.txt"


def fit_models(df: pd.DataFrame) -> str:
    lines = []
    specs = [
        ("viirs_fire_count", "Count"),
        ("viirs_fire_unique_pxday", "UniquePxDay"),
    ]
    thresholds = [10, 30, 50]

    for thr in thresholds:
        sub_thr = df[df["threshold"] == thr].copy()
        for fire_col, label in specs:
            sub = sub_thr.dropna(subset=[fire_col, "hansen_loss_ha"])
            if sub.empty:
                continue
            sub["log_loss"] = np.log(sub["hansen_loss_ha"] + 1)
            sub["log_fire"] = np.log(sub[fire_col] + 1)
            sub["log_gdp"] = np.log(sub["gdp_per_capita_ppp_const2017"] + 1)
            ag = sub["agriculture_value_added_share_gdp_pct"]
            sub["z_agri"] = (ag - ag.mean()) / ag.std(ddof=0) if ag.std(ddof=0) > 0 else 0

            model = smf.ols("log_loss ~ log_fire + z_agri + log_gdp + C(iso3)", data=sub).fit()
            lines.append(f"Threshold {thr}, Fire metric: {label}")
            lines.append(model.summary().as_text())
            lines.append("\n" + "=" * 72 + "\n")
    return "\n".join(lines)


def main() -> None:
    if not DATA_ALL.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ALL}")
    df = pd.read_csv(DATA_ALL)
    txt = fit_models(df)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(txt)
    logger.info("Wrote %s", OUT)


if __name__ == "__main__":
    main()
