"""Helper to load raw VIIRS fire detections for spatial filtering.

Spanish / Español:
Este módulo localiza y lee los archivos CSV de VIIRS para Sudamérica,
aplica filtros espaciales (caja geográfica), de tipo de fuente y de
confianza, elimina duplicados y devuelve un DataFrame con las detecciones
de fuego relevantes (latitud, longitud, fecha y país ISO3). Estas
detecciones se utilizan posteriormente para calcular métricas de
intensidad de fuego sobre píxeles forestales.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

import pandas as pd


ROOT = Path(__file__).resolve().parent
VIIRS_ROOTS = [
    ROOT / "Data/viirs-snpp",
    ROOT / "Data/viirs-snpp-1",
    ROOT / "Data/viirs-snpp-2",
    ROOT / "Data/viirs-snpp-3",
    ROOT / "Data/viirs-snpp-4",
    ROOT / "Data/viirs-snpp-5",
    ROOT / "Data/viirs-snpp-6",
    ROOT / "Data/viirs-snpp-7",
    ROOT / "Data/viirs-snpp-8",
    ROOT / "Data/viirs-snpp-9",
    ROOT / "Data/viirs-snpp-10",
]

SA_BBOX = (-85.0, -60.0, -30.0, 15.0)  # lon_min, lat_min, lon_max, lat_max
CONF_ACCEPT = {"n", "nominal", "h", "high"}
TARGET_ISOS = {"BRA", "BOL", "PER", "COL", "ECU", "VEN", "GUY", "SUR", "ARG", "CHL", "PRY", "URY"}


def iter_viirs_csv_files() -> Iterator[Path]:
    for root in VIIRS_ROOTS:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".csv"):
                    yield Path(dirpath) / fn


def infer_iso_from_filename(path: Path) -> str:
    name = path.stem
    parts = name.split("_")
    if len(parts) >= 3:
        candidate = parts[2][:3].upper()
        if candidate in TARGET_ISOS:
            return candidate
    return ""


def load_viirs_points(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load all VIIRS points within SA_BBOX and years.
    Returns DataFrame with columns: [lat, lon, year, iso3, acq_date]
    """
    files = list(iter_viirs_csv_files())
    files = [p for p in files if infer_iso_from_filename(p) in TARGET_ISOS]
    
    frames = []
    usecols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "type"]
    
    for csv_path in files:
        iso = infer_iso_from_filename(csv_path)
        # Read in chunks to be safe, though not strictly necessary if memory is ample
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=100_000):
            chunk = chunk.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
            
            # BBox filter
            mask_bbox = (
                (chunk["latitude"] >= SA_BBOX[1])
                & (chunk["latitude"] <= SA_BBOX[3])
                & (chunk["longitude"] >= SA_BBOX[0])
                & (chunk["longitude"] <= SA_BBOX[2])
            )
            sub = chunk.loc[mask_bbox].copy()
            if sub.empty:
                continue
            
            # Type filter: Exclude 2 (static) and 3 (offshore)
            # 0 = presumed vegetation fire, 1 = active volcano, 2 = other static land source, 3 = offshore
            sub = sub[~sub["type"].isin([2, 3])]
            if sub.empty:
                continue

            # Confidence filter
            conf = sub["confidence"].astype(str).str.lower()
            sub = sub[conf.isin(CONF_ACCEPT)]
            if sub.empty:
                continue
                
            # Date and Year
            sub["acq_date"] = pd.to_datetime(sub["acq_date"], errors="coerce").dt.date
            sub = sub.dropna(subset=["acq_date"])
            sub["year"] = pd.to_datetime(sub["acq_date"]).dt.year.astype(int)
            sub = sub.dropna(subset=["year"])
            sub = sub[(sub["year"] >= start_year) & (sub["year"] <= end_year)]
            
            if not sub.empty:
                sub["iso3"] = iso
                frames.append(sub[["latitude", "longitude", "year", "iso3", "acq_date"]])
    
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "year", "iso3", "acq_date"])
        
    return pd.concat(frames, ignore_index=True)
