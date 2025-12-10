"""V3 Hansen/FRA/WDPA/VIIRS processing with equal-area projection.

Key upgrades vs. V2:
- **Center Pixel Rule**: `all_touched=False` for all masks to prevent "Border Inflation" (double counting).
- **V3 WDPA**: Uses `amazon_wdpa_filtered_102033_v3.geojson` (time-filtered <= 2015).
- **Clean Cache**: Uses `Data/processed/hansen_tile_cache_v3` to ensure no V2 artifacts are used.

Spanish / Español:
Este módulo implementa la versión V3 del procesamiento espacial, integrando
la pérdida de cobertura arbórea de Hansen, las áreas protegidas WDPA, la
superficie forestal FRA y las detecciones activas de fuego VIIRS en una
rejilla común en proyección equal‑area ESRI:102033. Aplica la regla del
“píxel central” (``all_touched=False``) para evitar la inflación de bordes,
reproyecta y repara las geometrías de WDPA filtradas hasta 2015 y utiliza un
cache de teselas para acelerar el reprocesamiento sin reutilizar artefactos
de versiones anteriores. El resultado final es un conjunto de datos
agregado por país, año y umbral de cobertura arbórea listo para el análisis
estadístico y la reproducción del artículo.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from pyproj import Transformer
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping

import viirs_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_CRS = "ESRI:102033"
TARGET_ISOS = ["BRA", "BOL", "PER", "COL", "ECU", "VEN", "GUY", "SUR", "ARG", "CHL", "PRY", "URY"]


@dataclass
class HansenTileSet:
    key: str
    lossyear_path: Path
    treecover2000_path: Optional[Path]
    datamask_path: Path
    gain_path: Optional[Path]


def process_tile_wrapper(args) -> pd.DataFrame:
    """
    Wrapper to allow executor.map with per-tile VIIRS subsets.
    """
    (
        tile,
        regions_path,
        wdpa_path,
        viirs_points,
        start_year,
        end_year,
        treecover_thresholds,
        cache_dir,
        target_crs,
        target_isos,
    ) = args
    return process_single_tile(
        tile=tile,
        regions_path=regions_path,
        wdpa_path=wdpa_path,
        viirs_points=viirs_points,
        start_year=start_year,
        end_year=end_year,
        treecover_thresholds=treecover_thresholds,
        cache_dir=cache_dir,
        target_crs=target_crs,
        target_isos=target_isos,
    )


def find_matching_tiles(
    data_dir: Path, allow_missing_treecover: bool = False, strict: bool = True
) -> List[HansenTileSet]:
    files_by_key: Dict[str, Dict[str, Path]] = {}
    for path in data_dir.glob("Hansen_GFC-2024-v1.12_*_*.tif"):
        name = path.name
        key = None
        if "_lossyear_" in name:
            key = name.split("_lossyear_")[1].rsplit(".", 1)[0]
            files_by_key.setdefault(key, {})["lossyear"] = path
        elif "_treecover2000_" in name:
            key = name.split("_treecover2000_")[1].rsplit(".", 1)[0]
            files_by_key.setdefault(key, {})["treecover2000"] = path
        elif "_datamask_" in name:
            key = name.split("_datamask_")[1].rsplit(".", 1)[0]
            files_by_key.setdefault(key, {})["datamask"] = path
        elif "_gain_" in name:
            key = name.split("_gain_")[1].rsplit(".", 1)[0]
            files_by_key.setdefault(key, {})["gain"] = path

    tiles: List[HansenTileSet] = []
    for key, files in files_by_key.items():
        ly = files.get("lossyear")
        tc = files.get("treecover2000")
        dm = files.get("datamask")
        gn = files.get("gain")

        missing_layers_info = []
        if not ly:
            missing_layers_info.append("lossyear")
        if not dm:
            missing_layers_info.append("datamask")
        if not tc and not allow_missing_treecover:
            missing_layers_info.append("treecover2000 (not allowed to be missing)")

        if missing_layers_info:
            logger.warning("Skipping tile %s due to missing layers: %s", key, ", ".join(missing_layers_info))
            continue

        tiles.append(
            HansenTileSet(
                key=key,
                lossyear_path=ly,
                treecover2000_path=tc,
                datamask_path=dm,
                gain_path=gn
            )
        )

    logger.info("Matched %d Hansen tiles in %s", len(tiles), data_dir)
    return tiles


def load_regions(regions_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(regions_path)
    if "iso3" not in gdf.columns:
        raise RuntimeError(f"Expected iso3 column in {regions_path}")
    gdf["iso3"] = gdf["iso3"].str.upper()
    gdf = gdf[gdf["iso3"].isin(TARGET_ISOS)].copy()
    if gdf.empty:
        raise RuntimeError("No regions after ISO3 filter; check input.")
    if gdf.crs is None:
        raise RuntimeError("Regions file missing CRS.")
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_wdpa(wdpa_path: Path) -> Optional[gpd.GeoDataFrame]:
    if not wdpa_path.exists():
        logger.warning("WDPA path %s not found; protected overlay will be skipped.", wdpa_path)
        return None
    gdf = gpd.read_file(wdpa_path)
    if "iso3" not in gdf.columns:
        raise RuntimeError(f"WDPA file {wdpa_path} missing iso3 column.")
    gdf["iso3"] = gdf["iso3"].str.upper()
    gdf = gdf[gdf["iso3"].isin(TARGET_ISOS)].copy()
    if gdf.crs is None:
        raise RuntimeError("WDPA file missing CRS.")
    gdf = gdf.to_crs("EPSG:4326")
    try:
        from shapely import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    
    # Ensure STATUS_YR is int
    if "STATUS_YR" in gdf.columns:
        gdf["STATUS_YR"] = pd.to_numeric(gdf["STATUS_YR"], errors="coerce").fillna(0).astype(int)
    else:
        gdf["STATUS_YR"] = 0
        
    return gdf


def row_pixel_areas_ha(transform, rows: int, transformer: Transformer) -> np.ndarray:
    width_deg = transform.a
    height_deg = abs(transform.e)
    areas = np.zeros(rows, dtype=np.float64)
    for r in range(rows):
        lon_c, lat_c = rasterio.transform.xy(transform, r + 0.5, 0.5)
        lon_right = lon_c + width_deg
        lat_bottom = lat_c - height_deg
        x0, y0 = transformer.transform(lon_c, lat_c)
        x1, y1 = transformer.transform(lon_right, lat_c)
        x2, y2 = transformer.transform(lon_c, lat_bottom)
        width_m = abs(x1 - x0)
        height_m = abs(y2 - y0)
        areas[r] = (width_m * height_m) / 10_000.0
    return areas



import concurrent.futures
import functools

# Helper function must be top-level for pickling
def process_single_tile(
    tile: HansenTileSet,
    regions_path: Path,
    wdpa_path: Optional[Path],
    viirs_points: pd.DataFrame,
    start_year: int,
    end_year: int,
    treecover_thresholds: List[int],
    cache_dir: Path,
    target_crs: str,
    target_isos: List[str]
) -> pd.DataFrame:
    """
    Process a single Hansen tile. Returns a DataFrame of results.
    """
    # Re-initialize logger for worker
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"worker_{tile.key}")
    
    th_str = "_".join(map(str, sorted(treecover_thresholds)))
    cache_file = cache_dir / f"{tile.key}_th{th_str}.csv"
    
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        required_cols = {
            "iso3",
            "year",
            "threshold",
            "hansen_loss_ha",
            "protected_loss_ha",
            "viirs_fire_count",
            "viirs_fire_unique_pxday",
        }
        if required_cols.issubset(set(cached.columns)):
            logger.info("Loaded cached tile %s", tile.key)
            return cached
        else:
            logger.info("Cached tile %s missing new columns; recomputing.", tile.key)
    
    # Load Regions (Worker needs its own copy or we pass geometries? Passing path is safer for pickling)
    # But loading regions every time is slow.
    # Optimization: Pass the regions GDF? GDF is picklable.
    # Let's try loading it once inside worker if not passed.
    # Actually, 'regions' is small enough to pass.
    # But for now, let's load it to be safe and simple.
    regions = load_regions(regions_path)
    wdpa = load_wdpa(wdpa_path) if wdpa_path else None
    
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    years = list(range(start_year, end_year + 1))
    
    tile_records = []
    
    # 1. Load Rasters
    with rasterio.open(tile.lossyear_path) as loss_ds:
        tile_geom = box(*loss_ds.bounds)
        overlapping = regions[regions.intersects(tile_geom)]
        if overlapping.empty:
            return pd.DataFrame()
        
        logger.info("Processing tile %s (%d regions)", tile.key, len(overlapping))
        union_geom = overlapping.unary_union
        loss_arr, loss_transform = rio_mask(
            loss_ds, [mapping(union_geom)], crop=True, all_touched=False, filled=True
        )

    with rasterio.open(tile.datamask_path) as dm_ds:
        dm_arr, _ = rio_mask(dm_ds, [mapping(union_geom)], crop=True, all_touched=False, filled=True)

    tc_arr = None
    if tile.treecover2000_path:
        with rasterio.open(tile.treecover2000_path) as tc_ds:
            tc_arr, _ = rio_mask(tc_ds, [mapping(union_geom)], crop=True, all_touched=False, filled=True)
    
    gain_arr = None
    if tile.gain_path:
         with rasterio.open(tile.gain_path) as gain_ds:
            gain_arr, _ = rio_mask(gain_ds, [mapping(union_geom)], crop=True, all_touched=False, filled=True)

    loss_data = loss_arr[0]
    dm_data = dm_arr[0]
    tc_data = tc_arr[0] if tc_arr is not None else None
    gain_data = gain_arr[0] if gain_arr is not None else None
    
    rows, cols = loss_data.shape
    row_areas = row_pixel_areas_ha(loss_transform, rows, transformer)

    # VIIRS Filtering
    crop_bounds = rasterio.transform.array_bounds(rows, cols, loss_transform)
    in_box = viirs_points[
        (viirs_points["longitude"] >= crop_bounds[0]) &
        (viirs_points["longitude"] <= crop_bounds[2]) &
        (viirs_points["latitude"] >= crop_bounds[1]) &
        (viirs_points["latitude"] <= crop_bounds[3])
    ].copy()
    
    if not in_box.empty:
        rs, cs = rasterio.transform.rowcol(loss_transform, in_box["longitude"].values, in_box["latitude"].values)
        in_box["row"] = rs
        in_box["col"] = cs
        valid_idx = (in_box["row"] >= 0) & (in_box["row"] < rows) & (in_box["col"] >= 0) & (in_box["col"] < cols)
        in_box = in_box[valid_idx]
    
    for _, row_geo in overlapping.iterrows():
        iso3 = row_geo["iso3"]
        region_geom = row_geo.geometry
        try:
            region_mask = rasterio.features.geometry_mask(
                [mapping(region_geom)],
                out_shape=loss_data.shape,
                transform=loss_transform,
                invert=True,
                all_touched=False,
            )
        except Exception as exc:
            logger.error("Region mask failure for tile %s iso %s: %s", tile.key, iso3, exc)
            raise

        prot_mask = None
        if wdpa is not None:
            prot_row = wdpa[wdpa["iso3"] == iso3]
            if not prot_row.empty:
                try:
                    valid_geom = prot_row.unary_union
                    prot_mask = rasterio.features.geometry_mask(
                        [mapping(valid_geom)],
                        out_shape=loss_data.shape,
                        transform=loss_transform,
                        invert=True,
                        all_touched=False,
                    )
                except Exception as exc:
                    logger.error("WDPA mask failure for tile %s iso %s: %s", tile.key, iso3, exc)
                    raise

        for thresh in treecover_thresholds:
            mask_valid = (dm_data == 1) & region_mask
            if tc_data is not None:
                mask_valid &= (tc_data > thresh)
            
            if not mask_valid.any():
                continue
            
            for year in years:
                code = year - 2000
                if code <= 0: 
                    continue
                
                loss_pixels = (loss_data == code) & mask_valid
                
                counts_per_row = loss_pixels.sum(axis=1)
                area_ha = float((counts_per_row * row_areas).sum())
                
                prot_area = 0.0
                if prot_mask is not None:
                    prot_loss_pixels = loss_pixels & prot_mask
                    if prot_loss_pixels.any():
                        prot_area = float((prot_loss_pixels.sum(axis=1) * row_areas).sum())

                fire_count_t = 0
                fire_count_tm1 = 0
                
                if not in_box.empty:
                    fires_t = in_box[in_box["year"] == year]
                    if not fires_t.empty:
                        r_idx = fires_t["row"].values
                        c_idx = fires_t["col"].values
                        is_forest = mask_valid[r_idx, c_idx]
                        fires_t = fires_t[is_forest]
                        fire_count_t = int(len(fires_t))
                        if not fires_t.empty:
                            unique_t = fires_t.dropna(subset=["acq_date"]).drop_duplicates(subset=["row", "col", "acq_date"])
                            unique_fire_count_t = int(len(unique_t))
                        else:
                            unique_fire_count_t = 0
                    else:
                        unique_fire_count_t = 0
                    
                    fires_tm1 = in_box[in_box["year"] == year - 1]
                    if not fires_tm1.empty:
                        r_idx = fires_tm1["row"].values
                        c_idx = fires_tm1["col"].values
                        is_forest = mask_valid[r_idx, c_idx]
                        fires_tm1 = fires_tm1[is_forest]
                        fire_count_tm1 = int(len(fires_tm1))
                        if not fires_tm1.empty:
                            unique_tm1 = fires_tm1.dropna(subset=["acq_date"]).drop_duplicates(subset=["row", "col", "acq_date"])
                            unique_fire_count_tm1 = int(len(unique_tm1))
                        else:
                            unique_fire_count_tm1 = 0
                    else:
                        unique_fire_count_tm1 = 0
                else:
                    unique_fire_count_t = 0
                    unique_fire_count_tm1 = 0

                viirs_fire_metric = 0.5 * fire_count_t + 0.5 * fire_count_tm1
                viirs_fire_unique_metric = 0.5 * unique_fire_count_t + 0.5 * unique_fire_count_tm1

                if area_ha > 0 or viirs_fire_metric > 0:
                    tile_records.append({
                        "iso3": iso3,
                        "year": year,
                        "threshold": thresh,
                        "hansen_loss_ha": area_ha,
                        "protected_loss_ha": prot_area,
                        "viirs_fire_count": viirs_fire_metric,
                        "viirs_fire_unique_pxday": viirs_fire_unique_metric,
                    })

    tile_df = pd.DataFrame(tile_records)
    if not tile_df.empty:
         if gain_data is not None:
            gain_records = []
            for _, row_geo in overlapping.iterrows():
                iso3 = row_geo["iso3"]
                try:
                    region_mask = rasterio.features.geometry_mask(
                        [mapping(row_geo.geometry)],
                        out_shape=loss_data.shape,
                        transform=loss_transform,
                        invert=True,
                        all_touched=False,
                    )
                except Exception as exc:
                    logger.error("Gain mask failure for tile %s iso %s: %s", tile.key, iso3, exc)
                    raise
                
                mask_land = (dm_data == 1) & region_mask
                gain_pixels = (gain_data == 1) & mask_land
                if gain_pixels.any():
                    g_area = float((gain_pixels.sum(axis=1) * row_areas).sum())
                    gain_records.append({"iso3": iso3, "hansen_gain_2000_2012_ha": g_area})
            
            if gain_records:
                g_df = pd.DataFrame(gain_records)
                tile_df = tile_df.merge(g_df, on="iso3", how="left")

    # Always save cache, even if empty (to avoid re-processing)
    tile_df.to_csv(cache_file, index=False)
    return tile_df


def compute_hansen_loss(
    data_dir: Path,
    regions_path: Path,
    start_year: int,
    end_year: int,
    treecover_thresholds: List[int],
    allow_missing_treecover: bool,
    strict_tiles: bool,
    wdpa_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    workers: int = 5,
) -> pd.DataFrame:
    # Main process loads data once
    tiles = find_matching_tiles(data_dir, allow_missing_treecover=allow_missing_treecover, strict=strict_tiles)
    tile_key_set = {t.key for t in tiles}

    logger.info("Loading VIIRS fire points...")
    viirs_points = viirs_loader.load_viirs_points(start_year, end_year)
    logger.info("Loaded %d VIIRS points.", len(viirs_points))
    if not viirs_points.empty:
        # Pre-bin VIIRS detections to tiles to avoid scanning full frame per tile
        def make_tile_key(lon, lat):
            if pd.isna(lon) or pd.isna(lat):
                return None
            lat_top = int(np.ceil(lat / 10.0) * 10)
            lon_left = int(np.floor(lon / 10.0) * 10)
            ns = "N" if lat_top >= 0 else "S"
            ew = "E" if lon_left >= 0 else "W"
            return f"{abs(lat_top):02d}{ns}_{abs(lon_left):03d}{ew}"

        viirs_points["tile_key"] = [
            make_tile_key(lon, lat) for lon, lat in zip(viirs_points["longitude"], viirs_points["latitude"])
        ]
        viirs_points = viirs_points[viirs_points["tile_key"].isin(tile_key_set)]
        viirs_map = {
            k: g.drop(columns=["tile_key"]).reset_index(drop=True) for k, g in viirs_points.groupby("tile_key")
        }
    else:
        viirs_map = {}
    empty_viirs = pd.DataFrame(columns=["latitude", "longitude", "year", "iso3", "acq_date"])

    cache_dir = (cache_dir or Path("Data/processed/hansen_tile_cache_v3")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = [
        (
            tile,
            regions_path,
            wdpa_path,
            viirs_map.get(tile.key, empty_viirs),
            start_year,
            end_year,
            treecover_thresholds,
            cache_dir,
            TARGET_CRS,
            TARGET_ISOS,
        )
        for tile in tiles
    ]

    all_frames = []
    max_workers = max(1, workers)
    if max_workers == 1:
        logger.info("Processing %d tiles sequentially with pre-binned VIIRS data.", len(tasks))
        results = [process_tile_wrapper(t) for t in tasks]
    else:
        logger.info("Processing %d tiles with %d workers (pre-binned VIIRS).", len(tasks), max_workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_tile_wrapper, tasks))

    for res in results:
        if not res.empty:
            all_frames.append(res)
        else:
            logger.info("Tile result empty (likely no overlap): skipping append.")

    processed_tiles = len(results)
    logger.info("Processed %d tiles (requested %d).", processed_tiles, len(tasks))

    if not all_frames:
        return pd.DataFrame(columns=["iso3", "year", "threshold", "hansen_loss_ha"])

    combined = pd.concat(all_frames, ignore_index=True)
    
    group_cols = ["iso3", "year", "threshold"]
    agg_dict = {
        "hansen_loss_ha": "sum",
        "protected_loss_ha": "sum",
        "viirs_fire_count": "sum",
        "viirs_fire_unique_pxday": "sum",
    }
    if "hansen_gain_2000_2012_ha" in combined.columns:
        agg_dict["hansen_gain_2000_2012_ha"] = "sum"

    final_df = combined.groupby(group_cols, as_index=False).agg(agg_dict)
    return final_df.sort_values(group_cols)




def prepare_socio_economic_data(
    csv_path: Path, target_years: Iterable[int]
) -> pd.DataFrame:
    """
    Loads decadal socio-economic data (2000, 2010, 2020), interpolates for target years,
    and handles extrapolation (Hold Last Value).
    """
    if not csv_path.exists():
        logger.warning("Socio-economic data not found at %s", csv_path)
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    cols_needed = ["iso3", "year", "gdp_per_capita_ppp_const2017", "agriculture_value_added_share_gdp_pct"]
    
    # Check if columns exist
    available_cols = [c for c in cols_needed if c in df.columns]
    if len(available_cols) < len(cols_needed):
        logger.warning("Missing columns in socio-economic data. Found: %s", available_cols)
        return pd.DataFrame()

    df = df[available_cols].copy()
    df["iso3"] = df["iso3"].str.upper()
    df["year"] = df["year"].astype(int)
    
    # Filter for known decades to ensure clean interpolation
    df = df[df["year"].isin([2000, 2010, 2020])]
    
    interpolated_frames = []
    
    for iso in df["iso3"].unique():
        sub = df[df["iso3"] == iso].sort_values("year")
        if sub.empty:
            continue
            
        # Reindex to cover full range 2000-2023 (or max target year)
        full_range = range(2000, max(target_years) + 1)
        sub = sub.set_index("year").reindex(full_range)
        
        # Interpolate Linear (covers 2010-2020 gaps)
        sub = sub.interpolate(method="linear")
        
        # Extrapolate (Forward Fill for > 2020)
        sub = sub.ffill()
        
        # Filter to target years
        sub = sub.loc[target_years].reset_index()
        sub["iso3"] = iso
        interpolated_frames.append(sub)
        
    if not interpolated_frames:
        return pd.DataFrame()
        
    result = pd.concat(interpolated_frames, ignore_index=True)
    return result


def merge_all(
    hansen_df: pd.DataFrame,
    fra_path: Path,
    output_path: Path,
    panel_path: Optional[Path] = None,
    socio_path: Optional[Path] = None,
) -> pd.DataFrame:
    # 1. Merge FRA
    fra = pd.read_csv(fra_path)
    fra = fra.rename(
        columns={
            "forest_area_kha": "fao_forest_area_kha",
            "forest_area_ha": "fao_forest_area_ha",
            "net_change_ha": "fao_net_change_ha",
        }
    )
    fra["iso3"] = fra["iso3"].str.upper()
    fra["year"] = fra["year"].astype(int)

    merged = hansen_df.merge(fra, how="left", on=["iso3", "year"])
    merged["protected_loss_share"] = merged["protected_loss_ha"] / merged["hansen_loss_ha"]

    def calc_ratio(row):
        fao_nc = row.get("fao_net_change_ha")
        hansen = row.get("hansen_loss_ha")
        if pd.isna(fao_nc) or pd.isna(hansen):
            return np.nan
        if fao_nc >= 0:
            return np.nan 
        return hansen / (-fao_nc)

    merged["discrepancy_ratio"] = merged.apply(calc_ratio, axis=1)

    # 2. Merge Panel (Optional)
    if panel_path:
        panel = pd.read_csv(panel_path)
        if "iso3" in panel.columns and "year" in panel.columns:
            panel["iso3"] = panel["iso3"].str.upper()
            panel["year"] = panel["year"].astype(int)
            merged = merged.merge(panel, how="left", on=["iso3", "year"], suffixes=("", "_panel"))

    # 3. Merge Socio-Economic Data (New for V3 Refactoring)
    if socio_path:
        target_years = merged["year"].unique()
        socio_df = prepare_socio_economic_data(socio_path, target_years)
        if not socio_df.empty:
            merged = merged.merge(socio_df, how="left", on=["iso3", "year"])
            logger.info("Merged socio-economic data (GDP, AgShare).")
        else:
            logger.warning("Socio-economic data processing returned empty.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Wrote merged dataset to %s (rows=%d)", output_path, len(merged))
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 Hansen/FRA/WDPA/VIIRS processor (equal-area).")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with Hansen GFC-2024-v1.12 rasters.")
    parser.add_argument(
        "--regions-path",
        type=Path,
        default=Path("Data/processed/amazon_regions_102033.geojson"),
        help="Amazon regions GeoJSON reprojected to EPSG:102033.",
    )
    parser.add_argument(
        "--wdpa-path",
        type=Path,
        default=Path("Data/processed/amazon_wdpa_filtered_102033_v3.geojson"),
        help="Filtered WDPA GeoJSON in EPSG:102033 (V3: Time-filtered).",
    )
    parser.add_argument(
        "--fra-path",
        type=Path,
        default=Path("Data/processed/fao_forest_interpolated_2015_2023_pchip_v2.csv"),
        help="PCHIP-interpolated FRA forest area file.",
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[10, 30, 50], help="Treecover thresholds to process.")
    parser.add_argument("--allow-missing-treecover", action="store_true", default=False)
    parser.add_argument("--strict-tiles", action="store_true", default=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("Data/processed/hansen_tile_cache_v3"),
        help="Directory to store per-tile cached results (V3 Clean Cache).",
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=None,
        help="Optional panel CSV to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_paper_dataset_v3.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--socio-path",
        type=Path,
        default=Path("final_paper_dataset.csv"),
        help="Path to original dataset with socio-economic variables (GDP, AgShare).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of worker processes (set to 1 for sequential to avoid pickling huge VIIRS frames).",
    )
    parser.add_argument(
        "--all-thresholds-output",
        type=Path,
        default=None,
        help="Optional output CSV path for all-threshold dataset. Defaults to <output_stem>_all_thresholds.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    full_df = compute_hansen_loss(
        data_dir=args.data_dir,
        regions_path=args.regions_path,
        start_year=args.start_year,
        end_year=args.end_year,
        treecover_thresholds=args.thresholds,
        allow_missing_treecover=args.allow_missing_treecover,
        strict_tiles=args.strict_tiles,
        wdpa_path=args.wdpa_path,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )

    if args.all_thresholds_output:
        full_out = args.all_thresholds_output
    else:
        full_out = args.output.parent / f"{args.output.stem}_all_thresholds{args.output.suffix}"
    merge_all(
        hansen_df=full_df,
        fra_path=args.fra_path,
        output_path=full_out,
        panel_path=args.panel_path,
        socio_path=args.socio_path,
    )
    
    df_30 = full_df[full_df["threshold"] == 30].copy()
    if df_30.empty:
        logger.warning("No data for threshold 30 found. Saving empty or first available.")
        if not full_df.empty:
             df_30 = full_df
    
    merge_all(
        hansen_df=df_30,
        fra_path=args.fra_path,
        output_path=args.output,
        panel_path=args.panel_path,
        socio_path=args.socio_path,
    )


if __name__ == "__main__":
    main()
