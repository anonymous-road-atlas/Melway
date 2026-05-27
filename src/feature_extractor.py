"""
feature_extractor.py
--------------------
Stage 2 of the pipeline:
  1. Load modern road GeoJSON.
  2. Compute HIRONEX features (Var_line, Edge_contrast, ROI) for each segment.
  3. Predict existence probability via a pre-trained Random Forest.

Returns a pandas DataFrame (df_feat) with geometry, features, and Pred column.
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import shape, MultiLineString
from shapely.ops import linemerge
from tqdm import tqdm
from joblib import load

from logger import get_logger
from utils import compute_hironex_roi_for_line, sample_raster_rgb
import config as cfg

log = get_logger("melway.feature_extractor")


def extract_features(src: rasterio.DatasetReader, modern_path: str,
                     geo_crs, metric_crs: int) -> pd.DataFrame:
    """
    Extract HIRONEX features for all modern road segments (6 target classes).

    Parameters
    ----------
    src         : Open rasterio dataset (RGBA TIF, GEO_CRS).
    modern_path : Path to modern roads GeoJSON.
    geo_crs     : CRS of the TIF (rasterio CRS object).
    metric_crs  : EPSG integer for metric CRS.

    Returns
    -------
    DataFrame with columns: gid, osm_id, fclass, Var_line, Edge_contrast,
                             ROI, ROI_norm, geometry, pixel_count.
    """
    modern = gpd.read_file(modern_path)
    modern = modern[modern["fclass"].isin(cfg.TARGET_CLASSES)].copy()
    log.info("Loaded %d modern features (6 target classes)", len(modern))

    with open(modern_path, "r") as f:
        gj = json.load(f)

    modern_gids = set(modern["gid"].unique())
    features_raw = [
        ft for ft in gj["features"]
        if ft.get("properties", {}).get("fclass") in cfg.TARGET_CLASSES
        and ft.get("properties", {}).get("gid") in modern_gids
    ]

    rgb = np.stack([src.read(b + 1) for b in range(3)], axis=-1).astype(np.float32)
    results = []

    for feat in tqdm(features_raw, desc="Extracting HIRONEX features"):
        props = feat.get("properties", {})
        geom = shape(feat["geometry"])

        # Reproject geometry to TIF CRS if needed
        src_crs_str = gj.get("crs", {}).get("properties", {}).get("name", "")
        if src_crs_str and src_crs_str != geo_crs.to_string():
            geom = gpd.GeoSeries([geom], crs=src_crs_str).to_crs(geo_crs).iloc[0]

        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
            if isinstance(geom, MultiLineString):
                geom = max(geom.geoms, key=lambda g: g.length)
        if geom.geom_type != "LineString":
            continue

        gid = props.get("gid")
        if gid is None:
            continue

        # Pixel footprint for Var_line and Edge_contrast
        coords = list(geom.coords)
        pixel_coords = [src.index(x, y) for x, y in coords]
        pixel_coords = [(r, c) for r, c in pixel_coords
                        if 0 <= r < src.height and 0 <= c < src.width]
        if not pixel_coords:
            continue

        road_pixel_footprint: set = set()
        line_pixels = []
        for r, c in pixel_coords:
            for dr in range(-cfg.PIXEL_TOLERANCE, cfg.PIXEL_TOLERANCE + 1):
                for dc in range(-cfg.PIXEL_TOLERANCE, cfg.PIXEL_TOLERANCE + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < src.height and 0 <= cc < src.width:
                        line_pixels.append(rgb[int(rr), int(cc)])
                        road_pixel_footprint.add((int(rr), int(cc)))
        if not line_pixels:
            continue
        line_pixels = np.array(line_pixels)
        var_line = (
            np.mean(np.linalg.norm(np.diff(line_pixels, axis=0), axis=1))
            if len(line_pixels) > 1 else 0.0
        )

        neighbors = []
        outer_tol = cfg.PIXEL_TOLERANCE + 3
        for r, c in pixel_coords:
            for dr in range(-outer_tol, outer_tol + 1):
                for dc in range(-outer_tol, outer_tol + 1):
                    if abs(dr) <= cfg.PIXEL_TOLERANCE and abs(dc) <= cfg.PIXEL_TOLERANCE:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < src.height and 0 <= cc < src.width:
                        neighbors.append(rgb[int(rr), int(cc)])
        if not neighbors:
            continue
        neighbors = np.array(neighbors)
        edge_contrast = float(np.linalg.norm(line_pixels.mean(axis=0) - neighbors.mean(axis=0)))

        roi = compute_hironex_roi_for_line(
            geom, src, metric_crs,
            crosssect_dist=cfg.CROSSSECT_DIST,
            crosssect_length=cfg.CROSSSECT_LENGTH,
            samples_per_section=cfg.SAMPLES_PER_SECTION,
            target_band_length=cfg.TARGET_BAND_LENGTH,
            random_seed=cfg.RANDOM_SEED,
        )

        results.append({
            "gid": props.get("gid"),
            "osm_id": props.get("osm_id"),
            "code": props.get("code"),
            "fclass": props.get("fclass"),
            "name": props.get("name"),
            "pixel_count": len(road_pixel_footprint),
            "Var_line": float(var_line),
            "Edge_contrast": edge_contrast,
            "ROI": float(roi),
            "Road_detected": -1,
            "geometry": geom,
        })

    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("No features extracted. Check TIF/GeoJSON alignment.")

    df["ROI_norm"] = (df["ROI"] - df["ROI"].mean()) / (df["ROI"].std() + 1e-6)
    log.info("Feature extraction done: %d segments", len(df))
    return df


def predict(df_feat: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Apply a pre-trained Random Forest to df_feat and add a 'Pred' column.

    Pred == 1 → road existed in the historical period (remained).
    Pred == 0 → road did not exist (added after the historical period).
    """
    log.info("Loading model from %s", model_path)
    rf = load(model_path)
    df_feat = df_feat.copy()
    df_feat["Pred"] = rf.predict(df_feat[["Var_line", "Edge_contrast", "ROI_norm"]])
    n1 = int((df_feat["Pred"] == 1).sum())
    n0 = int((df_feat["Pred"] == 0).sum())
    log.info("RF prediction: %d remained (Pred=1), %d added (Pred=0)", n1, n0)
    return df_feat


def run(src: rasterio.DatasetReader, paths: dict) -> pd.DataFrame:
    """
    Run feature extraction + RF prediction for one map sheet.

    Returns df_feat as a GeoDataFrame with CRS set to the TIF's CRS.
    """
    geo_crs = src.crs
    df_feat = extract_features(src, paths["modern"], geo_crs, cfg.METRIC_EPSG)
    df_feat = predict(df_feat, paths["model"])
    df_feat = gpd.GeoDataFrame(df_feat, geometry="geometry", crs=geo_crs)
    return df_feat
