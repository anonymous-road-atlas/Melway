"""
road_comparator.py
------------------
Stage 3 of the pipeline:
  3.0  Load historical GeoJSON and unify CRS.
  3.1  Compare historical segments against the modern feature set via
       buffer-overlap ratio → assign status: "both" or "disappeared".
  3.2  Enrich "both" historical rows with matched df_feat attributes
       (Step 3.5 in the original notebook).
  4.0  Classify df_feat segments into remained / added categories.
  5.0  Visualise and export final GeoJSONs + overview PNG.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm

from logger import get_logger
import config as cfg

log = get_logger("melway.road_comparator")


# ---------------------------------------------------------------------------
# Step 3: historical vs modern comparison
# ---------------------------------------------------------------------------

def compare_historical_modern(df_feat: gpd.GeoDataFrame,
                               historical_path: str,
                               geo_crs,
                               metric_crs: int = cfg.METRIC_EPSG,
                               buffer_dist: float = cfg.BUFFER_DIST,
                               min_overlap_ratio: float = cfg.MIN_OVERLAP_RATIO,
                               target_classes: list = cfg.TARGET_CLASSES) -> gpd.GeoDataFrame:
    """
    Load historical roads and assign 'status' = "both" | "disappeared".

    Parameters
    ----------
    df_feat        : GeoDataFrame of modern extracted features (GEO_CRS).
    historical_path: Path to the raw historical GeoJSON (old_2).
    geo_crs        : TIF native CRS (geographic, degrees).
    """
    historical = gpd.read_file(historical_path)

    # Fix invalid CRS (LOCAL_CS written by some exporters)
    if historical.crs is None or "LOCAL_CS" in str(historical.crs):
        log.warning("Historical data has invalid CRS — forcing EPSG:7844")
        historical = historical.set_crs("EPSG:7844", allow_override=True)

    # Align CRS
    if historical.crs != geo_crs:
        historical = historical.to_crs(geo_crs)

    log.info("Historical before fclass filter: %d", len(historical))
    historical = historical[historical["fclass"].isin(target_classes)].copy()
    historical = historical[~historical.geometry.is_empty & historical.geometry.is_valid]
    log.info("Historical after fclass filter:  %d", len(historical))

    # Project both to metric CRS for buffer operations
    df_feat_metric = df_feat.to_crs(metric_crs)
    historical_metric = historical.to_crs(metric_crs)
    historical_metric["length_m"] = historical_metric.geometry.length

    status = []
    for _, row in tqdm(historical_metric.iterrows(),
                        desc="Comparing geometries (metric)",
                        total=len(historical_metric)):
        geom_m = row.geometry
        if geom_m.is_empty:
            status.append("unknown")
            continue

        buf = geom_m.buffer(buffer_dist)
        intersected = df_feat_metric[df_feat_metric.geometry.intersects(buf)]

        if not intersected.empty:
            geom_len = row["length_m"]
            overlap_len = intersected.intersection(buf).length.sum()
            ratio = 0.0 if geom_len < 0.01 else overlap_len / geom_len
            status.append("both" if ratio > min_overlap_ratio else "disappeared")
        else:
            status.append("disappeared")

    historical["status"] = status
    n_both = status.count("both")
    n_disp = status.count("disappeared")
    log.info("Comparison done — both: %d  disappeared: %d", n_both, n_disp)
    return historical


# ---------------------------------------------------------------------------
# Step 3.5: enrich "both" historical rows with df_feat attributes
# ---------------------------------------------------------------------------

def enrich_historical_with_features(historical: gpd.GeoDataFrame,
                                     df_feat: gpd.GeoDataFrame,
                                     metric_crs: int = cfg.METRIC_EPSG,
                                     buffer_dist: float = cfg.BUFFER_DIST) -> gpd.GeoDataFrame:
    """
    For each historical row with status="both", find the best-matching
    df_feat segment (highest overlap inside buffer) and merge its attributes.
    """
    log.info("Enriching 'both' historical rows with df_feat attributes...")

    df_feat_metric = df_feat.to_crs(metric_crs).copy()
    if "gid" in df_feat_metric.columns:
        df_feat_metric = df_feat_metric.rename(columns={"gid": "gid_df_feat"})

    historical_metric = historical.to_crs(metric_crs)
    sindex = df_feat_metric.sindex

    both_idx = historical.index[historical["status"] == "both"].tolist()
    matched_rows = []

    for idx in tqdm(both_idx, desc="Matching best df_feat for historical(both)"):
        geom_m = historical_metric.loc[idx, "geometry"]
        if geom_m is None or geom_m.is_empty:
            continue
        buf = geom_m.buffer(buffer_dist)
        cand_pos = list(sindex.intersection(buf.bounds))
        if not cand_pos:
            continue
        cands = df_feat_metric.iloc[cand_pos]
        cands = cands[cands.geometry.intersects(buf)]
        if cands.empty:
            continue
        overlap_len = cands.geometry.intersection(buf).length
        best_pos = int(np.argmax(overlap_len.values))
        best_row = cands.iloc[best_pos].copy()
        best_row["hist_index"] = idx
        best_row["overlap_m"] = float(overlap_len.iloc[best_pos])
        matched_rows.append(best_row)

    if not matched_rows:
        log.warning("No matches found to enrich 'both' features.")
        return historical

    matched_df = gpd.GeoDataFrame(matched_rows, geometry="geometry", crs=metric_crs)
    matched_df = matched_df.rename(columns={"geometry": "geometry_df_feat"})

    historical_enriched = historical.copy()
    historical_enriched["hist_index"] = historical_enriched.index
    historical_enriched = historical_enriched.merge(
        pd.DataFrame(matched_df.drop(columns="geometry_df_feat")),
        on="hist_index", how="left",
    ).drop(columns=["hist_index"])

    historical = gpd.GeoDataFrame(historical_enriched, geometry="geometry", crs=historical.crs)
    log.info("Enriched %d 'both' historical features.", len(matched_rows))
    return historical


# ---------------------------------------------------------------------------
# Step 4: classify df_feat into remained / added
# ---------------------------------------------------------------------------

def classify_changes(df_feat: pd.DataFrame,
                      historical: gpd.GeoDataFrame,
                      geo_crs) -> tuple:
    """
    Combine RF predictions and historical-match results into three categories:

    remained  : road existed both historically and in the modern data.
    added      : road appears only in modern data (not in historical).
    disappeared: road existed historically but is absent in modern data.

    Returns
    -------
    (remained_gdf, added_gdf, disappeared_gdf) — all in geo_crs.
    """
    both_df_feat_gids: set = set()
    if "gid_df_feat" in historical.columns:
        both_df_feat_gids = set(
            historical.loc[historical["status"] == "both", "gid_df_feat"]
            .dropna().astype(int).unique()
        )

    remained_pred = df_feat[df_feat["Pred"] == 1].copy()
    added_pred = df_feat[df_feat["Pred"] == 0].copy()

    # Correct Pred=0 rows that are actually matched historically
    added_is_actually_remained = added_pred[added_pred["gid"].isin(both_df_feat_gids)].copy()
    added_true = added_pred[~added_pred["gid"].isin(both_df_feat_gids)].copy()

    remained = pd.concat([remained_pred, added_is_actually_remained], ignore_index=True)
    remained = remained.drop_duplicates(subset=["gid"], keep="first")
    added = added_true.drop_duplicates(subset=["gid"], keep="first")
    disappeared = historical[historical["status"] == "disappeared"].copy()

    log.info(
        "Change classification — remained: %d  added: %d  disappeared: %d",
        len(remained), len(added), len(disappeared),
    )

    remained_gdf = gpd.GeoDataFrame(remained, geometry="geometry", crs=geo_crs)
    added_gdf = gpd.GeoDataFrame(added, geometry="geometry", crs=geo_crs)
    disappeared_gdf = gpd.GeoDataFrame(disappeared, geometry="geometry", crs=geo_crs)
    return remained_gdf, added_gdf, disappeared_gdf


# ---------------------------------------------------------------------------
# Step 5: visualise + export
# ---------------------------------------------------------------------------

def export_results(historical: gpd.GeoDataFrame,
                   remained_gdf: gpd.GeoDataFrame,
                   added_gdf: gpd.GeoDataFrame,
                   disappeared_gdf: gpd.GeoDataFrame,
                   outdir: str, map_id: str) -> str:
    """
    Save GeoJSONs (EPSG:4326) and an overview PNG.

    Returns the path to the overview PNG.
    """
    geo_crs = remained_gdf.crs

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    historical.plot(ax=ax, color="#aaaaaa", linewidth=0.4, label="Historical (All)")
    remained_gdf.plot(ax=ax, color="green", linewidth=1.3, label="Remained")
    added_gdf.plot(ax=ax, color="blue", linewidth=1.3, label="Added")
    disappeared_gdf.plot(ax=ax, color="red", linewidth=1.3, label="Disappeared")
    plt.legend()
    plt.title(f"Road Change Detection (Map {map_id}) — Remained / Added / Disappeared")
    plt.tight_layout()
    png_path = os.path.join(outdir, f"road_change_overview_m{map_id}.png")
    plt.savefig(png_path, dpi=300)
    plt.close(fig)

    historical.to_crs(4326).to_file(os.path.join(outdir, "historical_all.geojson"), driver="GeoJSON")
    remained_gdf.to_crs(4326).to_file(os.path.join(outdir, "remained.geojson"), driver="GeoJSON")
    added_gdf.to_crs(4326).to_file(os.path.join(outdir, "added.geojson"), driver="GeoJSON")
    disappeared_gdf.to_crs(4326).to_file(os.path.join(outdir, "disappeared.geojson"), driver="GeoJSON")

    log.info("Exported GeoJSONs to %s", outdir)
    return png_path


def build_and_export_historical_db(remained_gdf: gpd.GeoDataFrame,
                                    disappeared_gdf: gpd.GeoDataFrame,
                                    outdir: str) -> str:
    """
    Build historical_db.geojson = remained roads (modern geometry)
    + disappeared roads (historical geometry).
    """
    historical_db = pd.concat([remained_gdf, disappeared_gdf], ignore_index=True)
    historical_db = gpd.GeoDataFrame(historical_db, geometry="geometry", crs=remained_gdf.crs)
    out_path = os.path.join(outdir, "historical_db.geojson")
    historical_db.to_crs(4326).to_file(out_path, driver="GeoJSON")
    log.info("Historical DB saved: %s  (%d features)", out_path, len(historical_db))
    return out_path
