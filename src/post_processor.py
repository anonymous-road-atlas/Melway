"""
post_processor.py
-----------------
Clean-up passes applied to historical_db before final export:

  A. Drop disappeared segments that are shorter and parallel to a
     non-disappeared counterpart (likely duplicate extraction artefacts).

  B. Remove disappeared trunk/primary/tertiary/residential segments
     whose nearest non-disappeared neighbour is a residential road
     (indicates mis-classified motorway-service ramp artefacts).
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.strtree import STRtree

from logger import get_logger
import config as cfg

log = get_logger("melway.post_processor")

EPSG_M = cfg.METRIC_EPSG
GEOM_COL = "geometry"


# ---------------------------------------------------------------------------
# A. Parallel + shorter duplicate filter
# ---------------------------------------------------------------------------

def _dominant_direction(line):
    coords = np.asarray(line.coords, dtype=float)
    if coords.shape[0] < 2:
        return None
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _parallel_angle_deg(l1, l2) -> float:
    v1 = _dominant_direction(l1)
    v2 = _dominant_direction(l2)
    if v1 is None or v2 is None:
        return 90.0
    cos = max(-1.0, min(1.0, abs(float(np.dot(v1, v2)))))
    return math.degrees(math.acos(cos))


def drop_parallel_shorter_disappeared(df: pd.DataFrame,
                                       status_col: str = "status",
                                       disappeared_value: str = "disappeared",
                                       angle_th_deg: float = cfg.PARALLEL_ANGLE_TH_DEG
                                       ) -> pd.DataFrame:
    """
    Mark disappeared segments as 'to_drop_text_fp=True' if they are parallel
    (angle ≤ angle_th_deg) and shorter than their nearest non-disappeared neighbour.
    """
    df = df.copy()
    disp_idx = df.index[df[status_col] == disappeared_value].tolist()
    other_df = df[df[status_col] != disappeared_value]

    if not disp_idx or other_df.empty:
        df["to_drop_text_fp"] = False
        return df

    other_geoms = list(other_df.geometry.values)
    tree = STRtree(other_geoms)
    to_drop: set = set()

    for idx in disp_idx:
        g = df.at[idx, GEOM_COL]
        if g is None or g.is_empty:
            continue
        nearest = tree.nearest(g)
        ng = other_geoms[int(nearest)] if isinstance(nearest, (int, np.integer)) else nearest
        if not hasattr(ng, "coords"):
            continue
        if _parallel_angle_deg(g, ng) <= angle_th_deg and float(g.length) < float(ng.length):
            to_drop.add(idx)

    df["to_drop_text_fp"] = df.index.isin(to_drop)
    log.info("Parallel+shorter filter: marked %d segments for removal.", len(to_drop))
    return df


# ---------------------------------------------------------------------------
# B. Remove disappeared trunk near residential
# ---------------------------------------------------------------------------

def remove_disappeared_near_residential(gdf: gpd.GeoDataFrame,
                                         remove_fclasses: set = cfg.REMOVE_FCLASSES_NEAR_RESIDENTIAL
                                         ) -> tuple:
    """
    Remove disappeared segments of certain road classes whose nearest
    non-disappeared neighbour is a residential road.

    Returns
    -------
    (filtered_gdf, removed_gdf)
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry=GEOM_COL)

    gdf = gdf.copy()
    for col in ("status", "fclass_x", "fclass"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).str.strip().str.lower()

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS.")

    gdf_m = gdf.to_crs(EPSG_M)
    dis = gdf_m[gdf_m["status"] == "disappeared"].copy()
    non = gdf_m[gdf_m["status"] != "disappeared"].copy()
    log.info("Disappeared: %d  Non-disappeared: %d", len(dis), len(non))

    if len(dis) == 0 or len(non) == 0:
        return gdf, gdf.iloc[0:0].copy()

    dis_r = dis.reset_index().rename(columns={"index": "orig_idx"})
    non_r = non.reset_index().rename(columns={"index": "orig_idx_non"})
    non_r2 = non_r[["orig_idx_non", "fclass", GEOM_COL]].rename(columns={"fclass": "nearest_fclass"})

    joined = gpd.sjoin_nearest(dis_r, non_r2, how="left", distance_col="nearest_dist_m")
    fclass_col = "fclass_x" if "fclass_x" in joined.columns else "fclass"
    remove_mask = (
        joined[fclass_col].isin(remove_fclasses)
        & (joined["nearest_fclass"] == "residential")
    )
    removed = joined[remove_mask].copy()
    removed_orig = set(removed["orig_idx"].tolist())
    log.info("Near-residential filter: removed %d segments.", len(removed))

    filtered = gdf.drop(index=list(removed_orig))
    removed_gdf = gdf.loc[list(removed_orig)].copy()
    return filtered, removed_gdf


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def clean_historical_db(historical_db: pd.DataFrame, geo_crs) -> gpd.GeoDataFrame:
    """
    Apply both post-processing passes and return the cleaned GeoDataFrame.
    """
    # A. Parallel + shorter duplicate filter
    marked = drop_parallel_shorter_disappeared(historical_db, angle_th_deg=cfg.PARALLEL_ANGLE_TH_DEG)
    historical_db = marked[~marked["to_drop_text_fp"]].copy()

    if not isinstance(historical_db, gpd.GeoDataFrame):
        historical_db = gpd.GeoDataFrame(historical_db, geometry=GEOM_COL, crs=geo_crs)
    elif historical_db.crs is None:
        historical_db = historical_db.set_crs(geo_crs)

    # B. Trunk/primary/etc near residential filter
    historical_db_clean, removed = remove_disappeared_near_residential(historical_db)

    # Remove by gid from original df for consistency
    if not removed.empty and "gid" in removed.columns:
        remove_gids = set(removed["gid"].tolist())
        historical_db_clean = historical_db_clean[
            ~historical_db_clean["gid"].isin(remove_gids)
        ].copy()
        log.info("After gid removal, rows remaining: %d", len(historical_db_clean))

    if not isinstance(historical_db_clean, gpd.GeoDataFrame):
        historical_db_clean = gpd.GeoDataFrame(historical_db_clean, geometry=GEOM_COL, crs=geo_crs)
    elif historical_db_clean.crs is None:
        historical_db_clean = historical_db_clean.set_crs(geo_crs)

    return historical_db_clean
