"""
loop_detector.py
----------------
Detect circular loops in the modern road network and propagate their
existence label: if a loop (or any touching line) is predicted as
non-existent (Pred=0), the entire loop group is forced to Pred=1.
"""

import math
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge

from logger import get_logger
import config as cfg

log = get_logger("melway.loop_detector")

PROJECTED_EPSG = cfg.METRIC_EPSG
CLOSE_EPS_M = cfg.CLOSE_EPS_M
CANDIDATE_BUFFER_M = cfg.CANDIDATE_BUFFER_M
RADIAL_CV_MAX = cfg.RADIAL_CV_MAX
ISO_CIRC_MIN = cfg.ISO_CIRC_MIN


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iter_lines(geom):
    if geom is None:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            if g.geom_type == "LineString":
                yield g


def _collect_coords(geom) -> Optional[np.ndarray]:
    coords = []
    for ls in _iter_lines(geom):
        coords.extend(list(ls.coords))
    if not coords:
        return None
    return np.asarray(coords, dtype=float)


def is_closed_linestring(ls: LineString, eps_m: float = CLOSE_EPS_M) -> bool:
    coords = list(ls.coords)
    if len(coords) < 4:
        return False
    return Point(coords[0]).distance(Point(coords[-1])) <= eps_m


def geom_is_loop(geom, eps_m: float = CLOSE_EPS_M) -> bool:
    return any(is_closed_linestring(ls, eps_m) for ls in _iter_lines(geom))


def _make_polygon_from_loop(loop_geom) -> Optional[Polygon]:
    geom = loop_geom
    if geom.geom_type == "MultiLineString":
        try:
            merged = linemerge(geom)
        except Exception:
            merged = geom
        geom = merged if merged.geom_type == "LineString" else None
    if geom is None or geom.geom_type != "LineString":
        return None
    coords = list(geom.coords)
    if len(coords) < 4 or Point(coords[0]).distance(Point(coords[-1])) > CLOSE_EPS_M:
        return None
    try:
        poly = Polygon(coords)
        return poly if poly.is_valid and poly.area > 0 else None
    except Exception:
        return None


def radial_cv(loop_geom) -> Optional[float]:
    arr = _collect_coords(loop_geom)
    if arr is None or len(arr) < 10:
        return None
    center = arr.mean(axis=0)
    r = np.linalg.norm(arr - center, axis=1)
    mu = float(r.mean())
    return float(r.std() / mu) if mu != 0 else None


def iso_circularity(loop_geom) -> Optional[float]:
    poly = _make_polygon_from_loop(loop_geom)
    if poly is None:
        return None
    A, P = float(poly.area), float(poly.length)
    return float(4 * math.pi * A / (P * P)) if P != 0 else None


def endpoints_touch_loop(line_geom, loop_boundary, eps_m: float = CLOSE_EPS_M):
    for ls in _iter_lines(line_geom):
        coords = list(ls.coords)
        if len(coords) < 2:
            continue
        for pt in (Point(coords[0]), Point(coords[-1])):
            if pt.distance(loop_boundary) <= eps_m:
                return True, pt
    return False, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_circular_loops(modern_gdf: gpd.GeoDataFrame):
    """
    Find circular loops in *modern_gdf* (projected to metric CRS internally).

    Returns
    -------
    loops_df  : DataFrame of circular loops with metrics.
    touch_df  : DataFrame of lines that touch those loops.
    """
    gdf = modern_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf_m = gdf.to_crs(PROJECTED_EPSG)

    loop_mask = gdf_m.geometry.apply(lambda g: geom_is_loop(g, eps_m=CLOSE_EPS_M))
    loops_all = gdf_m[loop_mask].copy()

    if loops_all.empty:
        log.info("No loops found in modern road network.")
        return _empty_loops_df(), _empty_touch_df()

    loops_all["loop_length_m"] = loops_all.geometry.length
    loops_all["radial_cv"] = loops_all.geometry.apply(radial_cv)
    loops_all["iso_circ"] = loops_all.geometry.apply(iso_circularity)
    loops_all["is_circular"] = (
        (loops_all["radial_cv"].notna() & (loops_all["radial_cv"] <= RADIAL_CV_MAX))
        | (loops_all["iso_circ"].notna() & (loops_all["iso_circ"] >= ISO_CIRC_MIN))
    )
    loops = loops_all[loops_all["is_circular"]].copy()
    log.info("Loops found: %d total | %d circular", len(loops_all), len(loops))

    if loops.empty:
        return _empty_loops_df(), _empty_touch_df()

    sidx = gdf_m.sindex
    loop_rows, touch_rows = [], []

    for loop_idx, loop_row in loops.iterrows():
        loop_geom = loop_row.geometry
        loop_boundary = loop_geom.boundary
        bbox = loop_geom.buffer(CANDIDATE_BUFFER_M).bounds
        cands = gdf_m.iloc[list(sidx.intersection(bbox))]
        cands = cands[cands.index != loop_idx]

        endpoint_hits, any_hits = [], []
        for tidx, trow in cands.iterrows():
            hit, _ = endpoints_touch_loop(trow.geometry, loop_boundary, eps_m=CLOSE_EPS_M)
            if hit:
                endpoint_hits.append(tidx)
                touch_rows.append({
                    "loop_index": loop_idx,
                    "touch_index": tidx,
                    "loop_osm_id": loop_row.get("osm_id"),
                    "touch_osm_id": trow.get("osm_id"),
                    "touch_gid": trow.get("gid"),
                    "touch_fclass": trow.get("fclass"),
                    "mode": "endpoint",
                })
            elif trow.geometry.intersects(loop_geom):
                any_hits.append(tidx)

        ep_set = set(endpoint_hits)
        for tidx in any_hits:
            if tidx in ep_set:
                continue
            trow = cands.loc[tidx]
            touch_rows.append({
                "loop_index": loop_idx,
                "touch_index": tidx,
                "loop_osm_id": loop_row.get("osm_id"),
                "touch_osm_id": trow.get("osm_id"),
                "touch_gid": trow.get("gid"),
                "touch_fclass": trow.get("fclass"),
                "mode": "any_intersection_only",
            })

        loop_rows.append({
            "loop_index": loop_idx,
            "gid": loop_row.get("gid"),
            "osm_id": loop_row.get("osm_id"),
            "fclass": loop_row.get("fclass"),
            "loop_length_m": float(loop_row["loop_length_m"]),
            "radial_cv": loop_row.get("radial_cv"),
            "iso_circ": loop_row.get("iso_circ"),
            "n_touch_endpoint": len(endpoint_hits),
            "n_touch_any_intersection": len(any_hits),
        })

    loops_df = pd.DataFrame(loop_rows).sort_values(
        ["n_touch_endpoint", "loop_length_m"], ascending=[False, False]
    )
    touch_df = pd.DataFrame(touch_rows).sort_values(["loop_index", "mode"])
    return loops_df, touch_df


def apply_loop_label_propagation(df_feat: pd.DataFrame,
                                  loops_df: pd.DataFrame,
                                  touch_df: pd.DataFrame) -> pd.DataFrame:
    """
    If any segment in a circular-loop group is Pred=0, force the whole group to Pred=1.
    This corrects cases where a roundabout is partly detected as non-existing.
    """
    if loops_df.empty or touch_df.empty:
        return df_feat

    df = df_feat.copy()
    df["osm_id"] = df["osm_id"].astype(str)
    loops_df = loops_df.copy()
    loops_df["osm_id"] = loops_df["osm_id"].astype(str)
    touch_df = touch_df.copy()
    touch_df["loop_osm_id"] = touch_df["loop_osm_id"].astype(str)
    touch_df["touch_osm_id"] = touch_df["touch_osm_id"].astype(str)

    loop_to_lines: dict = defaultdict(set)
    for _, row in touch_df.iterrows():
        loop_to_lines[row["loop_osm_id"]].add(row["touch_osm_id"])

    zero_osm_ids: set = set()
    for loop_osm_id in loops_df["osm_id"]:
        related = set(loop_to_lines.get(loop_osm_id, set()))
        related.add(loop_osm_id)
        preds = df.loc[df["osm_id"].isin(related), "Pred"]
        if (preds == 0).any():
            zero_osm_ids.update(related)

    df.loc[df["osm_id"].isin(zero_osm_ids), "Pred"] = 1
    log.info("Loop label propagation: %d osm_ids forced to Pred=1", len(zero_osm_ids))
    return df


def _empty_loops_df():
    return pd.DataFrame(columns=[
        "loop_index", "gid", "osm_id", "fclass", "loop_length_m",
        "radial_cv", "iso_circ", "n_touch_endpoint", "n_touch_any_intersection",
    ])


def _empty_touch_df():
    return pd.DataFrame(columns=[
        "loop_index", "touch_index", "loop_osm_id", "touch_osm_id",
        "touch_gid", "touch_fclass", "mode",
    ])
