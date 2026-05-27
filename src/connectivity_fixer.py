"""
connectivity_fixer.py
---------------------
Post-process the RF predictions by examining topological connectivity:

Rule: if a contiguous group of Pred=0 segments is bridged between
      two *different* Pred=1 segments (one at each end), flip the
      whole group to Pred=1.

This handles roads that were split during skeletonisation and whose
fragments were individually scored low but are structurally connected.
"""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict, deque
from shapely.geometry import Point

from logger import get_logger
import config as cfg

log = get_logger("melway.connectivity_fixer")

EPSG_M = cfg.METRIC_EPSG
EPS_M = cfg.ENDPOINT_EPS_M
JOIN_M = cfg.JOIN_M
GEOM_COL = "geometry"
GRID_END = 0.01   # 1 cm bucket for endpoint de-dup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_crs(gdf: gpd.GeoDataFrame) -> str:
    minx, miny, maxx, maxy = gdf.total_bounds
    if (-180 <= minx <= 180) and (-90 <= miny <= 90) and (-180 <= maxx <= 180) and (-90 <= maxy <= 90):
        return "EPSG:4326"
    return f"EPSG:{EPSG_M}"


def _iter_lines(geom):
    if geom is None:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            if g.geom_type == "LineString":
                yield g


def _all_endpoints(geom) -> list:
    pts = []
    for part in _iter_lines(geom):
        coords = list(part.coords)
        if len(coords) >= 2:
            pts.append(Point(coords[0]))
            pts.append(Point(coords[-1]))
    return pts


def _bucket_key(pt: Point, grid: float = GRID_END):
    return (round(pt.x / grid) * grid, round(pt.y / grid) * grid)


def _pick_two_farthest(points: list):
    best_pair, best_d = None, -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = points[i].distance(points[j])
            if d > best_d:
                best_d = d
                best_pair = (points[i], points[j])
    return best_pair


def _component_endpoints(gdf_m: gpd.GeoDataFrame, comp_idxs: list,
                          grid: float = GRID_END) -> list:
    """Return external endpoints of a component (appear exactly once across all segments)."""
    cnt: dict = defaultdict(int)
    rep: dict = {}
    for idx in comp_idxs:
        for p in _all_endpoints(gdf_m.loc[idx].geometry):
            k = _bucket_key(p, grid)
            cnt[k] += 1
            rep[k] = p
    return [rep[k] for k, c in cnt.items() if c == 1]


def _pred1_hits(gdf_m: gpd.GeoDataFrame, sidx, pt: Optional[Point],
                eps_m: float = EPS_M) -> set:
    if pt is None:
        return set()
    buf = pt.buffer(eps_m)
    pos = list(sidx.intersection(buf.bounds))
    if not pos:
        return set()
    cand = gdf_m.iloc[pos]
    hit = cand[(cand["Pred"] == 1) & (cand.geometry.intersects(buf))]
    return set(hit.index.tolist())


def _build_pred0_components(gdf_m: gpd.GeoDataFrame, pred0_idxs: list,
                              join_m: float = JOIN_M) -> list:
    """Group Pred=0 segments into connected components based on endpoint proximity."""
    GRID_JOIN = join_m
    buckets: dict = defaultdict(list)
    for idx in pred0_idxs:
        for p in _all_endpoints(gdf_m.loc[idx].geometry):
            k = (round(p.x / GRID_JOIN) * GRID_JOIN, round(p.y / GRID_JOIN) * GRID_JOIN)
            buckets[k].append((idx, p))

    adj: dict = {idx: set() for idx in pred0_idxs}
    offsets = [-GRID_JOIN, 0.0, GRID_JOIN]
    for (bx, by), items in buckets.items():
        candidates = []
        for dx in offsets:
            for dy in offsets:
                candidates.extend(buckets.get((bx + dx, by + dy), []))
        for i, pi in items:
            for j, pj in candidates:
                if i != j and pi.distance(pj) <= join_m:
                    adj[i].add(j)
                    adj[j].add(i)

    visited: set = set()
    components = []
    for idx in pred0_idxs:
        if idx in visited:
            continue
        q = deque([idx])
        visited.add(idx)
        comp = [idx]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                    comp.append(v)
        components.append(comp)
    return components


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fix_connectivity(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Promote Pred=0 segments to Pred=1 when they are topologically bridged
    between two different Pred=1 segments.

    Parameters
    ----------
    df_feat : GeoDataFrame (or DataFrame with geometry column) with 'Pred' and 'osm_id'.

    Returns
    -------
    Updated copy of df_feat.
    """
    gdf = gpd.GeoDataFrame(df_feat.copy(), geometry=GEOM_COL)
    gdf["osm_id"] = gdf["osm_id"].astype(str)
    gdf["Pred"] = gdf["Pred"].astype(int)

    if gdf.crs is None:
        gdf = gdf.set_crs(_infer_crs(gdf))

    gdf_m = gdf.to_crs(EPSG_M).reset_index(drop=True)
    sidx_all = gdf_m.sindex

    pred0_idxs = gdf_m.index[gdf_m["Pred"] == 0].tolist()
    n0, n1 = len(pred0_idxs), int((gdf_m["Pred"] == 1).sum())
    log.info("Connectivity fixer: Pred=0=%d  Pred=1=%d", n0, n1)

    if n0 == 0 or n1 == 0:
        log.info("Nothing to fix.")
        return df_feat

    components = _build_pred0_components(gdf_m, pred0_idxs, join_m=JOIN_M)
    log.info("Pred=0 components: %d", len(components))

    promote_osm_ids: set = set()
    promoted_components = 0

    for comp in components:
        ends = _component_endpoints(gdf_m, comp, grid=GRID_END)
        if len(ends) < 2:
            continue
        endA, endB = _pick_two_farthest(ends) if len(ends) > 2 else (ends[0], ends[1])
        A1 = _pred1_hits(gdf_m, sidx_all, endA)
        B1 = _pred1_hits(gdf_m, sidx_all, endB)
        if not A1 or not B1:
            continue
        # Both ends must touch at least two different Pred=1 lines in total
        if len(A1 | B1) < 2:
            continue
        promote_osm_ids.update(gdf_m.loc[comp, "osm_id"].astype(str).tolist())
        promoted_components += 1

    log.info("Promoted components: %d  (%d osm_ids)", promoted_components, len(promote_osm_ids))

    df = df_feat.copy()
    df["osm_id"] = df["osm_id"].astype(str)
    df["Pred"] = df["Pred"].astype(int)
    mask = df["osm_id"].isin(promote_osm_ids)
    to_flip = int(((df["Pred"] == 0) & mask).sum())
    df.loc[mask & (df["Pred"] == 0), "Pred"] = 1
    log.info("Flipped Pred 0→1: %d rows", to_flip)
    return df
