"""
road_extractor.py
-----------------
Stage 1 of the pipeline:
  1. Load the RGBA GeoTIFF.
  2. Match BGR road colours → binary masks per fclass.
  3. Skeletonise each mask.
  4. Convert skeleton pixels → polyline segments (split at junctions).
  5. Filter segments against a modern-road topology prior.
  6. (Optional) Centerlineise thick double-line road classes.
  7. Smooth & export as GeoJSON.
"""

import json
import numpy as np
import cv2
import rasterio
from rasterio.transform import xy as rio_xy
from skimage.morphology import skeletonize
from shapely.geometry import LineString, mapping
from shapely.strtree import STRtree
import geopandas as gpd

from logger import get_logger
import config as cfg

log = get_logger("melway.road_extractor")

# 8-connectivity neighbour offsets
_NEIGHBORS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


# ---------------------------------------------------------------------------
# Step 2 – colour masks
# ---------------------------------------------------------------------------

def build_colour_masks(img_bgr: np.ndarray, fclass_colors: dict,
                       tolerance: int = cfg.COLOR_TOLERANCE) -> dict:
    """Return a dict {fclass: bool skeleton array} for each road class."""
    masks = {}
    for cls, color_list in fclass_colors.items():
        mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        for bgr in color_list:
            diff = np.abs(img_bgr.astype(np.int16) - np.array(bgr, dtype=np.int16))
            mask[np.all(diff <= tolerance, axis=2)] = 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # Remove thin text artefacts from the trunk class by distance transform
        if cls == "trunk":
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
            mask[dist < 2.2] = 0

        masks[cls] = skeletonize(mask > 0)
        log.debug("  %s skeleton pixels: %d", cls, int(masks[cls].sum()))

    return masks


# ---------------------------------------------------------------------------
# Step 3 – skeleton → polyline segments
# ---------------------------------------------------------------------------

def skel_to_paths(skel_bool: np.ndarray) -> list:
    """
    Convert a boolean skeleton image to a list of pixel-coordinate paths.

    Each path is a list of (row, col) tuples, split at junction pixels
    (degree ≠ 2) so that every path is a simple chain between two nodes.
    """
    coords = np.argwhere(skel_bool)
    if coords.size == 0:
        return []

    pix_set = set(map(tuple, coords.tolist()))
    neighbors_map: dict = {}
    degree: dict = {}

    for r, c in pix_set:
        neighs = [(r + dr, c + dc) for dr, dc in _NEIGHBORS if (r + dr, c + dc) in pix_set]
        neighbors_map[(r, c)] = neighs
        degree[(r, c)] = len(neighs)

    nodes = {p for p, d in degree.items() if d != 2}

    visited_edges: set = set()
    segments: list = []

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    def walk_segment(start, nxt):
        path = [start, nxt]
        prev, curr = start, nxt
        while curr not in nodes:
            neighs = neighbors_map[curr]
            if not neighs:
                break
            new = neighs[0] if len(neighs) == 1 or neighs[1] == prev else neighs[1]
            path.append(new)
            prev, curr = curr, new
        return path

    for start in nodes:
        for nxt in neighbors_map[start]:
            ek = edge_key(start, nxt)
            if ek in visited_edges:
                continue
            seg = walk_segment(start, nxt)
            for i in range(len(seg) - 1):
                visited_edges.add(edge_key(seg[i], seg[i + 1]))
            if len(seg) > 1:
                segments.append(seg)

    # Handle pure loops (all pixels have degree 2)
    if not nodes:
        start = next(iter(pix_set))
        loop = [start]
        prev, curr = None, start
        while True:
            neighs = neighbors_map[curr]
            if not neighs:
                break
            nxt = neighs[0] if prev is None or len(neighs) == 1 else (neighs[0] if neighs[1] == prev else neighs[1])
            ek = edge_key(curr, nxt)
            if ek in visited_edges:
                break
            visited_edges.add(ek)
            loop.append(nxt)
            prev, curr = curr, nxt
            if curr == start:
                break
        if len(loop) > 1:
            segments.append(loop)

    return segments


# ---------------------------------------------------------------------------
# Step 4 – topology filter
# ---------------------------------------------------------------------------

def filter_by_modern_topology(class_paths: dict, transform, modern_gdf: gpd.GeoDataFrame,
                               src_crs, min_path_len_pix: int = cfg.MIN_PATH_LEN_PIX,
                               buffer_dist: float = cfg.MODERN_BUFFER_DIST,
                               min_overlap_ratio: float = 0.0) -> dict:
    """
    Keep only skeleton paths that spatially connect to a modern road network.
    """
    modern_geoms = [
        g for g in modern_gdf.geometry
        if g is not None and not g.is_empty and g.geom_type in ("LineString", "MultiLineString")
    ]
    tree = STRtree(modern_geoms)

    def _pix_to_xy(r, c):
        x, y = rio_xy(transform, r, c)
        return float(x), float(y)

    def _connects(path):
        if len(path) < min_path_len_pix:
            return False
        line = LineString([_pix_to_xy(r, c) for r, c in path])
        if line.is_empty or line.length == 0:
            return False
        buf = line.buffer(buffer_dist)
        idxs = tree.query(buf)
        if len(idxs) == 0:
            return False
        if min_overlap_ratio > 0:
            overlap = sum(
                modern_geoms[int(i)].intersection(buf).length
                for i in idxs
                if modern_geoms[int(i)].intersects(buf)
            )
            return (overlap / max(line.length, 1e-6)) >= min_overlap_ratio
        return any(modern_geoms[int(i)].intersects(buf) for i in idxs)

    filtered = {}
    for cls, paths in class_paths.items():
        kept = [p for p in paths if _connects(p)]
        log.info("  topology filter %s: %d → %d paths", cls, len(paths), len(kept))
        filtered[cls] = kept
    return filtered


# ---------------------------------------------------------------------------
# Step 5 – centerlineise double-line roads
# ---------------------------------------------------------------------------

def centerlineize_paths(paths: list, height: int, width: int,
                         thickness_px: int = 7, close_ks: int = 5) -> list:
    """
    Rasterise paths as thick lines, skeletonise, then re-extract paths.
    Collapses parallel double-lines (e.g. divided carriageways) to a single centreline.
    """
    if not paths:
        return []

    band = np.zeros((height, width), dtype=np.uint8)
    for path in paths:
        if len(path) < 2:
            continue
        pts = np.array([(c, r) for r, c in path], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(band, [pts], isClosed=False, color=255, thickness=int(thickness_px))

    k = int(close_ks) | 1  # ensure odd
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    skel = skeletonize(band > 0)
    return skel_to_paths(skel)


# ---------------------------------------------------------------------------
# Step 6 – GeoJSON export with smoothing
# ---------------------------------------------------------------------------

def _decimate(path: list, step: int = cfg.DECIMATE_STEP) -> list:
    if len(path) <= 2:
        return path
    keep = path[::step]
    if keep[-1] != path[-1]:
        keep.append(path[-1])
    return keep


def _chaikin(coords: list, n_iter: int = cfg.CHAIKIN_ITERS) -> list:
    if len(coords) < 3:
        return coords
    pts = coords
    for _ in range(n_iter):
        new_pts = [pts[0]]
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            new_pts.extend([
                (0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2),
                (0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2),
            ])
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


def export_geojson(class_paths: dict, transform, crs_string: str, dst_path: str,
                   decimate_step: int = cfg.DECIMATE_STEP,
                   simplify_tol: float = cfg.SIMPLIFY_TOL,
                   use_chaikin: bool = cfg.USE_CHAIKIN,
                   chaikin_iters: int = cfg.CHAIKIN_ITERS) -> int:
    """
    Convert pixel paths to smoothed geographic LineStrings and write GeoJSON.

    Returns
    -------
    Number of features exported.
    """
    def _pix_to_xy(r, c):
        x, y = rio_xy(transform, r, c)
        return float(x), float(y)

    features = []
    fid = 1

    for cls, paths in class_paths.items():
        for path in paths:
            if len(path) < 2:
                continue
            path2 = _decimate(path, step=decimate_step)
            coords = [_pix_to_xy(r, c) for r, c in path2]
            if len(coords) < 2:
                continue
            if use_chaikin:
                coords = _chaikin(coords, n_iter=chaikin_iters)
            geom = LineString(coords)
            if geom.is_empty or geom.length == 0:
                continue
            geom = geom.simplify(simplify_tol, preserve_topology=True)
            if geom.is_empty or geom.length == 0:
                continue
            features.append({
                "type": "Feature",
                "id": fid,
                "properties": {"gid": fid, "fclass": cls, "length": float(geom.length)},
                "geometry": mapping(geom),
            })
            fid += 1

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump({
            "type": "FeatureCollection",
            "name": "roads_extracted",
            "crs": {"type": "name", "properties": {"name": crs_string}} if crs_string else None,
            "features": features,
        }, f, ensure_ascii=False)

    log.info("Exported %d features → %s", len(features), dst_path)
    return len(features)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(map_id: str, year: str, paths: dict) -> None:
    """
    Full road extraction pipeline for one map sheet.

    Parameters
    ----------
    map_id : e.g. "007"
    year   : e.g. "2020"
    paths  : path dict from config.get_paths()
    """
    log.info("=== Road extraction  MAP=%s  YEAR=%s ===", map_id, year)

    # 1. Load RGBA TIF
    src = rasterio.open(paths["rgba_tif"])
    transform = src.transform
    width, height = src.width, src.height
    crs_string = src.crs.to_string() if src.crs else None
    img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    log.info("Loaded TIF: %dx%d  CRS=%s", width, height, crs_string)

    # 2. Colour masks + skeletons
    log.info("Building colour masks...")
    masks = build_colour_masks(img_bgr, cfg.FCLASS_COLORS)

    # 3. Skeleton → paths
    log.info("Extracting skeleton paths...")
    class_paths = {cls: skel_to_paths(skel) for cls, skel in masks.items()}
    for cls, p in class_paths.items():
        log.info("  %s: %d raw paths", cls, len(p))

    # 4. Topology filter (modern roads)
    log.info("Applying topology filter against modern roads...")
    modern_gdf = gpd.read_file(paths["modern"]).to_crs(src.crs)
    class_paths = filter_by_modern_topology(class_paths, transform, modern_gdf, src.crs)

    # 5. Centerlineise thick road classes
    log.info("Centerlineising double-line roads...")
    for cls in list(class_paths.keys()):
        if cls not in cfg.CENTERLINE_CLASSES:
            continue
        before = len(class_paths[cls])
        class_paths[cls] = centerlineize_paths(
            class_paths[cls], height, width,
            thickness_px=cfg.THICKNESS_BY_CLASS.get(cls, 7),
        )
        log.info("  %s centerline: %d → %d paths", cls, before, len(class_paths[cls]))

    # 6. Export GeoJSON
    dst_path = paths["historical_raw"]
    export_geojson(class_paths, transform, crs_string, dst_path)
    src.close()
