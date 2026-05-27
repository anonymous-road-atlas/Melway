"""
utils.py
--------
Shared helper functions used across multiple pipeline stages.
"""

import math
import numpy as np
import geopandas as gpd
from rasterio.transform import xy as rio_xy
from rasterio.warp import transform as s_transform


def to_unit_vec(vx: float, vy: float):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n != 0 else (0.0, 0.0)


def pix_to_xy(transform, r: int, c: int):
    x, y = rio_xy(transform, r, c)
    return float(x), float(y)


def sample_raster_rgb(src, xy_list: list) -> np.ndarray:
    """Sample RGB values at a list of (x, y) map coordinates."""
    data = []
    for xy in xy_list:
        try:
            val = list(src.sample([xy]))[0][:3]
            if not np.any(np.isnan(val)):
                data.append(val)
        except Exception:
            continue
    return np.array(data)


def compute_roi_from_band(rgb_band_3d: np.ndarray, target_band_length: int,
                          random_seed: int = 42) -> float:
    """Compute the ROI feature value from a 3-D cross-section pixel band."""
    if rgb_band_3d.size == 0:
        return 0.0
    gray = np.mean(rgb_band_3d, axis=2)
    H, _ = gray.shape
    if H > target_band_length:
        rng = np.random.default_rng(random_seed)
        idx = np.sort(rng.choice(H, target_band_length, replace=False))
        gray = gray[idx, :]
    lr_grad2 = np.abs(np.diff(gray, axis=1, n=2))
    crosssum = np.sum(lr_grad2, axis=0)
    xs = np.arange(crosssum.shape[0], dtype=float)
    return float(np.trapz(crosssum, xs))


def compute_hironex_roi_for_line(geom, src, metric_crs: int,
                                 crosssect_dist: int, crosssect_length: int,
                                 samples_per_section: int, target_band_length: int,
                                 random_seed: int = 42) -> float:
    """
    Compute the HIRONEX ROI feature for a single road LineString.

    Parameters
    ----------
    geom        : Shapely geometry in the TIF's native CRS (GEO_CRS).
    src         : Open rasterio dataset.
    metric_crs  : EPSG integer for the metric projection (metres).
    """
    try:
        geom_m = gpd.GeoSeries([geom], crs=src.crs).to_crs(metric_crs).iloc[0]
    except Exception:
        return 0.0

    if geom_m.is_empty or geom_m.length <= 0:
        return 0.0

    L = geom_m.length
    ds = np.arange(0, L, crosssect_dist)
    if len(ds) == 0 or ds[-1] != L:
        ds = np.append(ds, L)

    half_len = crosssect_length / 2.0
    offsets = np.linspace(-half_len, half_len, samples_per_section)
    rgb_rows = []

    for d in ds:
        P = geom_m.interpolate(d)
        d_eps = min(1.0, L * 0.001)
        P1 = geom_m.interpolate(max(0, d - d_eps))
        P2 = geom_m.interpolate(min(L, d + d_eps))
        tx, ty = to_unit_vec(P2.x - P1.x, P2.y - P1.y)
        nx, ny = -ty, tx

        xy_metric = [(P.x + off * nx, P.y + off * ny) for off in offsets]
        if not xy_metric:
            continue
        x_m, y_m = zip(*xy_metric)

        try:
            x_g, y_g = s_transform(metric_crs, src.crs, x_m, y_m)
            xy_geo = list(zip(x_g, y_g))
        except Exception:
            continue

        rgb_band = sample_raster_rgb(src, xy_geo)
        if rgb_band.size > 0:
            rgb_rows.append(rgb_band)

    if not rgb_rows:
        return 0.0
    return compute_roi_from_band(np.stack(rgb_rows, axis=0), target_band_length,
                                 random_seed=random_seed)
