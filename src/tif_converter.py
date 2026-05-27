"""
tif_converter.py
----------------
Convert a palette (single-band, indexed-colour) GeoTIFF to a 4-band RGBA GeoTIFF.
This is always the first step before any image-processing.
"""

import numpy as np
import rasterio

from logger import get_logger

log = get_logger("melway.tif_converter")


def palette_to_rgba(src_path: str, dst_path: str) -> None:
    """
    Read a palette GeoTIFF and write an equivalent RGBA GeoTIFF.

    Parameters
    ----------
    src_path : path to the source indexed-colour GeoTIFF.
    dst_path : path where the RGBA output will be saved.
    """
    log.info("Converting palette TIFF → RGBA: %s", src_path)

    with rasterio.open(src_path) as src:
        band = src.read(1)
        cmap = src.colormap(1)
        h, w = band.shape
        rgba = np.zeros((4, h, w), dtype=np.uint8)

        for k, v in cmap.items():
            rgba[0, band == k] = v[0]  # R
            rgba[1, band == k] = v[1]  # G
            rgba[2, band == k] = v[2]  # B
            rgba[3, band == k] = 255 if len(v) < 4 else v[3]  # A

        profile = src.profile.copy()
        profile.update(count=4, dtype="uint8", photometric="RGB")

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(rgba)

    log.info("Palette TIFF converted → %s", dst_path)
