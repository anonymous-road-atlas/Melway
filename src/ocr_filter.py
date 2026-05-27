"""
ocr_filter.py
-------------
Remove "disappeared" historical segments that are likely text artefacts:
  1. Run EasyOCR on the RGBA TIF.
  2. Build a union polygon of all detected text regions.
  3. Remove any disappeared segment whose overlap with the text mask
     exceeds OCR_RATIO_TH (default 40 %).
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import Polygon
from shapely.ops import unary_union

from logger import get_logger
import config as cfg

log = get_logger("melway.ocr_filter")


def _overlap_ratio(line, mask_poly) -> float:
    if line is None or line.is_empty or line.length == 0:
        return 0.0
    inter = line.intersection(mask_poly)
    if inter.is_empty:
        return 0.0
    return float(inter.length) / float(line.length)


def remove_text_artefacts(historical: gpd.GeoDataFrame,
                           rgba_tif_path: str,
                           conf_th: float = cfg.OCR_CONF_TH,
                           text_buf_pix: float = cfg.TEXT_BUF_PIX,
                           ratio_th: float = cfg.OCR_RATIO_TH,
                           debug_png: str = "ocr_text_hits_debug.png") -> gpd.GeoDataFrame:
    """
    Filter out disappeared historical segments that substantially overlap
    with OCR-detected text polygons.

    Parameters
    ----------
    historical    : GeoDataFrame with a 'status' column.
    rgba_tif_path : Path to the RGBA GeoTIFF for this map sheet.
    conf_th       : Minimum OCR confidence (0–1).
    ratio_th      : Overlap-ratio threshold above which a segment is removed.
    debug_png     : Output path for the debug visualisation.

    Returns
    -------
    Filtered GeoDataFrame (in-place copy, same CRS as input).
    """
    try:
        import easyocr
    except ImportError:
        log.warning("easyocr not installed — skipping OCR filter.")
        return historical

    log.info("Running OCR on %s ...", rgba_tif_path)
    reader = easyocr.Reader(["en"], gpu=False)

    with rasterio.open(rgba_tif_path) as src_ocr:
        img_rgb = np.transpose(src_ocr.read([1, 2, 3]), (1, 2, 0))
        ocr_results = reader.readtext(img_rgb, detail=1, paragraph=False)

        if historical.crs is None:
            historical = historical.set_crs(src_ocr.crs)
        elif historical.crs != src_ocr.crs:
            historical = historical.to_crs(src_ocr.crs)

        pixel_size = max(abs(src_ocr.transform.a), abs(src_ocr.transform.e))
        left, bottom, right, top = src_ocr.bounds

    rows = []
    for res in ocr_results:
        if len(res) == 3:
            bbox, text, conf = res
        elif len(res) == 2:
            bbox, text = res
            conf = 1.0
        else:
            continue
        if conf < conf_th:
            continue
        pts_geo = [src_ocr.transform * (float(x), float(y)) for x, y in bbox]
        poly = Polygon(pts_geo)
        if poly.is_empty or not poly.is_valid:
            continue
        rows.append({"text": text, "conf": float(conf),
                      "geometry": poly.buffer(pixel_size * text_buf_pix)})

    if not rows:
        log.info("No OCR polygons detected — OCR filter skipped.")
        return historical

    text_gdf = gpd.GeoDataFrame(rows, crs=historical.crs)
    text_union = unary_union(text_gdf.geometry.values)
    log.info("OCR polygons detected: %d", len(text_gdf))

    disappeared = historical[historical["status"] == "disappeared"].copy()
    if len(disappeared) == 0:
        return historical

    disappeared["text_ratio"] = disappeared.geometry.apply(
        lambda g: _overlap_ratio(g, text_union)
    )
    hit_line_gdf = disappeared[disappeared["text_ratio"] >= ratio_th].copy()
    removed_count = len(hit_line_gdf)
    log.info(
        "Disappeared total: %d  removed by OCR ratio>=%.2f: %d",
        len(disappeared), ratio_th, removed_count,
    )

    # Debug visualisation
    with rasterio.open(rgba_tif_path) as src_ocr:
        img_rgb = np.transpose(src_ocr.read([1, 2, 3]), (1, 2, 0))
        left, bottom, right, top = src_ocr.bounds

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_rgb, extent=(left, right, bottom, top))
    ax.set_title(f"OCR text polygons & removed disappeared lines | ratio≥{ratio_th}")
    text_gdf.boundary.plot(ax=ax, linewidth=1.2, color="orange")
    if not hit_line_gdf.empty:
        hit_line_gdf.plot(ax=ax, linewidth=2.5, color="red")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(debug_png, dpi=200)
    plt.close(fig)
    log.info("OCR debug plot saved: %s", debug_png)

    if not hit_line_gdf.empty:
        historical = historical.drop(index=hit_line_gdf.index).reset_index(drop=True)
        log.info("Removed %d disappeared segments via OCR text mask.", removed_count)

    return historical
