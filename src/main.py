"""
main.py
-------
Entry point for the Melway historical road change detection pipeline.

Usage
-----
    python main.py --map_id 007 --year 2020

Steps executed
--------------
  1.  Convert palette GeoTIFF → RGBA GeoTIFF.
  2.  Extract historical road lines via colour matching + skeletonisation.
  3.  Compute HIRONEX features on modern road segments + RF prediction.
  4.  Fix loop labels & connectivity (topology post-processing).
  5.  Compare historical vs modern → assign status (both / disappeared).
  6.  Enrich "both" rows with modern feature attributes.
  7.  OCR filter: remove text artefacts from disappeared roads.
  8.  Build & post-process historical_db (parallel/residential filter).
  9.  Export final GeoJSONs + overview PNG.
"""

import os
import sys
import argparse
import time
import rasterio
import geopandas as gpd

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from logger import get_logger
from tif_converter import palette_to_rgba
from road_extractor import run as run_road_extraction
from feature_extractor import run as run_feature_extraction
from loop_detector import detect_circular_loops, apply_loop_label_propagation
from connectivity_fixer import fix_connectivity
from road_comparator import (
    compare_historical_modern,
    enrich_historical_with_features,
    classify_changes,
    export_results,
    build_and_export_historical_db,
)
from ocr_filter import remove_text_artefacts
from post_processor import clean_historical_db


def parse_args():
    parser = argparse.ArgumentParser(
        description="Melway historical road change detection pipeline."
    )
    parser.add_argument("--map_id", default=cfg.MAP_ID,
                        help="Map sheet ID, zero-padded (default: %(default)s)")
    parser.add_argument("--year", default=cfg.YEAR,
                        help="Year to process (default: %(default)s)")
    parser.add_argument("--next_year", default=cfg.NEXT_YEAR,
                        help="Next year for chained processing (default: %(default)s)")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip road extraction (use existing historical GeoJSON)")
    parser.add_argument("--log_dir", default=os.path.join(cfg.PROJECT_ROOT, "logs"),
                        help="Directory for log files (default: <project>/logs)")
    return parser.parse_args()


def main():
    args = parse_args()
    log = get_logger("melway.main", log_dir=args.log_dir)
    paths = cfg.get_paths(args.map_id, args.year, args.next_year)

    log.info("=" * 60)
    log.info("Melway Road Change Detection Pipeline")
    log.info("MAP_ID=%s  YEAR=%s  NEXT_YEAR=%s", args.map_id, args.year, args.next_year)
    log.info("=" * 60)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Palette TIF → RGBA TIF
    # ------------------------------------------------------------------
    log.info("[Step 1] Converting palette TIF to RGBA...")
    palette_to_rgba(paths["src_tif"], paths["rgba_tif"])

    # ------------------------------------------------------------------
    # Step 2: Road extraction (colour mask → skeleton → GeoJSON)
    # ------------------------------------------------------------------
    if not args.skip_extraction:
        log.info("[Step 2] Extracting historical road lines...")
        run_road_extraction(args.map_id, args.year, paths)
    else:
        log.info("[Step 2] Skipping road extraction (--skip_extraction set).")

    # ------------------------------------------------------------------
    # Step 3: Feature extraction + RF prediction
    # ------------------------------------------------------------------
    log.info("[Step 3] Extracting HIRONEX features and running RF model...")
    with rasterio.open(paths["rgba_tif"]) as src:
        df_feat = run_feature_extraction(src, paths)
        geo_crs = src.crs

    # ------------------------------------------------------------------
    # Step 4a: Loop label propagation
    # ------------------------------------------------------------------
    log.info("[Step 4a] Detecting circular loops and propagating labels...")
    modern_gdf = gpd.read_file(paths["modern"]).to_crs(geo_crs)
    loops_df, touch_df = detect_circular_loops(modern_gdf)
    df_feat = apply_loop_label_propagation(df_feat, loops_df, touch_df)

    # ------------------------------------------------------------------
    # Step 4b: Connectivity-based Pred 0→1 promotion
    # ------------------------------------------------------------------
    log.info("[Step 4b] Fixing predictions via connectivity analysis...")
    df_feat = fix_connectivity(df_feat)

    # ------------------------------------------------------------------
    # Step 5: Historical vs modern comparison
    # ------------------------------------------------------------------
    log.info("[Step 5] Comparing historical vs modern roads...")
    historical = compare_historical_modern(
        df_feat, paths["historical_raw"], geo_crs,
        metric_crs=cfg.METRIC_EPSG,
        buffer_dist=cfg.BUFFER_DIST,
        min_overlap_ratio=cfg.MIN_OVERLAP_RATIO,
    )

    # ------------------------------------------------------------------
    # Step 5.5: Enrich "both" rows with df_feat attributes
    # ------------------------------------------------------------------
    log.info("[Step 5.5] Enriching 'both' historical rows with feature attributes...")
    historical = enrich_historical_with_features(
        historical, df_feat,
        metric_crs=cfg.METRIC_EPSG,
        buffer_dist=cfg.BUFFER_DIST,
    )

    # ------------------------------------------------------------------
    # Step 6: OCR filter
    # ------------------------------------------------------------------
    log.info("[Step 6] Removing text artefacts from disappeared roads (OCR)...")
    ocr_debug_png = os.path.join(paths["outdir"], "ocr_text_hits_debug.png")
    historical = remove_text_artefacts(
        historical, paths["rgba_tif"],
        debug_png=ocr_debug_png,
    )

    # ------------------------------------------------------------------
    # Step 7: Change classification + initial export
    # ------------------------------------------------------------------
    log.info("[Step 7] Classifying road changes and exporting results...")
    remained_gdf, added_gdf, disappeared_gdf = classify_changes(df_feat, historical, geo_crs)
    overview_png = export_results(historical, remained_gdf, added_gdf, disappeared_gdf,
                                   paths["outdir"], args.map_id)
    log.info("Overview plot saved: %s", overview_png)

    # ------------------------------------------------------------------
    # Step 8: Build and post-process historical_db
    # ------------------------------------------------------------------
    log.info("[Step 8] Building and cleaning historical DB...")
    import pandas as pd
    # Normalise both GDFs to the same CRS before concat to avoid minor label mismatches
    disappeared_gdf = disappeared_gdf.to_crs(geo_crs)
    remained_gdf = remained_gdf.to_crs(geo_crs)
    historical_db = pd.concat([remained_gdf, disappeared_gdf], ignore_index=True)
    historical_db = clean_historical_db(historical_db, geo_crs)

    # Re-extract clean disappeared subset for final export
    disappeared_clean = historical_db[historical_db["status"] == "disappeared"].copy()
    if not isinstance(disappeared_clean, gpd.GeoDataFrame):
        disappeared_clean = gpd.GeoDataFrame(disappeared_clean, geometry="geometry", crs=geo_crs)
    elif disappeared_clean.crs is None:
        disappeared_clean = disappeared_clean.set_crs(geo_crs)

    disappeared_clean.to_crs(4326).to_file(
        os.path.join(paths["outdir"], "disappeared.geojson"), driver="GeoJSON"
    )

    out_path = os.path.join(paths["outdir"], "historical_db_final.geojson")
    historical_db.to_crs(4326).to_file(out_path, driver="GeoJSON")
    log.info("historical_db_final.geojson saved: %s (%d features)", out_path, len(historical_db))

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Pipeline complete in %.1f s", elapsed)
    log.info("Outputs in: %s", paths["outdir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
