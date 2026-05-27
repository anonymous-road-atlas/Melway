"""
config.py
---------
Central configuration for the Melway road change detection pipeline.
All tuneable parameters live here; everything else imports from this module.
"""

import os

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: my-map/src/config.py -> my-map/)
# All data/model/output paths resolve relative to this root so the pipeline
# is location-independent and can be executed from anywhere.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_FILENAME = "rf_model_20251110.joblib"

# ---------------------------------------------------------------------------
# Run-time target (overridden by CLI args in main.py)
# ---------------------------------------------------------------------------
MAP_ID = "003"   # zero-padded map sheet number, e.g. "003"
YEAR = "2020"
NEXT_YEAR = "2020"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
def get_paths(map_id: str, year: str, next_year: str,
              data_dir: str = DATA_DIR,
              outputs_dir: str = OUTPUTS_DIR,
              models_dir: str = MODELS_DIR) -> dict:
    """
    Build the dict of absolute paths for this run.

    Layout (inside my-map/):
        data/melway/{year}/m{map_id}.tif
        data/melway/{year}/mel_roads_m{map_id}.geojson
        models/rf_model_*.joblib
        outputs/{year}/m{map_id}/...
    """
    base = os.path.join(data_dir, "melway", year)
    outdir = os.path.join(outputs_dir, year, f"m{map_id}")
    os.makedirs(outdir, exist_ok=True)

    modern_path = (
        os.path.join(outputs_dir, next_year, f"m{map_id}", "historical_db.geojson")
        if year != "2020"
        else os.path.join(base, f"mel_roads_m{map_id}.geojson")
    )

    return {
        "base_dir": base,
        "src_tif": os.path.join(base, f"m{map_id}.tif"),
        "rgba_tif": os.path.join(base, f"m{map_id}_rgba.tif"),
        "historical_raw": os.path.join(base, f"mel_roads_m{map_id}_old_2.geojson"),
        "modern": modern_path,
        "outdir": outdir,
        "model": os.path.join(models_dir, MODEL_FILENAME),
    }

# ---------------------------------------------------------------------------
# Road color samples (BGR order, for cv2)
# ---------------------------------------------------------------------------
FCLASS_COLORS = {
    "residential": [
        (68, 176, 243),
        (116, 181, 212),
        (50, 153, 153),
    ],
    "motorway": [(75, 172, 0)],
    "trunk": [(32, 32, 32)],
    "primary": [(143, 143, 143)],
    "tertiary": [(143, 143, 143), (14, 236, 135)],
    "unclassified": [(31, 155, 180)],
}

TARGET_CLASSES = list(FCLASS_COLORS.keys())

# ---------------------------------------------------------------------------
# Skeleton / topology
# ---------------------------------------------------------------------------
COLOR_TOLERANCE = 10       # BGR distance tolerance for color matching
MIN_PATH_LEN_PIX = 10      # discard skeleton fragments shorter than this
MODERN_BUFFER_DIST = 30    # buffer (map units) when checking modern road overlap

# Centerline post-processing
CENTERLINE_CLASSES = {"trunk", "primary", "tertiary"}
THICKNESS_BY_CLASS = {"trunk": 9, "primary": 7, "tertiary": 5}

# GeoJSON smoothing
DECIMATE_STEP = 3
SIMPLIFY_TOL = 2.0
USE_CHAIKIN = True
CHAIKIN_ITERS = 2

# ---------------------------------------------------------------------------
# HIRONEX feature extraction
# ---------------------------------------------------------------------------
PIXEL_TOLERANCE = 2
CROSSSECT_DIST = 25
CROSSSECT_LENGTH = 200
SAMPLES_PER_SECTION = 21
TARGET_BAND_LENGTH = 30
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Road comparison (historical vs modern)
# ---------------------------------------------------------------------------
BUFFER_DIST = 10           # metres
MIN_OVERLAP_RATIO = 0.3    # 30 % overlap required for "remained"

# ---------------------------------------------------------------------------
# CRS identifiers
# ---------------------------------------------------------------------------
GEO_EPSG = 7844    # GDA2020 Geographic (degrees) — matches TIF
METRIC_EPSG = 7855 # GDA2020 / MGA Zone 55 (metres)

# ---------------------------------------------------------------------------
# Loop / connectivity
# ---------------------------------------------------------------------------
CLOSE_EPS_M = 1.0
CANDIDATE_BUFFER_M = 5.0
RADIAL_CV_MAX = 0.10
ISO_CIRC_MIN = 0.80
ENDPOINT_EPS_M = 1.0
JOIN_M = 1.0

# ---------------------------------------------------------------------------
# OCR filter
# ---------------------------------------------------------------------------
OCR_CONF_TH = 0.3
TEXT_BUF_PIX = 1.0
OCR_RATIO_TH = 0.4

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
PARALLEL_ANGLE_TH_DEG = 15
REMOVE_FCLASSES_NEAR_RESIDENTIAL = {"trunk", "residential", "primary", "tertiary"}
