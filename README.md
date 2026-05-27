# Melway Historical Road Change Detection Pipeline

**Live demo:** <https://anonymous-road-atlas.github.io/melway/>

Detect and classify road changes between historical Melway map sheets (GeoTIFF)
and a modern OpenStreetMap-derived road network.

## Overview

Given a scanned Melway map as a palette GeoTIFF and a modern road GeoJSON,
the pipeline produces three labelled road datasets:

| Label | Meaning |
|-------|---------|
| **Remained** | Road exists in both the historical map and modern data |
| **Added** | Road appears only in modern data (built after the map date) |
| **Disappeared** | Road exists in the historical map but is absent in modern data |

A `historical_db_final.geojson` is also produced for use as the
"historical prior" in the next year's processing run.

## Directory Structure

This project is self-contained — all code, sample data, and the trained model
live at the repository root:

```
.
├── src/                          # Python source modules
│   ├── main.py                   # Entry point — orchestrates all stages
│   ├── config.py                 # All tunable parameters + path resolver
│   ├── logger.py                 # Logging setup (stdout + file)
│   ├── utils.py                  # Shared numeric / raster helpers
│   ├── tif_converter.py          # Palette GeoTIFF → RGBA GeoTIFF
│   ├── road_extractor.py         # Colour mask → skeleton → topology filter
│   ├── feature_extractor.py      # HIRONEX features + RF prediction
│   ├── loop_detector.py          # Circular loop detection & label propagation
│   ├── connectivity_fixer.py     # Bridge-aware Pred=0 → Pred=1 promotion
│   ├── road_comparator.py        # Historical vs modern change classification
│   ├── ocr_filter.py             # OCR-based text-artefact removal
│   └── post_processor.py         # Parallel / residential-proximity cleanup
├── data/
│   └── melway/{year}/            # Sample GeoTIFF + modern road GeoJSON
│       ├── m{map_id}.tif
│       └── mel_roads_m{map_id}.geojson
├── models/
│   └── rf_model_*.joblib         # Pre-trained Random Forest
├── outputs/                      # Results written here (default)
│   └── {year}/m{map_id}/
├── logs/                         # Per-run pipeline logs
├── index.html                    # Interactive Leaflet demo viewer
└── README.md
```

All paths are resolved relative to the project root
(see `PROJECT_ROOT` in `config.py`), so you can run the pipeline
from any working directory.

## Requirements

```
numpy
pandas
geopandas
rasterio
shapely
opencv-python
scikit-image
tqdm
joblib
matplotlib
easyocr          # optional — OCR filter is skipped if not installed
```

Install all dependencies:

```bash
pip install numpy pandas geopandas rasterio shapely opencv-python scikit-image tqdm joblib matplotlib easyocr
```

## Usage

From the repository root (or anywhere):

```bash
# Run the included sample (map sheet m003, year 2020)
python src/main.py --map_id 003 --year 2020

# Skip road re-extraction (reuse existing historical GeoJSON)
python src/main.py --map_id 003 --year 2011 --skip_extraction

# Custom log directory
python src/main.py --map_id 003 --year 2020 --log_dir ./my_logs
```

Outputs land in `outputs/{year}/m{map_id}/` by default.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--map_id` | `007` | Map sheet number (zero-padded) |
| `--year` | `2020` | Year of the historical map to process |
| `--next_year` | `2020` | Next year for chained multi-year processing |
| `--skip_extraction` | false | Reuse existing `mel_roads_m{ID}_old_2.geojson` |
| `--log_dir` | `./logs` | Directory for `pipeline.log` |

## Input Files

Already bundled under `data/` and `models/`:

```
data/melway/{year}/m{map_id}.tif               # Palette GeoTIFF
data/melway/{year}/mel_roads_m{map_id}.geojson # Modern road GeoJSON
models/rf_model_20251110.joblib                # Pre-trained RF model
```

To process additional map sheets, drop the corresponding `.tif` and
`mel_roads_*.geojson` into `data/melway/{year}/`.

## Output Files

All outputs are written to `outputs/{year}/m{map_id}/`:

```
remained.geojson            # Roads present in both periods
added.geojson                # Roads added since the historical map
disappeared.geojson          # Roads that have disappeared
historical_all.geojson       # All historical roads with status labels
historical_db_final.geojson  # Cleaned historical DB for chained runs
road_change_overview_m{id}.png
ocr_text_hits_debug.png      # Debug visualisation of OCR filter
```

## Configuration

All parameters are centralised in `config.py`.  Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BUFFER_DIST` | 10 m | Buffer radius for historical ↔ modern overlap test |
| `MIN_OVERLAP_RATIO` | 0.30 | Minimum overlap fraction for "remained" label |
| `COLOR_TOLERANCE` | 10 | BGR distance tolerance for colour matching |
| `MIN_PATH_LEN_PIX` | 10 | Minimum skeleton path length (pixels) |
| `OCR_RATIO_TH` | 0.40 | OCR overlap ratio to trigger text-artefact removal |
| `PARALLEL_ANGLE_TH_DEG` | 15° | Angle threshold for the parallel-duplicate filter |

## Demo Viewer

An interactive web viewer (`index.html`, Leaflet + Bootstrap) visualises the
pipeline outputs across years and map sheets. It loads the `remained`,
`added`, and `disappeared` GeoJSON layers from `outputs/{year}/m{map_id}/`
(or `melway_outputs/` for the full metropolitan atlas) and lets the user
step through time.

Serve the directory with any static HTTP server and open `index.html`:

```bash
python -m http.server 8000   # then visit http://localhost:8000/
```

A hosted demo (showing a subset of map sheets — rendering the full
metropolitan atlas in-browser is infeasible) is available at
<https://anonymous-road-atlas.github.io/melway/>.

## Logs

Each pipeline run writes a timestamped log file
`{log_dir}/pipeline_YYYYMMDD_HHMMSS.log` alongside the console output —
a fresh file is created on every invocation so runs are never overwritten.
Each entry includes timestamp, level, module name, and message.

## Citation

If you use this pipeline in your research, please cite:

> [Paper title and reference to be added upon publication]
