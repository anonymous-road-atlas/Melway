# Melway Road Change Timeline — Demo Website Guide

An interactive web map that visualises how Melbourne's road network changed across four decades, using Melway historical map data compared against a modern road network.

---

## Quick Start

Because the page fetches local GeoJSON files via `fetch()`, you must serve it through an HTTP server — browsers block direct `file://` access.

### Option 1 — Python (no install required)

```bash
cd my-map
python3 -m http.server 8080
```

Open your browser at **http://localhost:8080**

### Option 2 — Node.js

```bash
npx serve my-map
```

### Option 3 — VS Code Live Server

Right-click `index.html` → **Open with Live Server**.

---

## Interface Overview

```
┌──────────────────────────────────────────────────────┐
│                      MAP AREA                        │
│          (OpenStreetMap + GeoJSON overlays)          │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│  Year:  [ 2001 ] [ 2006 ] [ 2011 ] [●2020]           │
│  ─────────────────────────────────────────────────── │
│  Layers:  [Added] [Remained] [Disappeared]           │
│  Focus:   [CBD] [North] [South] [West] [East]        │
└──────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────┐
│  Road Type Explanation (legend)                      │
└──────────────────────────────────────────────────────┘
```

---

## Controls

### Year Buttons

Switches the active dataset to a different historical snapshot.

| Button | Data year |
|--------|-----------|
| 2001 | Earliest snapshot |
| 2006 | — |
| 2011 | — |
| 2020 | Default (loaded on startup) |

Clicking a year button:
1. Removes all currently rendered road layers from the map.
2. Reads `melway_outputs/<year>/blocks.json` to discover which map blocks exist.
3. Loads `historical_db_final.geojson` (grey base layer) for every block — always visible.
4. Loads `added`, `persisted`, and `disappeared` GeoJSON for every block — shown only if the corresponding toggle is active.

> The map viewport is **not** reset when switching years, so you can compare the same area across time.

---

### Layer Toggle Buttons

These three buttons independently show or hide a road change category.

| Button | Colour | Meaning |
|--------|--------|---------|
| **Added** | Blue `#007bff` | Road appears in the selected year but not in the previous snapshot |
| **Remained** | Green `#40be7f` | Road exists in both the previous and selected year |
| **Disappeared** | Red `#ff0000` | Road existed historically but is absent in the selected year |

The grey base layer (**Historical DB**) is always visible and cannot be toggled off.

Clicking a toggle button switches it on or off without moving the map or reloading data.

---

### Focus Area Buttons

Instantly fly the camera to a pre-defined region of Melbourne.

| Button | Area |
|--------|------|
| CBD | Central Business District |
| North | North Melbourne (around m003) |
| South | Southern suburbs (around m168) |
| West | Western suburbs (around m220) |
| East | Eastern suburbs (around m121) |

These use Leaflet's `flyToBounds()` for a smooth animated pan/zoom.

---

## Data Structure

```
melway_outputs/
└── <year>/                        # 2001 | 2006 | 2011 | 2020
    ├── blocks.json                # ["m003", "m007", "m043", ...]
    └── <block>/                   # one directory per map sheet
        ├── historical_db_final.geojson   # base road network (always shown)
        ├── added.geojson
        ├── persisted.geojson
        └── disappeared.geojson
```

`blocks.json` controls which blocks are loaded for each year — missing blocks or missing files are silently skipped, so partial datasets work fine.

---

## How the Loading Works (Technical)

1. `loadYear(year)` is called on startup with `"2020"`.
2. It fetches `melway_outputs/2020/blocks.json` → gets an array of block IDs.
3. For each block, six `fetch()` calls fire in parallel:
   - `historical_db_final.geojson` → rendered immediately, added to the map.
   - `added`, `persisted`, `disappeared` → stored in memory, added to the map only if the toggle is active.
4. After 300 ms, the map auto-fits its viewport to the loaded base layers (only on first load).
5. Subsequent `loadYear()` calls clear the previous layers and repeat steps 2–3. The viewport is not changed.

---

## Adding a New Year

1. Create `melway_outputs/<new_year>/`.
2. Add `blocks.json`:
   ```json
   ["m003", "m007"]
   ```
3. Add the four GeoJSON files for each block.
4. Register the year in `index.html` (line ~196):
   ```js
   const years = ["2001", "2006", "2011", "2020", "<new_year>"];
   ```

---

## Adding a New Focus Area

In `index.html`, follow this pattern:

**HTML** — add a button inside the Focus Areas `controls-row`:
```html
<button id="btnMyArea" type="button" class="btn btn-outline-dark focus-btn">
  My Area
</button>
```

**JS** — define bounds and wire the click handler:
```js
const myAreaBounds = L.latLngBounds(
  [-37.90, 144.90],   // [south, west]
  [-37.80, 145.00]    // [north, east]
);

document.getElementById("btnMyArea").addEventListener("click", () => {
  setActiveFocusButton("btnMyArea");
  map.flyToBounds(myAreaBounds, { padding: [20, 20] });
});
```

---

## Dependencies

All loaded from CDN — no installation required.

| Library | Version | Role |
|---------|---------|------|
| [Leaflet](https://leafletjs.com/) | 1.7.1 | Interactive map and GeoJSON rendering |
| [Bootstrap](https://getbootstrap.com/) | 5.1.3 | Responsive layout and button styles |
| OpenStreetMap tiles | — | Base map background |
