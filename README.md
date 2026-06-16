# GraphDigitizer

> A computer-vision pipeline that converts scientific figures (scatter plots, bar charts,
> line plots) into structured numerical data — CSV, JSON, or Excel — with no manual point-clicking.
>
> Backend: **FastAPI** wrapping the CV pipeline. Frontend: **React + Vite** with
> a Konva canvas (drag misplaced markers, right-click to delete), TanStack Table
> (inline-edit rows), Zustand (client state), and SheetJS (CSV/JSON/XLSX export).

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Pipeline Overview](#pipeline-overview)
3. [Repository Layout](#repository-layout)
4. [File-by-File Reference](#file-by-file-reference)
5. [Quick Start](#quick-start)
6. [Deployment Options](#deployment-options)
7. [Development Milestones](#development-milestones)
8. [Dependencies](#dependencies)

---

## What It Does

GraphDigitizer accepts an image (PNG/JPEG) containing one of three chart types
and returns the underlying data points as machine-readable numbers.

| Chart Type | What Is Detected | Output |
|---|---|---|
| **Scatter plot** | Marker centroids per series (color/shape) | (x, y) table per series |
| **Bar chart** | Bar-top pixel position per group | (label, value) table |
| **Line plot** | Discrete marker blobs along traced curves | (x, y) table per line |

Once `/api/calibrate` returns the affine matrix and points, every interactive
edit (drag a marker, edit a cell, delete a row, export to CSV/JSON/XLSX) is
pure client-side state — no server round-trip.

---

## Pipeline Overview

```
Input image (PNG / JPEG)
        │
        ▼
┌─────────────────────────────┐
│  Stage 1 · Ingest & Pre-    │  pipeline/preprocess.py
│  process                    │  • Deskew (Hough lines → rotate)
│                             │  • CLAHE contrast on L-channel
│                             │  • Upscale if image is small
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Global OCR pass            │  PaddleOCR (cached at app startup)
│                             │  • Single OCR call; results reused by
│                             │    axes detector + text-mask builder
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2 · Axes Detection   │  pipeline/axes_detector.py
│                             │  • Probabilistic Hough → x/y axes
│                             │  • Tick mark extraction
│                             │  • Match OCR tokens → tick labels
│                             │  • Linear / log scale regression
│                             │  • Grid-line inpainting
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3 · Detection +      │  pipeline/parallel_router.py
│  Routing                    │  • Build text mask (text_mask.py)
│                             │  • Run all 3 detectors (sequential today;
│                             │    interface designed for ThreadPool)
│                             │  • Confidence vote → primary chart type
│                             │  • Mixed-chart flag when 2 detectors > 0.5
└──────┬──────────┬───────────┘
       │          │          │
  Scatter      Bar Top    Line Markers
       │          │          │
 scatter_    bar_        line_
 detector.py detector.py detector.py
       │          │          │
       └──────────┴──────────┘
                  │
                  ▼
┌─────────────────────────────┐
│  Stage 5 · Coordinate       │  pipeline/coordinate_transform.py
│  Transform                  │  • Affine calibration from axes
│                             │  • Log-scale handling
│                             │  • Export: CSV / XLSX + overlay PNG
└─────────────────────────────┘
```

---

## Repository Layout

```
graph-digitizer-tool/
│
├── api/                          ← FastAPI backend (replaces the old Streamlit app)
│   ├── __init__.py
│   ├── main.py                   ← /digitize, /calibrate, /image, /session, /healthz
│   └── schemas.py                ← Pydantic v2 request / response models
│
├── frontend/                     ← React + Vite web client
│   ├── package.json
│   ├── vite.config.js            ← Dev proxy /api → localhost:8000
│   ├── tailwind.config.js / postcss.config.js / components.json
│   ├── index.html
│   └── src/
│       ├── main.jsx, App.jsx, index.css
│       ├── lib/
│       │   ├── api.js            ← fetch wrappers for /digitize, /calibrate
│       │   └── utils.js          ← cn() helper for shadcn / Tailwind
│       ├── store/
│       │   └── useDigitizerStore.js   ← Zustand: image, axes, points, affine, history
│       └── components/
│           ├── FileUploadZone.jsx     ← drag-drop PNG/JPG → /api/digitize
│           ├── ImageCanvas.jsx        ← react-konva: drag, zoom, right-click delete
│           ├── CalibrationPanel.jsx   ← per-tick inputs + "Detect points"
│           ├── DataTable.jsx          ← TanStack Table: click-to-edit, sort, delete
│           ├── ExportBar.jsx          ← SheetJS: CSV / JSON / XLSX (client-side)
│           └── ui/                    ← shadcn primitives (button, card, input, …)
│
├── pipeline/                     ← Core CV pipeline (no UI dependency, unchanged)
│   ├── preprocess.py             ← Stage 1 · Ingest & pre-process
│   ├── axes_detector.py          ← Stage 2 · Axis localisation + tick OCR matching
│   ├── text_mask.py              ← Shared text-region mask from global OCR
│   ├── parallel_router.py        ← Stage 3 · Detector dispatch + confidence routing
│   ├── scatter_detector.py       ← Stage 4a · Scatter marker detection
│   ├── bar_detector.py           ← Stage 4b · Bar top detection
│   ├── line_detector.py          ← Stage 4c · Line tracing + marker isolation
│   └── coordinate_transform.py   ← Stage 5 · Pixel → data transform + export
│
├── docs/
│   └── ui/                       ← Static HTML mockups of each app view
│       ├── styles.css
│       ├── index.html
│       ├── 01-upload.html
│       ├── 02-calibration.html
│       ├── 03-data-editor.html
│       └── 04-results.html
│
├── scripts/                      ← One-off debug & per-figure regression scripts
│
├── tests/
│   ├── sample_figures/           ← Input figures grouped by chart category
│   ├── ground_truth/             ← 52 CSVs + _manifest.csv (per-figure truth data)
│   ├── result/                   ← Pipeline outputs per image stem
│   └── test_preprocess.py
│
├── Example 1.xlsx
├── requirements.txt
└── README.md
```

---

## File-by-File Reference

### `api/main.py`
FastAPI app. Loads PaddleOCR once via the `lifespan` context, keeps an in-memory
session cache (30-min TTL) of the preprocessed image + global OCR + `AxesInfo`,
and exposes five JSON routes:

| Route | Purpose |
|---|---|
| `POST /api/digitize` | multipart upload → Stage 1 + global OCR + Stage 2 |
| `POST /api/calibrate` | apply corrected ticks, refit scale, run Stage 3 + transform |
| `GET /api/image/{id}` | serve the preprocessed PNG for the canvas |
| `DELETE /api/session/{id}` | release the cache entry |
| `GET /api/healthz` | liveness probe |

### `api/schemas.py`
Pydantic v2 models (`DigitizeResponse`, `CalibrateRequest`, `CalibrateResponse`,
`PointPayload`, etc.). All numeric fields are plain `float`/`int` so the JSON
payload is portable and the frontend can apply the affine matrix locally.

### `frontend/src/store/useDigitizerStore.js`
Zustand store — holds `sessionId`, `imageUrl`, `axes`, `points`, `affine`, and an
edit history stack. Implements `movePoint`, `updatePointField`, `deletePoint`,
and inline pixel↔data math (linear and log10).

### `frontend/src/components/ImageCanvas.jsx`
react-konva canvas. Renders the preprocessed image plus one `<Circle>` per
detected point; markers are `draggable` (drag → `movePoint`), right-click /
double-click deletes, wheel zooms, stage drag pans.

### `frontend/src/components/CalibrationPanel.jsx`
Per-tick number inputs for both axes plus scale R² chips. The "Detect points"
button posts the corrected ticks to `/api/calibrate`.

### `frontend/src/components/DataTable.jsx`
TanStack Table. Click an x/y cell to edit it (commits on Enter / blur). Rows
sort by series / x / y; hover highlights the matching canvas marker; per-row
delete button. An Undo button reverses the last edit using the store's history.

### `frontend/src/components/ExportBar.jsx`
SheetJS-backed export buttons (CSV / JSON / XLSX). Runs entirely in the browser
— no `/api/export` round-trip.

### `pipeline/preprocess.py`
**Stage 1.** Accepts bytes or a NumPy array. Returns a normalised BGR image
(`load_image_from_bytes`, `preprocess_image`) ready for axis detection.

### `pipeline/axes_detector.py`
**Stage 2.** Locates axes lines, extracts tick positions, matches them to tokens
from the global OCR pass, fits a linear or log scale model, and removes grid lines
via inpainting. Returns an `AxesInfo` dataclass (with `inpainted_image`,
`plot_region`, etc.).

### `pipeline/text_mask.py`
Builds a binary text mask from the global PaddleOCR result + plot region. Shared
input to every Stage-4 detector so text never gets mistaken for a marker / bar
edge / line stroke.

### `pipeline/parallel_router.py`
**Stage 3.** Runs scatter, bar, and line detectors (sequentially today — the
interface is designed for a `ThreadPoolExecutor` swap-in), collects their
`DetectionResult` objects (pixel positions + confidence), selects the winner, and
flags mixed charts when two detectors both exceed 0.5. Returns a `RoutingResult`.

### `pipeline/scatter_detector.py`
**Stage 4a.** Blob/contour-based detection of marker shapes (circles, squares,
triangles, crosses). Color-space clustering separates multi-series.

### `pipeline/bar_detector.py`
**Stage 4b.** Finds filled rectangles via contour analysis (aspect ratio filter).
Groups bars by LAB-space K-means colour. Extracts top-edge y-pixel per bar.
Optionally detects error bars / whiskers and dedupes them.

### `pipeline/line_detector.py`
**Stage 4c.** Skeletonises colour-segmented curves, identifies discrete marker
blobs as local width spikes, handles dashed-line gaps, and returns marker
centroids along each skeleton.

### `pipeline/coordinate_transform.py`
**Stage 5.** Builds a pixel→data affine from the calibrated axes, handles log
axes, and exports results to CSV / XLSX plus a validation overlay PNG. Outputs
are written to `tests/result/<stem>/`.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher ([download](https://www.python.org/downloads/))
- Node.js 18 or higher (20+ recommended) ([download](https://nodejs.org/))
- `git` (to clone the repository)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/graph-digitizer-tool.git
cd graph-digitizer-tool
```

---

### Step 2 — Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
```

**macOS / Linux**
```bash
python3 -m venv venv
```

---

### Step 3 — Activate the virtual environment

**Windows (PowerShell)**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source venv/bin/activate
```

---

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PaddleOCR model weights (~100 MB) the
> first time the app starts. Subsequent runs use the cached weights.

---

### Step 5 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

This installs React, Vite, Tailwind, shadcn primitives, Konva, TanStack Table,
Zustand, and SheetJS (~185 packages).

---

### Step 6 — Launch the app (two terminals)
**Terminal 1 - backend (FastAPI app)**

```bash
python -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend (Vite dev server on port 5173):**

```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**. The Vite dev server proxies `/api/*` to the
FastAPI backend, so no CORS configuration is needed in development.

---

### Step 7 — Use the app

1. **Upload a figure** — drag-and-drop a single PNG/JPG chart into the upload
   zone. The backend runs Stage 1 (preprocess) + global OCR + Stage 2 (axes).
2. **Verify calibration** — review the detected tick values in the right panel,
   edit any that OCR misread, then click **Detect points**. The backend runs
   Stage 3 (routing) + Stage 5 (transform) and returns the points + affine
   matrix.
3. **Refine on the canvas** — drag any misplaced marker to its correct pixel;
   the (x, y) values update via the inverse affine. Right-click or
   double-click a marker to delete it.
4. **Refine in the table** — click an x or y cell to edit it; the corresponding
   marker moves on the canvas. Click the row delete button to remove a point.
5. **Export** — click **CSV**, **JSON**, or **XLSX**. SheetJS builds the file
   in the browser and triggers a download.

All editing in steps 3–5 is pure client state — there is no `/api/export` and no
re-run of the pipeline.

---

### Running the pipeline from Python (no UI)

```python
from pipeline.preprocess          import preprocess_image, load_image_from_bytes
from pipeline.axes_detector       import detect_axes
from pipeline.parallel_router     import route
from pipeline.coordinate_transform import transform_and_export

with open("my_figure.png", "rb") as f:
    img = preprocess_image(load_image_from_bytes(f.read()))

axes    = detect_axes(img, global_ocr_results=[])  # pass OCR list if available
routing = route(img, axes)
result  = transform_and_export(
    detection   = routing.winning_detections,
    axes        = axes,
    original_img= img,
    output_dir  = "./tests/result/my_figure",
    formats     = ["csv", "xlsx", "overlay_png"],
    stem        = "my_figure",
)
print(f"Extracted {len(result.points)} points → {result.output_paths}")
```

---

### Deactivating the virtual environment

```bash
deactivate
```