# GraphDigitizer

> A computer-vision pipeline that converts scientific figures (scatter plots, bar charts,
> line plots) into structured numerical data — CSV, JSON, or Excel — with no manual point-clicking.

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

All outputs include a validation overlay PNG for visual inspection, plus the raw
results saved to disk under `tests/result/<image-stem>/`.

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
├── main.py                       ← Streamlit entry point (run from repo root)
│
├── app/                          ← Streamlit web application
│   ├── __init__.py
│   ├── ui_components.py          ← Upload widget, preview, download buttons
│   └── calibration.py            ← Interactive axis calibration step
│
├── pipeline/                     ← Core CV pipeline (no UI dependency)
│   ├── preprocess.py             ← Stage 1 · Ingest & pre-process
│   ├── axes_detector.py          ← Stage 2 · Axis localisation + tick OCR matching
│   ├── text_mask.py              ← Shared text-region mask from global OCR
│   ├── parallel_router.py        ← Stage 3 · Detector dispatch + confidence routing
│   ├── scatter_detector.py       ← Stage 4a · Scatter marker detection
│   ├── scatter_copy.py           ← Scatter detector variant / experiments
│   ├── bar_detector.py           ← Stage 4b · Bar top detection
│   ├── line_detector.py          ← Stage 4c · Line tracing + marker isolation
│   └── coordinate_transform.py   ← Stage 5 · Pixel → data transform + export
│
├── scripts/                      ← One-off debug & per-figure regression scripts
│   ├── backtest_all.py
│   ├── extract_example1.py
│   ├── test_one_image.py
│   └── test_*.py / debug_*.py / viz_*.py
│
├── tests/
│   ├── sample_figures/           ← Input figures grouped by chart category
│   │   ├── Demo/
│   │   ├── bar chart/
│   │   ├── bar chart (no xticks)/
│   │   ├── stacked bar chart/
│   │   ├── points only/
│   │   ├── intepolated line (focus on points)/
│   │   ├── multiple line charts/
│   │   └── [done] normal line chart/
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

### `main.py`
Streamlit entry point at the repo root. Bootstraps page layout, initialises session
state, loads PaddleOCR once via `@st.cache_resource`, runs Stage 1 + Stage 2 on
upload, then Stage 3 + Stage 5 after the user confirms calibration. Supports a
batch queue for multiple uploaded images.

### `app/ui_components.py`
Reusable Streamlit widgets: image upload, preview, detected-point overlay viewer,
result data table, format selector, and download buttons.

### `app/calibration.py`
Interactive calibration step. Renders the detected axes / tick labels and lets the
user correct any OCR misread before Stage 3 fires.

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

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first install downloads PaddleOCR model weights (~100 MB) the
> first time the app starts. Subsequent runs use the cached weights.

---

### Step 5 — Launch the web app

```bash
streamlit run main.py
```

Streamlit will open **http://localhost:8501** in your default browser.

---

### Step 6 — Use the app

1. **Upload a figure** — drag-and-drop one or more PNG/JPEG charts. Multiple
   files are queued; you advance with the **Next image** button.
2. **Wait for axis detection** — PaddleOCR runs a single global pass, then the
   axes detector matches tokens to ticks.
3. **Verify calibration** — review the detected tick values, edit any that the
   OCR misread, then click **Confirm Calibration**.
4. **View results** — the extracted data table appears below the image, with an
   overlay PNG showing detected points.
5. **Download** — save results as **CSV**, **XLSX**, or the **overlay PNG**.
   Files are also persisted on disk at `tests/result/<image-stem>/`.

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

---

## Deployment Options

| Option | Command / Notes |
|---|---|
| **Local (Streamlit)** | `streamlit run main.py` — opens on http://localhost:8501 |
| **Streamlit Community Cloud** | Push to GitHub; connect the repo at [share.streamlit.io](https://share.streamlit.io). Set entry point to `main.py`. |
| **Custom server** | Any host with Python 3.10+; repeat Steps 2–5 above, then expose port 8501. |

> Docker and GitHub Actions CI/CD are planned (M7) but not yet present in the repo.

---

## Development Milestones

| Milestone | Deliverable | Status |
|---|---|---|
| **M1** — Axes pipeline | Hough + PaddleOCR axes detection, affine transform, CSV output | ✅ |
| **M2** — Bar detector | Contour-based bar top detection; error-bar / whisker handling | 🟡 |
| **M3** — Scatter detector | Blob/contour marker detection; colour-based series separation | ✅ |
| **M4** — Line detector | Skeleton tracing + marker isolation; dashed-line handling | ✅ |
| **M5** — Routing | Confidence scoring + mixed-chart detection (parallel execution pending) | 🟡 |
| **M6** — Streamlit app | Calibration correction, overlay preview, multi-format export, batch queue | ✅ |
| **M7** — GitHub deploy | CI/CD, Docker, executable builds | ⬜ |
