# GraphDigitizer for Published Graphs

**Pipeline Proposal · Environmental Research Tools**

A computer-vision pipeline that detects and extracts data point values from scatterplots, bar graphs, and line plots in scientific figures — turning image pixels back into numbers.

`OpenCV` `PaddleOCR` `Streamlit` `Python 3.10+`

---

## 01 · Problem Statement

### Why this pipeline exists

Environmental researchers routinely extract values from published figures — a process done manually today: zoom in, squint, estimate. For a single paper with 8 figures this can cost 2–4 hours. GraphDigitizer automates the three core scenarios:

**Graph Type A — Scatter Plot**
Detect individual point markers, classify their series (color/shape), and map pixel (x, y) → data (x, y) using calibrated axes.

**Graph Type B — Bar Chart**
Segment bar regions, identify their tops, and compute heights relative to the y-axis scale — including grouped and stacked variants.

**Graph Type C — Line Plot**
Trace curve paths, isolate marker points on each line, and recover the series values with sub-pixel interpolation.

---

## 02 · Architecture Overview

### Five-stage pipeline (with a shared OCR pass)

| Stage | Name | Description |
|-------|------|-------------|
| 01 | **Ingest & Pre-process** | Accept image, deskew, CLAHE contrast, upscale if needed |
| —  | **Global OCR pass** | One PaddleOCR call; results are shared by Stage 2 and the text mask |
| 02 | **Axes Detection** | Locate axes, ticks, grid lines; match OCR tokens to tick positions |
| 03 | **Detection + Routing** | Run all three detectors; winner chosen by confidence vote |
| 04 | **Target Object Extract** | Winning detector returns pixel positions of points / bar tops / line markers |
| 05 | **Coordinate Transform** | Pixel → data space via affine + OCR-parsed scale; export CSV / XLSX / overlay |

> **Design principle:** Each stage is a standalone module with a clean interface. Researchers can override any single stage (e.g. correct an OCR tick label in the calibration UI) without rerunning the full pipeline. Stage 3 also naturally handles **mixed charts** (e.g. a line + scatter overlay) by letting multiple detectors report results simultaneously.

> **OCR is hoisted to the top level.** PaddleOCR is heavy to initialise, so the app loads it once via `@st.cache_resource` and passes the single OCR result down to both the axes detector and the shared text-mask builder (`pipeline/text_mask.py`). This avoids running OCR three times on the same figure.

---

## 03 · Core Methods — Specific Approaches

### Algorithms & models

#### 1 — Image Pre-processing

Minimal processing for screenshot input — paper figures have white backgrounds, near-zero noise, and no color cast. Heavy filtering is counterproductive because it destroys the thin edges (tick marks, bar tops, marker borders) that later stages depend on.

- **Deskew:** Hough-line transform on horizontal/vertical edges → rotate to align axes
- **Upscale if small:** small figures are upscaled with `cv2.INTER_CUBIC` before any detection — improves OCR and sub-pixel accuracy
- **CLAHE contrast:** Applied on the L-channel in LAB color space. Helps distinguish faint grid lines and low-contrast markers without blurring edges
- **No denoising filter:** for clean screenshots, blur / bilateral / median all add edge degradation with no noise benefit, so they are skipped
- **No white-balance:** paper backgrounds are consistently white; nothing to correct

*Tools: OpenCV 4.x · scikit-image · Pillow*

---

#### 2 — Axes & Scale Detection

The most critical stage — errors here propagate to all extracted values.

- **Axis localization:** Detect the two longest horizontal + vertical lines via probabilistic Hough transform; these are the x-axis and y-axis
- **Tick mark extraction:** Short perpendicular line segments along each axis
- **OCR on tick labels:** **PaddleOCR** is run once globally for the figure; the axes detector matches tokens spatially to detected tick positions rather than running its own OCR pass
- **Scale fitting:** Linear or log-scale regression on (pixel position → parsed number). Detects log axes via residual curvature test
- **Grid line removal:** Suppress detected grid lines via inpainting before data point detection (the inpainted image is what every Stage-4 detector consumes)

*Tools: PaddleOCR 2.8 · OpenCV · NumPy regression*

---

#### 3 — Detection & Target-Object Routing

Instead of a dedicated chart-type classifier model, all three detectors run on the same input. The pipeline asks: *"what object do I need to extract a value from?"* — and lets each detector answer with a confidence score. This eliminates a model dependency and naturally handles mixed charts where both scatter markers and line markers coexist in the same figure.

- **Shared text mask:** `pipeline/text_mask.py` builds a binary mask of text regions from the global OCR result + plot region. Every detector subtracts this so OCR tokens never get mistaken for markers / bar edges / line strokes
- **Execution model:** the dispatcher in `parallel_router.py` runs the three detectors **sequentially today** behind an interface designed for a `ThreadPoolExecutor` drop-in (parallelisation is an open M5 task)
- **Per-detector confidence score:**
  - Scatter: N detections × mean blob/contour confidence
  - Bar: N contours × rectangularity score (aspect ratio fit to bar shape)
  - Line: N skeleton pixels × continuity score (ratio of connected vs. broken skeleton)
- **Winner selection:** Highest confidence detector is declared the primary chart type. If two detectors both score > 0.5, the figure is flagged as a **mixed chart** and both result sets are returned
- **Target object mapping:** scatter → *point centroid* | bar → *bar top y-pixel* | line → *marker blob centroid along skeleton*
- **No model required:** Pure CV heuristics — no GPU, no fine-tuned weights, no training data needed

*Tools: Python `concurrent.futures` (planned) · OpenCV · NumPy scoring*

---

#### 4A — Scatter Point Detection

Locate individual data markers at sub-pixel precision.

- **Primary detector:** OpenCV blob + contour analysis tuned for scientific marker symbols (circles, squares, triangles, crosses). No YOLO / no neural-net dependency today — a fine-tuned detector is a future option but not currently wired in
- **Series separation:** K-means clustering in HSV color space to separate series with different marker colors; spatial fallback when colors are similar
- **Sub-pixel center:** Blob centroid via `cv2.connectedComponentsWithStats` after per-series color mask
- **Overlap handling:** Non-maximum suppression to deduplicate closely packed points
- **Hollow markers / pink-on-white / on-curve markers:** dedicated handling lives in the marker-aware code paths (see `scripts/test_markers_22.py`, `debug_hollow.py`, `test_scatter_on_22.py`)

*Tools: OpenCV · K-means (scikit-learn)*

---

#### 4B — Bar Top Detection

Segment each bar and find its upper boundary precisely.

- **Bar segmentation:** Detect filled rectangles via contour analysis (`cv2.findContours` + aspect ratio filter)
- **Color clustering:** Group bars by fill color (for grouped bar charts) using LAB-space KMeans
- **Top edge extraction:** For each bar contour, compute the topmost row of filled pixels → bar top y-pixel
- **Error bar / whisker detection:** Thin vertical lines extending above bar tops are extracted as uncertainty bounds; a dedupe pass removes whiskers double-counted as bars
- **Stacked bars:** Detect internal horizontal lines inside bar columns; each segment treated as a sub-value
- **Bar charts with no x-ticks:** `tests/sample_figures/bar chart (no xticks)/` is a dedicated regression set

*Tools: OpenCV contours · scikit-learn KMeans*

---

#### 4C — Line Marker Detection

Trace curves and isolate discrete data markers from connecting lines.

- **Line tracing:** Skeletonize each color-segmented curve via morphological thinning (`skimage.morphology.skeletonize`); sample pixels at uniform arc-length intervals
- **Marker isolation:** Detect thickened blobs along the skeleton (local width spikes > 2× median line width) as discrete data points
- **Dashed line handling:** Gap interpolation — connect segments separated by short gaps in the skeleton
- **Series by color:** Per-color-channel mask before tracing to handle overlapping lines of different colors
- **Interpolated lines (focus on points):** dedicated category in `tests/sample_figures/`

*Tools: skimage morphology · SciPy curve fitting*

---

#### 5 — Coordinate Transformation & Output

Convert pixel positions to real data values.

- **Affine transform:** Pixel→data affine matrix derived from the calibrated axes (origin + max x + max y). Handles slight axis rotation/shear
- **Log-scale handling:** Fit `value = 10^(a × pixel + b)` if a log axis is detected
- **Output formats:** CSV, XLSX, and a validation overlay PNG showing detected points on the original figure
- **Persistence:** Results are written to `tests/result/<image-stem>/` so the user can locate them on disk after the session ends

*Tools: NumPy linalg · openpyxl · matplotlib overlay*

---

## 04 · Approach Comparison

### Method selection rationale

| Approach | Accuracy | Speed | Offline? | Best For |
|----------|----------|-------|----------|----------|
| **Confidence-vote routing** (Stage 3) | High | Fast | ✓ | All chart types; mixed scatter+line figures |
| **Blob / contour marker detection** (scatter) | High | Fast | ✓ | Standard scatter markers (●, ▲, ■, ×) |
| **Color mask + blob centroid** | Medium | Very fast | ✓ | Well-separated series with distinct colors |
| **Contour analysis** (bar tops) | High | Fast | ✓ | Clean bar charts, grouped and stacked bars |
| **Skeleton tracing** (line markers) | Medium | Moderate | ✓ | Smooth line plots; fewer overlapping series |
| **Multimodal LLM fallback** | High | Slow + $ | ✗ | Complex or degraded figures; last resort only |

> **Recommended stack:** Run the full CV pipeline first — no classifier model needed. If overall detection confidence is low on any figure, optionally escalate to a multimodal LLM API call for that figure only.

---

## 05 · Deployment

The current version ships as a Streamlit web app launched from the repo root.

### GitHub repository structure (actual)

```
graph-digitizer-tool/
├── main.py                       ← Streamlit entry point
├── app/
│   ├── __init__.py
│   ├── ui_components.py          ← Upload, preview, download buttons
│   └── calibration.py            ← Interactive axis calibration UI
├── pipeline/
│   ├── preprocess.py             ← Stage 1: deskew, CLAHE, upscale
│   ├── axes_detector.py          ← Stage 2: Hough + OCR-token matching
│   ├── text_mask.py              ← Shared text-region mask
│   ├── parallel_router.py        ← Stage 3: detector dispatch + confidence vote
│   ├── scatter_detector.py       ← Blob/contour scatter detection
│   ├── scatter_copy.py           ← Scatter variant for A/B comparison
│   ├── bar_detector.py           ← Contour analysis for bar tops
│   ├── line_detector.py          ← Skeleton tracing for line markers
│   └── coordinate_transform.py   ← Stage 5: pixel→data affine + export
├── scripts/                      ← Debug & per-figure regression scripts
│   ├── backtest_all.py
│   ├── extract_example1.py
│   ├── test_one_image.py
│   └── test_*.py / debug_*.py / viz_*.py
├── tests/
│   ├── sample_figures/           ← Inputs grouped by chart category
│   ├── ground_truth/             ← 52 per-figure CSVs + _manifest.csv
│   ├── result/                   ← Pipeline outputs per image stem
│   └── test_preprocess.py
├── Example 1.xlsx
├── requirements.txt
└── README.md
```

### Planned automation (not yet present)

- **On every push to main — auto-deploy** to Streamlit Community Cloud
- **On GitHub Release tag — build executables** via PyInstaller
- **On every PR — accuracy CI** against `tests/ground_truth/`

These are part of M7 and are not wired up in the repo today.

---

## 06 · User Experience Flow

### What the researcher sees

**Step 1 — Upload figure(s)**
Drag one or more PNG/JPG charts into the browser. Multi-file uploads are queued; the app shows *"Image i of N"* and exposes a **Next image** button.

**Step 2 — Verify calibration**
App proposes tick values from the global OCR pass. The researcher can edit any misread tick label and click **Confirm Calibration**.

**Step 3 — Review detections**
Detected points are shown as an overlay on the figure, alongside the extracted data table.

**Step 4 — Export**
One-click download as CSV, XLSX, or the overlay PNG. The same files are also persisted on disk under `tests/result/<image-stem>/`.

