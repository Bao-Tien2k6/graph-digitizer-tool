"""
app/main.py
===========
Streamlit Entry Point

Responsibility
--------------
Bootstrap the Streamlit application, manage top-level session state, and route
between the two main views:
  1. Upload & Process  — the user submits a figure; the pipeline runs.
  2. Results           — detected data table, overlay image, download buttons.

No computer-vision logic lives here.  This file only orchestrates the UI and
calls into the pipeline package.

Session state keys
------------------
  st.session_state.uploaded_img   : np.ndarray | None  — current BGR image
  st.session_state.axes_info      : AxesInfo | None     — Stage 2 output
  st.session_state.routing_result : RoutingResult | None — Stage 3 output
  st.session_state.transform_result: TransformResult | None — Stage 5 output
  st.session_state.calibration_done: bool               — user confirmed axes
"""

import re
from pathlib import Path
import streamlit as st
from paddleocr import PaddleOCR
# Pipeline imports
from pipeline.preprocess         import preprocess_image, load_image_from_bytes
from pipeline.axes_detector      import detect_axes
from pipeline.parallel_router    import route
from pipeline.coordinate_transform import transform_and_export

# OCR engine
@st.cache_resource(show_spinner="Loading OCR Engine (First time only)...")
def get_ocr_engine():
    """Load PaddleOCR once and cache it in memory for the whole session."""
    return PaddleOCR(use_angle_cls=True, lang='en')

# Address of extracted results
RESULTS_DIR = Path(__file__).resolve().parent / "tests" / "result"

# UI component imports
from app.ui_components import (
    render_upload_widget,
    render_preview,
    render_download_buttons,
)
from app.calibration import render_calibration_step


# ---------------------------------------------------------------------------
# Page configuration  (must be the first Streamlit call)
# ---------------------------------------------------------------------------

def _configure_page() -> None:
    """
    Set Streamlit page title, icon, and layout.
    Called once at module level before any other st.* call.
    """
    st.set_page_config(
        page_title="PlotDigitizer",
        page_icon="📊",
        layout="wide",
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """
    Initialise all session state keys with their default values on first load.
    Must be idempotent (called on every rerun).
    """
    defaults = {
        "uploaded_img": None,
        "axes_info": None,
        "global_ocr_results": None,
        "routing_result": None,
        "transform_result": None,
        "calibration_done": False,
        "result_dir": None,
        "upload_name": None,
        # Batch queue state
        "queue_keys": [],     # identity of the current upload set
        "queue_idx": 0,       # index of the image being processed
        "processed_key": None,  # key of the image the pipeline last ran on
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_image_state() -> None:
    """Clear all per-image state so the next image starts fresh."""
    st.session_state.uploaded_img = None
    st.session_state.axes_info = None
    st.session_state.global_ocr_results = None # 
    st.session_state.routing_result = None
    st.session_state.transform_result = None
    st.session_state.calibration_done = False
    st.session_state.result_dir = None
    st.session_state.processed_key = None
    # Tick-editor widgets are keyed per tick index; drop them so a new image's
    # inputs initialise from its own detected values instead of reusing stale ones.
    for key in [k for k in st.session_state.keys() if k.startswith("tick_")]:
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _run_pipeline(raw_bytes: bytes) -> None:
    """Execute Stages 1 and 2; store results in session state."""
    try:
        with st.spinner("Loading image…"):
            img = load_image_from_bytes(raw_bytes)
        with st.spinner("Pre-processing (deskew, upscale, CLAHE)…"):
            img = preprocess_image(img)

        # OCR starting ...
        with st.spinner("Running global text detection…"):
            engine = get_ocr_engine()
            raw_result = engine.ocr(img, cls=True)
            global_ocr = raw_result[0] if raw_result and raw_result[0] else []
            st.session_state.global_ocr_results = global_ocr

        with st.spinner("Detecting axes and tick labels…"):
            # Pass the global results down!
            axes_info = detect_axes(img, global_ocr_results=global_ocr)

        st.session_state.uploaded_img = img
        st.session_state.axes_info = axes_info
        st.session_state.routing_result = None
        st.session_state.transform_result = None
        st.session_state.calibration_done = False
        st.success("Axes detected — please verify the calibration below.")
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        import traceback
        st.code(traceback.format_exc())


def _result_stem() -> str:
    """Filename-safe stem derived from the uploaded image name."""
    name = st.session_state.get("upload_name") or "extracted"
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "extracted"


def _run_detection_and_export() -> None:
    """Execute Stages 3 and 5; store results in session state."""
    img = st.session_state.uploaded_img
    axes_info = st.session_state.axes_info
    if img is None or axes_info is None:
        st.error("No image or axes info available — please re-upload.")
        return

    try:
        with st.spinner("Running parallel chart-type detection…"):
            # Pass the OCR results to the router!
            routing = route(img, axes_info, 
                global_ocr_results=st.session_state.global_ocr_results)
        st.session_state.routing_result = routing

        # Persist outputs under tests/result/<image-stem>/ so they survive the
        # session and the user can locate them on disk.
        stem = _result_stem()
        output_dir = RESULTS_DIR / stem
        output_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner(f"Transforming {routing.primary_chart_type} detections to data values…"):
            result = transform_and_export(
                detection=routing.winning_detections,
                axes=axes_info,
                original_img=img,
                output_dir=output_dir,
                formats=["csv", "xlsx", "overlay_png"],
                stem=stem,
            )
        st.session_state.transform_result = result
        st.session_state.result_dir = str(output_dir)
        n = len(result.points)
        st.success(f"Extracted {n} data point(s) as **{routing.primary_chart_type}** chart.")
        st.info(f"Results saved to: `{output_dir}`")
    except Exception as exc:
        st.error(f"Detection/export error: {exc}")
        import traceback
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Top-level layout function — Streamlit reruns this on every interaction.

    Layout structure
    ----------------
    Title row
    ────────────────────────────────────────
    Left column (40%)         Right column (60%)
      Upload widget             Preview / overlay image
      Calibration step
    ────────────────────────────────────────
    Results table (full width, shown only after detection)
    Download buttons
    """
    _configure_page()
    _init_session_state()

    st.title("📊 PlotDigitizer")
    st.caption("Extract numerical data from scientific figures.")

    col_left, col_right = st.columns([4, 6])

    with col_left:
        # Step 1: upload (one or many images)
        files = render_upload_widget()

        if not files:
            # Uploader cleared — reset the queue so a fresh upload starts clean.
            if st.session_state.queue_keys:
                st.session_state.queue_keys = []
                st.session_state.queue_idx = 0
                _reset_image_state()
        else:
            keys = [f"{f.name}_{f.size}" for f in files]

            # New upload set → restart the queue at the first image.
            if st.session_state.queue_keys != keys:
                st.session_state.queue_keys = keys
                st.session_state.queue_idx = 0
                _reset_image_state()

            idx = min(st.session_state.queue_idx, len(files) - 1)
            current = files[idx]

            if len(files) > 1:
                st.markdown(f"**Image {idx + 1} of {len(files)}** — `{current.name}`")

            # Run Stage 1+2 once per image (not on every widget interaction).
            if st.session_state.processed_key != keys[idx]:
                _reset_image_state()
                st.session_state.processed_key = keys[idx]
                st.session_state.upload_name = current.name
                _run_pipeline(current.getvalue())

            # Step 2: calibration (until the user confirms for this image).
            if st.session_state.axes_info is not None and not st.session_state.calibration_done:
                render_calibration_step(
                    img=st.session_state.uploaded_img,
                    axes_info=st.session_state.axes_info,
                    on_confirm=_run_detection_and_export,
                )

            # Step 4: advance to the next image in the batch.
            if st.session_state.transform_result is not None and idx < len(files) - 1:
                if st.button("➡️ Next image", type="primary", use_container_width=True):
                    st.session_state.queue_idx = idx + 1
                    _reset_image_state()
                    st.rerun()
            elif st.session_state.transform_result is not None and len(files) > 1:
                st.success("All images in the batch have been processed.")

    with col_right:
        if st.session_state.uploaded_img is not None:
            overlay = (
                st.session_state.transform_result.output_paths.get("overlay_png")
                if st.session_state.transform_result else None
            )
            render_preview(
                img=st.session_state.uploaded_img,
                overlay_path=overlay,
            )

    # Step 3: results table + downloads (below both columns)
    if st.session_state.transform_result is not None:
        render_download_buttons(st.session_state.transform_result)
        result_dir = st.session_state.get("result_dir")
        if result_dir:
            st.caption(f"📁 Saved to disk: `{result_dir}`")


if __name__ == "__main__":
    main()
