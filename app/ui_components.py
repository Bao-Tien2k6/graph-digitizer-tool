"""
app/ui_components.py
====================
Reusable Streamlit widgets for PlotDigitizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from pipeline.coordinate_transform import TransformResult


def render_upload_widget() -> list:
    """
    Render the image upload widget (accepts multiple files for batch work).
    Returns the list of uploaded file objects (empty list if none).
    """
    uploaded = st.file_uploader(
        "Upload scientific figure(s) (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload one or many charts. They are processed one at a time — "
             "verify each calibration, extract, then move to the next. "
             "Multi-panel figures: crop to a single panel first.",
    )
    return uploaded or []


def render_preview(
    img: np.ndarray,
    overlay_path: Optional[Path] = None,
) -> None:
    """
    Show the uploaded image, and if available, the detection overlay.
    """
    if overlay_path is not None and Path(overlay_path).exists():
        st.image(str(overlay_path), caption="Detection overlay", use_container_width=True)
    else:
        # Show BGR → RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Uploaded figure", use_container_width=True)


# def render_result_table(result: TransformResult) -> None:
#     """Display extracted data points as a simple two-column (x, y) table."""
#     if not result.points:
#         st.warning("No data points were extracted.")
#         return

#     st.subheader("Extracted Data Points")

#     import pandas as pd
#     from collections import defaultdict

#     series_map: dict[int, list] = defaultdict(list)
#     for p in result.points:
#         series_map[p.series_id].append(p)
#     multi = len(series_map) > 1

#     for sid, pts in sorted(series_map.items()):
#         rows = [
#             {"x": p.x, "y": p.y}
#             for p in sorted(pts, key=lambda p: p.x)
#         ]
#         df = pd.DataFrame(rows, columns=["x", "y"])
#         if multi:
#             st.markdown(f"**Series {sid}**")
#         st.dataframe(df, use_container_width=True, hide_index=True)

#     st.caption(f"Chart type: **{result.chart_type}** · "
#                f"{len(result.points)} point(s) across "
#                f"{len(series_map)} series")


def render_download_buttons(result: TransformResult) -> None:
    """Render download buttons for all exported files."""
    if not result.output_paths:
        return

    st.subheader("Download Results")
    cols = st.columns(len(result.output_paths))

    labels = {
        "csv": ("📄 CSV", "text/csv"),
        "json": ("📋 JSON", "application/json"),
        "xlsx": ("📊 Excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "overlay_png": ("🖼️ Overlay PNG", "image/png"),
    }

    for col, (fmt, path) in zip(cols, result.output_paths.items()):
        path = Path(path)
        if not path.exists():
            continue
        label, mime = labels.get(fmt, (fmt, "application/octet-stream"))
        with col:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=path.name,
                mime=mime,
                use_container_width=True,
            )
