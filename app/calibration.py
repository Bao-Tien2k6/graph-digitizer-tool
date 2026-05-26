"""
app/calibration.py
==================
Axis calibration confirmation step.

Displays the auto-detected tick values so the researcher can verify them
before committing to the coordinate transform. Allows manual override of
individual tick values if OCR misread them.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import streamlit as st

from pipeline.axes_detector import AxesInfo, TickInfo


def render_calibration_step(
    img: np.ndarray,
    axes_info: AxesInfo,
    on_confirm: Callable,
) -> None:
    """
    Show detected axis tick values and let the user confirm or correct them.

    Parameters
    ----------
    img : np.ndarray
        BGR image (for display).
    axes_info : AxesInfo
        Auto-detected axes (modified in-place if user corrects values).
    on_confirm : Callable
        Called with no arguments when the user clicks "Confirm & Extract".
    """
    st.subheader("Step 2 — Verify Axis Calibration")
    st.caption(
        "Review the automatically detected tick values. "
        "If OCR misread a label, correct it below before extracting."
    )

    # Show axes overlay
    _render_axes_overlay(img, axes_info)

    with st.expander("X-axis ticks", expanded=True):
        _render_tick_editor(axes_info.x_axis.ticks, axis_label="X")

    with st.expander("Y-axis ticks", expanded=True):
        _render_tick_editor(axes_info.y_axis.ticks, axis_label="Y")

    # Scale info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("X-axis scale", axes_info.x_axis.scale_type.value,
                  delta=f"R²={axes_info.x_axis.scale_r2:.4f}")
    with col2:
        st.metric("Y-axis scale", axes_info.y_axis.scale_type.value,
                  delta=f"R²={axes_info.y_axis.scale_r2:.4f}")

    if st.button("✅ Confirm & Extract Data", type="primary", use_container_width=True):
        st.session_state.calibration_done = True
        on_confirm()
        st.rerun()


def _render_axes_overlay(img: np.ndarray, axes_info: AxesInfo) -> None:
    """Draw axis lines and tick positions on the image for visual verification."""
    overlay = img.copy()
    h, w = overlay.shape[:2]

    # Draw x-axis line
    xrow = axes_info.x_axis.line_pixel
    cv2.line(overlay, (0, xrow), (w, xrow), (0, 200, 0), 2)

    # Draw y-axis line
    ycol = axes_info.y_axis.line_pixel
    cv2.line(overlay, (ycol, 0), (ycol, h), (0, 200, 0), 2)

    # Draw detected x-ticks
    for tk in axes_info.x_axis.ticks:
        x = int(tk.pixel_pos)
        cv2.line(overlay, (x, xrow - 8), (x, xrow + 8), (0, 100, 255), 2)

    # Draw detected y-ticks
    for tk in axes_info.y_axis.ticks:
        y = int(tk.pixel_pos)
        cv2.line(overlay, (ycol - 8, y), (ycol + 8, y), (0, 100, 255), 2)

    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption="Detected axes (green) and ticks (orange)",
             use_container_width=True)


def _render_tick_editor(ticks: list[TickInfo], axis_label: str) -> None:
    """Show each tick's OCR value and allow the user to correct it."""
    if not ticks:
        st.info(f"No {axis_label}-axis ticks detected.")
        return

    for i, tk in enumerate(ticks):
        conf_color = "🟢" if tk.ocr_confidence >= 0.7 else "🔴"
        col_a, col_b = st.columns([3, 1])
        with col_a:
            new_val = st.number_input(
                f"{conf_color} Tick at pixel {tk.pixel_pos} "
                f"(OCR: '{tk.raw_text}', conf={tk.ocr_confidence:.2f})",
                value=float(tk.label_value),
                format="%.4g",
                key=f"tick_{axis_label}_{i}",
                label_visibility="visible",
            )
            tk.label_value = new_val
        with col_b:
            st.caption(f"px {tk.pixel_pos}")
