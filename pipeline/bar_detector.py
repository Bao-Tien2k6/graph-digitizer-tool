"""
pipeline/bar_detector.py
========================
Stage 4b — Bar Top Detection

Bars in scientific figures are commonly drawn as 3D "cylinders" with a light
highlight running down the centre, and they sit on (and visually merge at) the
x-axis baseline. Contour/connected-component detection therefore either fuses
all bars into one comb-shaped blob or splits each cylinder in two at the
highlight. To be robust we instead profile the plot column-by-column: for every
column we measure the height of the foreground run anchored at the baseline
(bottom of the plot region). Real bars are wide bands of tall base-anchored
runs separated by background gaps; the highlight stays part of the bar because
we treat any non-white pixel as foreground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.axes_detector import AxesInfo
from pipeline.parallel_router import DetectionResult
from pipeline.preprocess import BGRImage


# Module-level parameters
NON_WHITE_MAX_GREY   = 250     # pixels darker than this count as bar foreground
MIN_BAR_HEIGHT_FRAC  = 0.04    # ignore base-anchored runs shorter than this * roi_h
MIN_BAR_WIDTH_PX     = 8       # discard sliver column groups
MIN_BAR_WIDTH_FRAC   = 0.4     # discard groups narrower than this * median bar width
N_SERIES_MAX         = 8


# Public data structure
@dataclass
class BarRegion:
    bar_index: int
    series_id: int
    left_x: float
    right_x: float
    top_y: float
    base_y: float
    rectangularity: float = 1.0
    error_bar_top_y: Optional[float] = None
    error_bar_bot_y: Optional[float] = None

    @property
    def center_x(self) -> float:
        return (self.left_x + self.right_x) / 2.0


# Main entry point
def detect(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray = None) -> DetectionResult:
    """Full bar detection pipeline."""
    roi = _crop_to_plot_region(img, axes.plot_region)
    if roi.size == 0:
        return DetectionResult(chart_type="bar", confidence=0.0)

    bars = _extract_bars_by_column_profile(roi)

    if not bars:
        return DetectionResult(chart_type="bar", confidence=0.0,
                               detector_meta={"n_bars": 0})

    bars = _group_by_series_color(roi, bars)
    bars = _detect_error_bars(roi, bars)
    confidence = _compute_confidence(bars, roi.shape[1])

    x_off = axes.plot_region[0]
    y_off = axes.plot_region[1]
    pixel_points = [
        {
            # x/y are the values the coordinate transform reads: a bar maps to
            # its top-centre (category position, bar height).
            "x": b.center_x + x_off,
            "y": b.top_y + y_off,
            "series": b.series_id,
            "bar_index": b.bar_index,
            "left_x": b.left_x + x_off,
            "right_x": b.right_x + x_off,
            "top_y": b.top_y + y_off,
            "base_y": b.base_y + y_off,
            "error_top_y": (b.error_bar_top_y + y_off
                            if b.error_bar_top_y is not None else None),
            "error_bot_y": (b.error_bar_bot_y + y_off
                            if b.error_bar_bot_y is not None else None),
        }
        for b in bars
    ]

    return DetectionResult(
        chart_type="bar",
        confidence=confidence,
        pixel_points=pixel_points,
        detector_meta={"n_bars": len(bars)},
    )


# Step helpers
def _crop_to_plot_region(img: BGRImage, plot_region: Tuple[int, int, int, int]) -> BGRImage:
    x_min, y_min, x_max, y_max = plot_region
    return img[y_min:y_max, x_min:x_max]


def _base_anchored_heights(roi: BGRImage) -> Tuple[np.ndarray, int]:
    """
    For each column, return the height of the contiguous non-white run anchored
    near the bottom row of the ROI, tolerating small gaps.
    """
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    h, w = grey.shape
    
    # 1. Identify foreground using the more generous threshold
    fg = (grey < NON_WHITE_MAX_GREY).astype(np.uint8)
    
    # 2. Morphological closing to seal internal cracks (like bright highlights) 
    # and bridge tiny gaps at the baseline.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    fg_closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    heights = np.zeros(w, dtype=int)
    for x in range(w):
        col = fg_closed[:, x]
        c = 0
        
        # 3. Flexible Anchoring: Search the bottom 10 pixels for the start of the bar
        # This prevents failure if the crop left a small white gap above the x-axis.
        start_y = h - 1
        gap_allowance = 10
        
        for offset in range(gap_allowance):
            if start_y - offset >= 0 and col[start_y - offset]:
                start_y = start_y - offset
                break
        else:
            # If no foreground is found in the bottom 10 pixels, this is background.
            continue 
            
        # 4. Measure the contiguous run upwards
        for y in range(start_y, -1, -1):
            if col[y]:
                c += 1
            else:
                break
                
        # Add the bottom gap back into the height so top_y coordinates are correct
        heights[x] = c + (h - 1 - start_y)
        
    return heights, h


def _extract_bars_by_column_profile(roi: BGRImage) -> List[BarRegion]:
    """Segment the base-anchored column-height profile into individual bars."""
    heights, roi_h = _base_anchored_heights(roi)
    w = len(heights)

    min_h = MIN_BAR_HEIGHT_FRAC * roi_h
    is_bar_col = heights > min_h

    # Group consecutive bar columns. Require a gap of >= MIN_GAP_PX
    # background columns before closing a group, so neighboring bars whose
    # outlines briefly touch (1-2 px) do not fuse into one.
    MIN_GAP_PX = 3
    groups: List[List[int]] = []
    start: Optional[int] = None
    gap_run = 0
    for x in range(w):
        if is_bar_col[x]:
            if start is None:
                start = x
            gap_run = 0
        else:
            if start is not None:
                gap_run += 1
                if gap_run >= MIN_GAP_PX:
                    groups.append([start, x - gap_run])
                    start = None
                    gap_run = 0
    if start is not None:
        groups.append([start, w - 1])

    groups = [g for g in groups if (g[1] - g[0] + 1) >= MIN_BAR_WIDTH_PX]
    if not groups:
        return []

    # Drop sliver groups (frame edges, residual tick marks) relative to the
    # typical bar width.
    median_w = float(np.median([g[1] - g[0] + 1 for g in groups]))
    groups = [g for g in groups
              if (g[1] - g[0] + 1) >= MIN_BAR_WIDTH_FRAC * median_w]
    if not groups:
        return []

    bars: List[BarRegion] = []
    for i, (x0, x1) in enumerate(groups):
        # Bar top is the highest reach of the band (use a high percentile of the
        # per-column heights so a stray tall column does not dominate, and the
        # error-bar whisker — which only covers the central columns — does not
        # inflate the height across the whole band).
        band = heights[x0:x1 + 1]
        bar_h = float(np.percentile(band, 75))
        top = roi_h - bar_h
        bars.append(BarRegion(
            bar_index=i,
            series_id=0,
            left_x=float(x0),
            right_x=float(x1),
            top_y=top,
            base_y=float(roi_h),
            rectangularity=1.0,
        ))
    return bars


def _group_by_series_color(roi: BGRImage, bars: List[BarRegion]) -> List[BarRegion]:
    """Assign series IDs by bar fill color in LAB space."""
    if len(bars) < 2:
        return bars
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    colors = []
    for b in bars:
        x1 = int(b.left_x) + 3
        x2 = int(b.right_x) - 3
        y1 = int(b.top_y) + 5
        y2 = int(b.base_y) - 5
        if x2 > x1 and y2 > y1:
            patch = lab[y1:y2, x1:x2]
            colors.append(np.median(patch.reshape(-1, 3), axis=0))
        else:
            colors.append(np.array([128.0, 128.0, 128.0]))

    colors = np.array(colors, dtype=float)

    # Decide how many distinct colours are really present: if every bar shares
    # essentially the same fill (the common single-series case), force one
    # series instead of letting KMeans invent clusters from noise.
    spread = float(np.max(np.linalg.norm(colors - colors.mean(axis=0), axis=1)))
    if spread < 12.0:
        for b in bars:
            b.series_id = 0
        return bars

    max_k = min(N_SERIES_MAX, len(bars))
    from sklearn.cluster import KMeans
    best_labels = np.zeros(len(bars), dtype=int)
    best_k, prev_inertia = 1, float("inf")
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(colors)
        if k > 1 and prev_inertia > 0:
            if (prev_inertia - km.inertia_) / prev_inertia < 0.10:
                break
        best_labels, best_k, prev_inertia = labels, k, km.inertia_
    for b, label in zip(bars, best_labels):
        b.series_id = int(label)
    return bars


def _detect_error_bars(roi: BGRImage, bars: List[BarRegion]) -> List[BarRegion]:
    """Detect error bar whiskers above bar tops."""
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for b in bars:
        cx = int(b.center_x)
        search_top = max(0, int(b.top_y) - 25)
        search_bot = int(b.top_y)
        if search_bot <= search_top:
            continue
        strip = grey[search_top:search_bot, max(0, cx - 3):cx + 4]
        if strip.size == 0:
            continue
        col_min = strip.min(axis=1)
        dark_rows = np.where(col_min < 100)[0]
        if len(dark_rows) >= 3:
            b.error_bar_top_y = float(dark_rows[0] + search_top)
            b.error_bar_bot_y = float(dark_rows[-1] + search_top)
    return bars


def _compute_confidence(bars: List[BarRegion], roi_w: int) -> float:
    """
    Confidence that this figure is a bar chart.

    A genuine bar chart shows several bars of similar width, regularly spaced.
    Score on bar count (saturating at ~5), width consistency and spacing
    regularity so that a clean bar chart scores high enough to win routing
    against the line detector's skeleton-pixel confidence, while incidental
    column bands on a line/scatter plot stay low.
    """
    n = len(bars)
    if n < 2:
        return 0.0

    widths = np.array([b.right_x - b.left_x + 1 for b in bars], dtype=float)
    centers = np.array(sorted(b.center_x for b in bars), dtype=float)

    width_cv = float(np.std(widths) / max(np.mean(widths), 1.0))
    width_score = max(0.0, 1.0 - width_cv)

    if n >= 3:
        gaps = np.diff(centers)
        gap_cv = float(np.std(gaps) / max(np.mean(gaps), 1.0))
        spacing_score = max(0.0, 1.0 - gap_cv)
    else:
        spacing_score = 0.5

    count_score = min(1.0, n / 5.0)

    return float(count_score * (0.5 * width_score + 0.5 * spacing_score))
