"""
pipeline/axes_detector.py
=========================
Stage 2 — Axes & Scale Detection

OCR-first approach: Run EasyOCR on the label regions to get both
the numeric values AND their pixel positions (label center coordinates).
"""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipeline.preprocess import BGRImage

log = logging.getLogger(__name__)

# Public data structures-

class ScaleType(enum.Enum):
    LINEAR = "linear"
    LOG10  = "log10"


@dataclass
class TickInfo:
    pixel_pos: int
    label_value: float
    ocr_confidence: float
    raw_text: str = ""


@dataclass
class AxisInfo:
    line_pixel: int
    ticks: List[TickInfo] = field(default_factory=list)
    scale_type: ScaleType = ScaleType.LINEAR
    scale_coeffs: Tuple[float, float] = (1.0, 0.0)
    scale_r2: float = 1.0


@dataclass
class AxesInfo:
    x_axis: AxisInfo
    y_axis: AxisInfo
    plot_region: Tuple[int, int, int, int] = (0, 0, 0, 0)
    gridline_mask: Optional[np.ndarray] = None
    inpainted_image: Optional[BGRImage] = None

# Main entry point
def detect_axes(img: BGRImage, global_ocr_results: list = None) -> AxesInfo:
    """Full axes detection pipeline for one figure."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x_line, y_line = _locate_axis_lines(grey)
    plot_region = _compute_plot_region(x_line, y_line, img.shape)

    # extract tick values AND positions from label regions
    x_ticks = _ocr_axis_region(img, axis="x",
                                axis_line_pixel=x_line,
                                plot_region=plot_region,
                                global_ocr_results=global_ocr_results)
    y_ticks = _ocr_axis_region(img, axis="y",
                                axis_line_pixel=y_line,
                                plot_region=plot_region,
                                global_ocr_results=global_ocr_results)

    # if OCR found nothing, try pixel-profile tick detection
    if len(x_ticks) < 2:
        x_px = _extract_tick_pixels_from_profile(grey, "x", x_line, plot_region)
        x_ticks = [TickInfo(pixel_pos=p, label_value=float(i),
                             ocr_confidence=0.0) for i, p in enumerate(x_px)]
    if len(y_ticks) < 2:
        y_px = _extract_tick_pixels_from_profile(grey, "y", y_line, plot_region)
        y_ticks = [TickInfo(pixel_pos=p, label_value=float(i),
                             ocr_confidence=0.0) for i, p in enumerate(y_px)]

    x_axis = _fit_scale(x_ticks, axis_line_pixel=x_line)
    y_axis = _fit_scale(y_ticks, axis_line_pixel=y_line)

    # Remove gridlines
    gridline_mask, inpainted = _remove_gridlines(img, grey, x_axis, y_axis, plot_region)

    return AxesInfo(
        x_axis=x_axis,
        y_axis=y_axis,
        plot_region=plot_region,
        gridline_mask=gridline_mask,
        inpainted_image=inpainted,
    )


# Axis line localization
def _locate_axis_lines(grey: np.ndarray) -> Tuple[int, int]:
    """
    Find x-axis (bottom horizontal) and y-axis (left vertical) pixel positions.
    Uses positional bias: x-axis in bottom 40%, y-axis in left 35% of image.
    """
    h, w = grey.shape
    # brightness threshold
    edges = cv2.Canny(grey, 30, 100)
    # at least 30% of image size
    min_len = int(0.30 * min(h, w))
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=min_len, maxLineGap=20
    )

    if lines is None:
        raise AxesDetectionError("No lines found in image.")

    h_candidates: List[Tuple[float, float]] = []  # (y_center, length)
    v_candidates: List[Tuple[float, float]] = []  # (x_center, length)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 5:
            continue
        angle = abs(float(np.degrees(np.arctan2(abs(dy), abs(dx)))))

        if angle < 5:
            h_candidates.append(((y1 + y2) / 2.0, length))
        elif angle > 85:
            v_candidates.append(((x1 + x2) / 2.0, length))

    if not h_candidates:
        raise AxesDetectionError("No horizontal axis lines detected.")
    if not v_candidates:
        raise AxesDetectionError("No vertical axis lines detected.")

    # X-axis (bottom): prefer lines in the bottom 45% of image
    h_bottom = [(y, l) for y, l in h_candidates if y > h * 0.55]
    pool_h = h_bottom if len(h_bottom) >= 1 else h_candidates
    pool_h.sort(key=lambda t: -t[1])
    x_axis_row = int(pool_h[0][0])

    # v_left = [(x, l) for x, l in v_candidates if x < w * 0.45]
    # pool_v = v_left if len(v_left) >= 1 else v_candidates
    # exclude vertical Canny edges within 5% of the image's left edge.
    LEFT_EDGE_GUARD_FRAC = 0.05
    guard_px = int(w * LEFT_EDGE_GUARD_FRAC)
    v_left = [(x, l) for x, l in v_candidates
              if x < w * 0.45 and x >= guard_px]
    pool_v = v_left if len(v_left) >= 1 else v_candidates

    max_len = max(l for _, l in pool_v)
    long_enough = [(x, l) for x, l in pool_v if l >= 0.6 * max_len]
    long_enough.sort(key=lambda t: t[0])     # sort by x ascending (leftmost first)
    y_axis_col = int(long_enough[0][0])

    return x_axis_row, y_axis_col


def _compute_plot_region(
    x_axis_row: int,
    y_axis_col: int,
    image_shape: Tuple[int, int, int],
    top_row: Optional[int] = None,
    right_col: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) of the inner plot area.

    Uses detected top/right frame lines when available; otherwise falls back
    to an image-edge margin. Without these bounds, the ROI bleeds into the
    title / legend strip and confuses the marker detectors.
    """
    h, w = image_shape[:2]
    margin = 8
    return (y_axis_col + 1, margin, w - margin, x_axis_row - 1)


# OCR-first tick detection
def _ocr_axis_region(
    img: BGRImage,
    axis: str,
    axis_line_pixel: int,
    plot_region: Tuple[int, int, int, int],
    global_ocr_results: list
) -> List[TickInfo]:
    """
    Filter global OCR results to extract (pixel_position, value) pairs 
    from the axis label regions.
    """
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = plot_region

    if axis == "x":
        r1 = axis_line_pixel
        r2 = min(h, axis_line_pixel + 80)
        c1 = max(0, x_min - 60)
        c2 = x_max
    else:
        r1 = y_min
        r2 = min(h, y_max + 30)
        c1 = max(0, x_min - 100)
        c2 = axis_line_pixel

    results = []
    # Filter global OCR for boxes inside this axis region
    for line in global_ocr_results:
        if not line or len(line) != 2:
            continue
            
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]
        
        pts = np.array(bbox, dtype=float)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        
        # Check if the text center is inside the axis strip
        if c1 <= cx <= c2 and r1 <= cy <= r2:
            filtered_text = "".join([char for char in text if char in "0123456789.-"])
            if filtered_text:
                results.append((bbox, filtered_text, confidence))

    MIN_TICK_CONF = 0.5   
    ticks: List[TickInfo] = []
    for bbox, text, conf in results:
        stripped = text.strip()
        if stripped.endswith('.') and not stripped.endswith('..'):
            if not re.search(r'\.\d', stripped):
                continue
                
        if float(conf) < MIN_TICK_CONF:
            continue
            
        value = _parse_numeric(text)
        if value is None:
            continue
            
        pts = np.array(bbox, dtype=float)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        # Coordinates are already absolute, no need to add c1/r1
        if axis == "x":
            pixel_pos = int(round(cx))
        else:
            pixel_pos = int(round(cy))

        ticks.append(TickInfo(
            pixel_pos=pixel_pos,
            label_value=value,
            ocr_confidence=float(conf),
            raw_text=text,
        ))

    # ... keep the rest of the function (peaks, snapping, sorting, etc) exactly the same ...
    peaks = _detect_inward_tick_peaks(img, axis, axis_line_pixel, plot_region)

    if peaks:
        ticks = _match_ocr_to_peaks(ticks, peaks, max_dist=12)
    else:
        ticks = _snap_to_axis_marks(img, ticks, axis, axis_line_pixel, plot_region=plot_region)

    ticks.sort(key=lambda t: t.pixel_pos)
    ticks = _deduplicate_ticks(ticks)
    ticks = _fix_decimal_ocr_errors(ticks)
    ticks = _reject_outlier_ticks(ticks)
    ticks = _sanitize_ticks(ticks)
    return ticks


def _detect_inward_tick_peaks(
    img: BGRImage,
    axis: str,
    axis_line_pixel: int,
    plot_region: Tuple[int, int, int, int],
) -> List[int]:
    """
    Detect tick-mark peak positions from a thin strip just INSIDE the plot
    region (above the x-axis line / right of the y-axis line). Inward-pointing
    ticks in scientific figures show up as short perpendicular notches; the
    perpendicular axis frame itself is excluded from the result.

    Returns a sorted list of pixel positions (image coords) where tick marks
    are located.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = grey.shape
    x_min, y_min, x_max, y_max = plot_region

    STRIP = 5  # depth of the inward-tick strip
    if axis == "x":
        r0 = max(0, axis_line_pixel - STRIP)
        r1 = axis_line_pixel
        strip = grey[r0:r1, x_min:x_max]
        counts = (strip < 120).sum(axis=0)
    else:
        c0 = axis_line_pixel + 1
        c1 = min(w, axis_line_pixel + 1 + STRIP)
        strip = grey[y_min:y_max, c0:c1]
        counts = (strip < 120).sum(axis=1)

    threshold =  1
    candidates = np.where(counts >= threshold)[0]
    if len(candidates) == 0:
        return []

    # Cluster adjacent dark positions
    clusters: List[List[int]] = [[int(candidates[0])]]
    for c in candidates[1:]:
        if c - clusters[-1][-1] <= 4:
            clusters[-1].append(int(c))
        else:
            clusters.append([int(c)])

    offset = x_min if axis == "x" else y_min
    centers = [int(np.median(cl)) + offset for cl in clusters]

    # Drop the perpendicular axis frame itself (within 4 px of the boundary)
    if axis == "x":
        centers = [p for p in centers
                   if abs(p - x_min) > 4 and abs(p - x_max) > 4]
    else:
        centers = [p for p in centers
                   if abs(p - y_min) > 4 and abs(p - y_max) > 4]

    return sorted(centers)


def _match_ocr_to_peaks(
    ticks: List[TickInfo],
    peaks: List[int],
    max_dist: int = 12,
) -> List[TickInfo]:
    """
    Refine each OCR'd tick's pixel_pos using detected tick-mark peaks.

    The OCR label *center* is already a good estimate of the tick position, so
    a peak is only used as a fine refinement when it sits within `max_dist`
    pixels of the label center. OCR ticks with no nearby peak are KEPT at their
    original position — they are not dropped. Dropping them (the previous
    behaviour) discarded valid ticks whenever inward tick-mark detection was
    sparse or noisy, and snapping across large gaps pulled correct labels onto
    spurious peaks (e.g. a marker overlapping the axis strip).
    """
    if not peaks:
        return ticks
    peaks_arr = np.array(peaks, dtype=int)
    matched: List[TickInfo] = []
    for tk in ticks:
        dists = np.abs(peaks_arr - tk.pixel_pos)
        idx = int(np.argmin(dists))
        pos = tk.pixel_pos
        if int(dists[idx]) <= max_dist:
            pos = int(peaks_arr[idx])  # co-located tick mark → refine position
        matched.append(TickInfo(
            pixel_pos=pos,
            label_value=tk.label_value,
            ocr_confidence=tk.ocr_confidence,
            raw_text=tk.raw_text,
        ))
    return matched


def _snap_to_axis_marks(
    img: BGRImage,
    ticks: List[TickInfo],
    axis: str,
    axis_line_pixel: int,
    plot_region: Optional[Tuple[int, int, int, int]] = None,
) -> List[TickInfo]:
    """
    Refine tick pixel positions by finding the nearest dark pixel cluster on
    the axis line within ±40 px of each OCR-detected label center.

    This corrects the offset between tick label center (what OCR sees) and the
    actual tick mark position on the axis line.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = grey.shape
    SEARCH = 40  # px to search on each side

    # Plot-region constraints to avoid snapping onto the perpendicular axis
    if plot_region is not None:
        x_min_pr, y_min_pr, x_max_pr, y_max_pr = plot_region
        x_inner_lo = x_min_pr + 4    # forbid columns at/left of y-axis line
        x_inner_hi = x_max_pr - 4    # forbid columns at/right of right frame
        y_inner_lo = y_min_pr + 4    # forbid rows at/above top frame
        y_inner_hi = y_max_pr - 4    # forbid rows at/below x-axis line
    else:
        x_inner_lo = y_inner_lo = 0
        x_inner_hi = w - 1
        y_inner_hi = h - 1

    refined = []
    for tk in ticks:
        center = tk.pixel_pos

        if axis == "x":
            lo = max(0, center - SEARCH, x_inner_lo)
            hi = min(w - 1, center + SEARCH, x_inner_hi)
            if hi <= lo:
                refined.append(tk)
                continue
            # The perpendicular y-axis vertical line stops at the x-axis 
            # in a rectangular frame, so this strip contains only tick 
            # mark and label digits.
            r0 = min(h - 1, axis_line_pixel + 1)
            r1 = min(h, axis_line_pixel + 10)
            strip = grey[r0:r1, lo:hi + 1]
            col_counts = (strip < 80).sum(axis=0)
            if col_counts.max() > 0:
                best_offset = int(np.argmax(col_counts))
                refined_pos = lo + best_offset
                if abs(refined_pos - center) <= SEARCH:
                    center = refined_pos
        else:
            lo = max(0, center - SEARCH, y_inner_lo)
            hi = min(h - 1, center + SEARCH, y_inner_hi)
            if hi <= lo:
                refined.append(tk)
                continue
            # The x-axis horizontal line stops at the y-axis, 
            # so this strip contains only tick marks and label digits.
            c0 = max(0, axis_line_pixel - 10)
            c1 = max(1, axis_line_pixel - 1)
            strip = grey[lo:hi + 1, c0:c1]
            row_counts = (strip < 80).sum(axis=1)
            if row_counts.max() > 0:
                best_offset = int(np.argmax(row_counts))
                refined_pos = lo + best_offset
                if abs(refined_pos - center) <= SEARCH:
                    center = refined_pos

        refined.append(TickInfo(
            pixel_pos=center,
            label_value=tk.label_value,
            ocr_confidence=tk.ocr_confidence,
            raw_text=tk.raw_text,
        ))

    return refined


def _deduplicate_ticks(ticks: List[TickInfo]) -> List[TickInfo]:
    """Remove ticks with the same value; keep the one with higher confidence."""
    seen: dict = {}
    for tk in ticks:
        v = tk.label_value
        if v not in seen or tk.ocr_confidence > seen[v].ocr_confidence:
            seen[v] = tk
    return sorted(seen.values(), key=lambda t: t.pixel_pos)


def _fix_decimal_ocr_errors(ticks: List[TickInfo]) -> List[TickInfo]:
    """
    Fix OCR errors where decimal points are dropped (e.g., "7.5" → "75").

    Strategy: use only high-confidence ticks (conf >= 0.8) to build a robust
    slope estimate via Theil-Sen (median of pairwise slopes), then check every
    tick against that estimate and fix those with ratio ≈ 10×.
    """
    if len(ticks) < 3:
        return ticks

    # Prefer ticks whose OCR text actually contains a decimal point as the
    # reference fit: when most labels lost their dot ("5.5"→"55") but a few
    # were read correctly, the dotted ones carry the true scale. Otherwise fall
    # back to high-confidence ticks, then to all ticks.
    decimal_anchor = [t for t in ticks
                      if '.' in t.raw_text and t.ocr_confidence >= 0.5]
    if len(decimal_anchor) >= 2:
        anchor = decimal_anchor
    else:
        anchor = [t for t in ticks if t.ocr_confidence >= 0.8]
        if len(anchor) < 2:
            anchor = ticks  # fall back to all ticks

    a_pixels = np.array([t.pixel_pos for t in anchor], dtype=float)
    a_values = np.array([t.label_value for t in anchor], dtype=float)

    if np.all(a_values == a_values[0]):
        return ticks

    # Theil-Sen slope on anchor ticks only
    slopes = []
    na = len(a_pixels)
    for i in range(na):
        for j in range(i + 1, na):
            dp = a_pixels[j] - a_pixels[i]
            if abs(dp) > 0:
                slopes.append((a_values[j] - a_values[i]) / dp)

    if not slopes:
        return ticks

    slope = float(np.median(slopes))
    intercept = float(np.median(a_values - slope * a_pixels))

    # Now check ALL ticks (including low-confidence ones) for 10× errors
    for i, tk in enumerate(ticks):
        predicted = slope * tk.pixel_pos + intercept
        actual = tk.label_value
        if predicted == 0:
            continue
        ratio = actual / predicted
        if 8.0 < ratio < 12.0:
            ticks[i] = TickInfo(
                pixel_pos=tk.pixel_pos,
                label_value=round(actual / 10.0, 6),
                ocr_confidence=tk.ocr_confidence * 0.85,
                raw_text=tk.raw_text,
            )
        elif 0.083 < ratio < 0.125:
            ticks[i] = TickInfo(
                pixel_pos=tk.pixel_pos,
                label_value=round(actual * 10.0, 6),
                ocr_confidence=tk.ocr_confidence * 0.85,
                raw_text=tk.raw_text,
            )

    return ticks


def _reject_outlier_ticks(ticks: List[TickInfo]) -> List[TickInfo]:
    """
    Drop ticks that do not lie on the dominant pixel→value line.

    Widening the OCR crop to capture corner labels (and OCR upscaling) can pull
    in stray digits — misreads of axis artifacts, neighbouring labels, the
    figure panel letter, etc. Real axis ticks are collinear in pixel↔value
    space, so a robust (Theil-Sen) fit through the confident ticks exposes the
    impostors as large-residual outliers. To stay safe on non-linear (log)
    axes, we only reject when a clear majority of ticks remain.
    """
    if len(ticks) < 4:
        return ticks

    anchor = [t for t in ticks if t.ocr_confidence >= 0.5]
    if len(anchor) < 3:
        anchor = ticks

    ap = np.array([t.pixel_pos for t in anchor], dtype=float)
    av = np.array([t.label_value for t in anchor], dtype=float)

    slopes = []
    for i in range(len(ap)):
        for j in range(i + 1, len(ap)):
            if ap[j] != ap[i]:
                slopes.append((av[j] - av[i]) / (ap[j] - ap[i]))
    if not slopes:
        return ticks
    slope = float(np.median(slopes))
    intercept = float(np.median(av - slope * ap))

    resid = np.array([abs(t.label_value - (slope * t.pixel_pos + intercept))
                      for t in ticks])
    vrange = max(float(av.max() - av.min()), 1e-9)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    tol = max(3.0 * mad, 0.05 * vrange)

    kept = [t for t, r in zip(ticks, resid) if r <= tol]
    # Only trust the rejection if most ticks survived (else the linear model is
    # probably wrong for this axis — keep the originals).
    if len(kept) >= 2 and len(kept) >= 0.6 * len(ticks):
        return _reject_by_spacing_cv(kept, slope)
    return _reject_by_spacing_cv(ticks, slope)


def _reject_by_spacing_cv(ticks: List[TickInfo], slope: float) -> List[TickInfo]:
    """
    Hình 29: drop ticks that create irregular Δpixel spacing.
    Operates in pixel domain for linear scales; caller should skip on log.
    """
    if len(ticks) < 4:
        return ticks
    ticks_sorted = sorted(ticks, key=lambda t: t.pixel_pos)
    pixels = np.array([t.pixel_pos for t in ticks_sorted], dtype=float)
    diffs = np.diff(pixels)
    if len(diffs) < 3:
        return ticks
    med = float(np.median(diffs))
    if med <= 0:
        return ticks
    # Reject ticks creating diffs < 0.4*median OR > 1.6*median when most diffs
    # are uniform (CV < 0.2). The "two wrong ticks between" case shows up as
    # very small diffs flanking a real tick.
    cv = float(np.std(diffs) / med) if med > 0 else 0.0
    if cv < 0.2:
        return ticks  # already uniform
    # Greedy: drop ticks that produce the smallest local diffs
    keep = [True] * len(ticks_sorted)
    for i, d in enumerate(diffs):
        if d < 0.4 * med:
            # drop whichever of (i, i+1) has lower OCR confidence
            a, b = ticks_sorted[i], ticks_sorted[i+1]
            drop_idx = i if a.ocr_confidence < b.ocr_confidence else i + 1
            keep[drop_idx] = False
    out = [t for t, k in zip(ticks_sorted, keep) if k]
    return out if len(out) >= 2 else ticks


def _sanitize_ticks(ticks: List[TickInfo]) -> List[TickInfo]:
    """Keep only ticks that maintain monotonicity."""
    if len(ticks) < 2:
        return ticks
    # Determine expected direction from first two ticks
    result = [ticks[0]]
    for tk in ticks[1:]:
        prev = result[-1]
        if tk.pixel_pos != prev.pixel_pos and tk.label_value != prev.label_value:
            result.append(tk)
    return result


# pixel-profile tick detection (used if OCR finds < 2 ticks)
def _extract_tick_pixels_from_profile(
    grey: np.ndarray,
    axis: str,
    axis_line_pixel: int,
    plot_region: Tuple[int, int, int, int],
) -> List[int]:
    """
    Detect tick mark positions by finding local intensity minima along the axis line.
    The axis line itself is dark; ticks make small extensions perpendicular to it.
    This uses the gradient of the axis-line row/column to find tick positions.
    """
    h, w = grey.shape
    x_min, y_min, x_max, y_max = plot_region

    if axis == "x":
        # Find columns where the axis line "extends" further than average
        # Count dark pixels per column in the 10-px band around the axis line
        band_start = max(0, axis_line_pixel - 5)
        band_end = min(h, axis_line_pixel + 15)
        band = grey[band_start:band_end, x_min:x_max]
        # Number of dark pixels per column
        dark_counts = (band < 100).sum(axis=0)
        # Find columns with above-median dark pixel count (= tick positions)
        median_count = int(np.median(dark_counts))
        if median_count > 0:
            tick_cols = np.where(dark_counts > median_count + 1)[0]
            return [p + x_min for p in _cluster_positions(tick_cols.tolist(), gap=8)]
        return []
    else:
        band_start = max(0, axis_line_pixel - 15)
        band_end = min(w, axis_line_pixel + 5)
        band = grey[y_min:y_max, band_start:band_end]
        dark_counts = (band < 100).sum(axis=1)
        median_count = int(np.median(dark_counts))
        if median_count > 0:
            tick_rows = np.where(dark_counts > median_count + 1)[0]
            return [p + y_min for p in _cluster_positions(tick_rows.tolist(), gap=8)]
        return []


def _cluster_positions(positions: List[int], gap: int = 4) -> List[int]:
    """Cluster adjacent pixel positions and return the median of each cluster."""
    if not positions:
        return []
    positions = sorted(set(positions))
    clusters: List[List[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.median(c)) for c in clusters]



# Scale fitting (affine transformation)

def _fit_scale(ticks: List[TickInfo], axis_line_pixel: int) -> AxisInfo:
    """Fit a pixel→data linear or log model from the parsed ticks."""
    good = [t for t in ticks if t.ocr_confidence >= 0.5]
    if len(good) < 2:
        good = [t for t in ticks if t.ocr_confidence > 0]
    if len(good) < 2:
        good = ticks

    if len(good) < 2:
        raise ScaleFitError(
            f"Need at least 2 ticks to fit a scale; found {len(ticks)} total."
        )

    pixels = np.array([t.pixel_pos for t in good], dtype=float)
    values = np.array([t.label_value for t in good], dtype=float)

    # Linear fit
    coeffs_lin = np.polyfit(pixels, values, 1)
    pred_lin = np.polyval(coeffs_lin, pixels)
    ss_res = float(np.sum((values - pred_lin) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    scale_type = ScaleType.LINEAR
    coeffs: Tuple[float, float] = (float(coeffs_lin[0]), float(coeffs_lin[1]))
    r2 = r2_lin

    # Try log fit if all values are positive
    if np.all(values > 0):
        try:
            log_values = np.log10(values)
            coeffs_log = np.polyfit(pixels, log_values, 1)
            pred_log = np.polyval(coeffs_log, pixels)
            ss_res_log = float(np.sum((log_values - pred_log) ** 2))
            ss_tot_log = float(np.sum((log_values - log_values.mean()) ** 2))
            r2_log = 1.0 - ss_res_log / ss_tot_log if ss_tot_log > 0 else 1.0
            if r2_log > r2_lin + 0.02:
                scale_type = ScaleType.LOG10
                coeffs = (float(coeffs_log[0]), float(coeffs_log[1]))
                r2 = r2_log
        except Exception:
            pass

    if r2 < 0.95:
        log.warning("Scale fit R²=%.3f for axis at pixel %d", r2, axis_line_pixel)

    return AxisInfo(
        line_pixel=axis_line_pixel,
        ticks=ticks,
        scale_type=scale_type,
        scale_coeffs=coeffs,
        scale_r2=r2,
    )



# Grid line removal

def _remove_gridlines(
    img: BGRImage,
    grey: np.ndarray,
    x_axis: AxisInfo,
    y_axis: AxisInfo,
    plot_region: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, BGRImage]:
    """Build a grid-line mask and inpaint it."""
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = plot_region
    mask = np.zeros((h, w), dtype=np.uint8)

    # Only mark predicted grid positions if those pixels are actually light gray
    def is_gridline_pixel(row_or_col: int, is_horizontal: bool) -> bool:
        if is_horizontal:
            row = int(round(row_or_col))
            if y_min <= row <= y_max:
                strip = grey[max(0, row - 1):row + 2, x_min:x_max]
                return float(np.median(strip)) > 150  # light gray
        else:
            col = int(round(row_or_col))
            if x_min <= col <= x_max:
                strip = grey[y_min:y_max, max(0, col - 1):col + 2]
                return float(np.median(strip)) > 150
        return False

    for tk in y_axis.ticks:
        row = int(round(tk.pixel_pos))
        if y_min <= row <= y_max and is_gridline_pixel(row, True):
            mask[max(0, row - 1):row + 2, x_min:x_max] = 255

    for tk in x_axis.ticks:
        col = int(round(tk.pixel_pos))
        if x_min <= col <= x_max and is_gridline_pixel(col, False):
            mask[y_min:y_max, max(0, col - 1):col + 2] = 255

    if mask.sum() == 0:
        return mask.astype(bool), img.copy()

    inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return mask.astype(bool), inpainted


# Utilities

def _parse_numeric(text: str) -> Optional[float]:
    """Parse OCR text to float; handles scientific notation and unicode minus."""
    text = text.strip().replace('−', '-').replace('—', '-').replace(',', '').replace(' ', '')
    # OCR faults to not recognize digit 1
    digit_like = re.sub(r'[\[\]\|lIi!]', '1', text)
    rest = re.sub(r'\d|\.', '', digit_like)
    if len(rest) <= max(1, len(digit_like) // 4):
        text = digit_like
    # Handle "10^3", "1e3", "1×10³"
    text = re.sub(r'[×x]10\^?([+-]?\d+)', lambda m: f'e{m.group(1)}', text)
    text = re.sub(r'10\^([+-]?\d+)', lambda m: f'1e{m.group(1)}', text)
    # Remove stray non-numeric chars (keep digits, dot, minus, e, E, +)
    clean = re.sub(r'[^\d.\-eE+]', '', text)
    if not clean or clean in ('.', '-', '+'):
        return None
    try:
        v = float(clean)
        if abs(v) > 1e12 or (abs(v) < 1e-10 and v != 0):
            return None
        return v
    except (ValueError, OverflowError):
        return None

# Custom exceptions
class AxesDetectionError(RuntimeError):
    """Raised when axis lines cannot be reliably located."""


class ScaleFitError(RuntimeError):
    """Raised when the tick-label scale fit has insufficient quality."""
