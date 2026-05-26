"""
pipeline/line_detector.py
=========================
Stage 4c — Line Marker Detection

Two-branch architecture (evaluation finding):
  Branch A — monochrome markers: skeleton → local width spikes → centroids (main)
  Branch B — distinct-color markers: isolated blob detection on separate color mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from pipeline.axes_detector import AxesInfo
from pipeline.parallel_router import DetectionResult
from pipeline.preprocess import BGRImage
from pipeline.text_mask import build_text_mask

# Module-level parameters
N_SERIES_MAX           = 8
HUE_BANDWIDTH_DEG      = 20
MARKER_WIDTH_RATIO     = 1.6   
MARKER_WIDTH_RATIO_MIN = 1.25  # Raised back to 1.25 to ignore line-aliasing thickness variations
MIN_MARKER_AREA_PX2    = 20
MIN_HOLE_AREA_PX2      = 4     # Minimum size of an enclosed loop for hollow markers
MAX_GAP_PX             = 15
CONFIDENCE_SATURATION  = 5000
BOUNDING_THICKNESS = 20

# Minimum skeleton pixels to consider a color mask a "continuous line"
CONTINUOUS_LINE_MIN_PX = 60
# Minimum pixels in isolated blob to be a marker
ISOLATED_BLOB_MIN_PX   = 15

# Public data structure
@dataclass
class LineMarker:
    x_px: float
    y_px: float
    series_id: int = 0
    skeleton_arc_length: float = 0.0
    blob_area_px2: float = 50.0


# Main entry point
def detect(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray = None) -> DetectionResult:
    """Full line-plot marker detection pipeline with two-branch architecture."""
    roi = _crop_to_plot_region(img, axes.plot_region)
    if roi.size == 0:
        return DetectionResult(chart_type="line", confidence=0.0)

    # remove text pixels before color segmentation (uses pre-built mask from router)
    x0, y0, x1, y1 = axes.plot_region
    if text_mask is None:
        text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    text_mask_roi = text_mask[y0:y1, x0:x1]
    roi = roi.copy()
    roi[text_mask_roi > 0] = 255
    
    # --- NEW: Mask out the plot frame bounding box so ticks/lines aren't detected ---
    # h, w = roi.shape[:2]
    # cv2.rectangle(roi, (0, 0), (w-1, h-1), (255, 255, 255), thickness=BOUNDING_THICKNESS)
    h, w = roi.shape[:2]
    t = BOUNDING_THICKNESS
    half = t // 2
    cv2.rectangle(roi, (half, half), (w - 1 - half, h - 1 - half),
                  (255, 255, 255), thickness=t)

    color_masks = _segment_by_color(roi)
    if not color_masks:
        return DetectionResult(chart_type="line", confidence=0.0,
                               detector_meta={"n_markers": 0, "n_series": 0,
                                              "total_skeleton_px": 0})

    all_markers: List[LineMarker] = []
    total_skel_px = 0
    connected_skel_px = 0

    for series_id, mask in enumerate(color_masks):
        if _is_continuous_line(mask):
            # Branch A: skeletonize + width-spike marker extraction
            skel = _skeletonize_mask(mask)
            skel_bridged, gap_px = _bridge_gaps(skel)

            markers = _extract_markers_from_skeleton(skel_bridged, mask, series_id)
            
            # # Extract line endpoints (fixes beginning/ending squares) ---
            # endpoints = _extract_endpoints_from_skeleton(skel_bridged, mask, series_id)
            # markers.extend(endpoints)

            # Add hollow markers (enclosed loops missed by width-spike detection)
            hollow_markers = _detect_hollow_markers(mask, series_id)
            markers.extend(hollow_markers)

            sk_sum = int(skel_bridged.sum())
            total_skel_px += sk_sum
            connected_skel_px += max(0, sk_sum - gap_px)
        else:
            # Branch B: isolated blobs → direct centroid detection
            markers = _detect_isolated_blobs(mask, series_id)
            # Approximate skeleton contribution for confidence scoring
            approx_skel = len(markers) * 15
            total_skel_px += approx_skel
            connected_skel_px += approx_skel

        all_markers.extend(markers)

    # Deduplicate markers that are spatially very close (< 15 px)
    all_markers = _deduplicate_markers(all_markers, threshold_px=15)
    # Merge clusters that are actually one curve
    _merge_interleaved_line_series(all_markers)
    # NEW: drop whisker-cap artifacts vertically stacked with real markers
    all_markers = _dedupe_x_column(all_markers, x_tol=10.0)
    # Re-number series IDs to be contiguous starting from 0
    _renumber_series(all_markers)

    confidence = _compute_confidence(total_skel_px, connected_skel_px)

    x_off = axes.plot_region[0]
    y_off = axes.plot_region[1]
    pixel_points = [
        {
            "series": m.series_id,
            "x": m.x_px + x_off,
            "y": m.y_px + y_off,
            "arc_length": m.skeleton_arc_length,
            "area": m.blob_area_px2,
        }
        for m in all_markers
    ]

    return DetectionResult(
        chart_type="line",
        confidence=confidence,
        pixel_points=pixel_points,
        detector_meta={
            "n_markers": len(all_markers),
            "n_series": len(color_masks),
            "total_skeleton_px": total_skel_px,
        },
    )

def _detect_hollow_markers(color_mask: np.ndarray, series_id: int) -> List[LineMarker]:
    """Find enclosed loops (hollow markers) in the mask filter Hollow Markers via circularity and inertia"."""
    hollow_markers = []
    
    # RETR_CCOMP extracts internal holes as child contours
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return hollow_markers
        
    for i, cnt in enumerate(contours):
        # hierarchy[0][i][3] holds the parent ID. If it's not -1, this contour is a hole!
        if hierarchy[0][i][3] != -1:
            area = cv2.contourArea(cnt)
            if area >= MIN_HOLE_AREA_PX2:
                # --- NEW: Filter out jagged JPEG artifacts using circularity and aspect ratio ---
                perim = cv2.arcLength(cnt, True)
                if perim == 0:
                    continue
                circularity = 4 * np.pi * (area / (perim * perim))
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(h, 1)
                
                # Real markers have holes that are relatively round or square (aspect ratio ~1.0)
                # Artifacts are usually jagged slivers (low circularity, extreme aspect ratios)
                if circularity > 0.75 and 0.4 <= aspect_ratio <= 2.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        hollow_markers.append(LineMarker(
                            x_px=cx, y_px=cy,
                            series_id=series_id,
                            blob_area_px2=area * 3, # Rough estimate of the outer marker area
                        ))
    return hollow_markers


def _segment_by_color(roi: BGRImage) -> List[np.ndarray]:
    """
    Produce binary masks for each foreground color series.
    Includes a fallback for black/monochrome scientific plots.
    """

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Standard Foreground mask (ignores pure white and pure black)
    fg_mask = cv2.inRange(hsv,
                          np.array([0,  40,  40], dtype=np.uint8),
                          np.array([179, 255, 240], dtype=np.uint8))
    
    dark_mask = cv2.inRange(hsv,
                            np.array([0, 0, 0], dtype=np.uint8),
                            np.array([179, 255, 60], dtype=np.uint8))
    
    fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(dark_mask))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    masks: List[np.ndarray] = []

    # 1. Attempt standard color clustering
    if fg_mask.sum() > 0:
        h_channel = hsv[:, :, 0]
        fg_hues = h_channel[fg_mask > 0].astype(float)
        
        hist, bin_edges = np.histogram(fg_hues, bins=36, range=(0, 180))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        hue_centers = _find_hue_peaks(hist, bin_centers, min_count=30)

        for hue_center in hue_centers[:N_SERIES_MAX]:
            hw = HUE_BANDWIDTH_DEG // 2
            lo = int(round(hue_center - hw))
            hi = int(round(hue_center + hw))

            if lo < 0:
                m1 = cv2.inRange(h_channel, int(lo + 180), 180)
                m2 = cv2.inRange(h_channel, 0, int(hi))
                color_mask = cv2.bitwise_or(m1, m2)
            elif hi > 179:
                m1 = cv2.inRange(h_channel, int(lo), 179)
                m2 = cv2.inRange(h_channel, 0, int(hi - 180))
                color_mask = cv2.bitwise_or(m1, m2)
            else:
                color_mask = cv2.inRange(h_channel, int(lo), int(hi))

            color_mask = cv2.bitwise_and(color_mask, fg_mask)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

            if color_mask.sum() > 50:
                masks.append(color_mask)

    # 2. MONOCHROME FALLBACK: If no colors found, or image is mostly grayscale
    s_channel = hsv[:, :, 1]
    median_sat = np.median(s_channel)
    
    if len(masks) == 0 or median_sat < 15:
        # Isolate black/dark-gray pixels
        gray_mask = cv2.inRange(hsv, 
                                np.array([0, 0, 0]), 
                                np.array([180, 40, 130])) 
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

        if gray_mask.sum() > 50:
            masks.append(gray_mask)

    return masks


def _find_hue_peaks(hist: np.ndarray, bin_centers: np.ndarray,
                    min_count: int = 30) -> List[float]:
    """Find dominant hue values from a histogram."""
    peaks: List[float] = []
    used = np.zeros(len(hist), dtype=bool)

    for _ in range(N_SERIES_MAX):
        if hist[~used].max() < min_count:
            break
        masked_hist = hist.copy()
        masked_hist[used] = 0
        idx = int(np.argmax(masked_hist))
        peaks.append(float(bin_centers[idx]))
        hw = max(1, HUE_BANDWIDTH_DEG // 10)
        lo = max(0, idx - hw)
        hi = min(len(hist), idx + hw + 1)
        used[lo:hi] = True

    return peaks


# Branch A: continuous line → skeleton + width spikes
def _is_continuous_line(mask: np.ndarray) -> bool:
    """Returns True if the mask contains a long continuous skeleton."""
    skel = skeletonize(mask.astype(bool))
    skel_u8 = skel.astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(skel_u8)
    if n_labels <= 1:
        return False
    for label_id in range(1, n_labels):
        component_size = int((labels == label_id).sum())
        if component_size >= CONTINUOUS_LINE_MIN_PX:
            return True
    return False


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce the binary mask to a 1-pixel-wide skeleton."""
    return skeletonize(mask.astype(bool))


def _bridge_gaps(skeleton: np.ndarray) -> Tuple[np.ndarray, int]:
    """Bridge short gaps (≤ MAX_GAP_PX) in the skeleton for dashed lines."""
    skel_u8 = skeleton.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel_u8)

    bridged = skel_u8.copy()
    gap_px = 0

    if n_labels <= 2:
        return bridged.astype(bool), 0

    component_pixels: List[np.ndarray] = []
    for i in range(1, n_labels):
        px = np.argwhere(labels == i) 
        component_pixels.append(px)

    for i in range(len(component_pixels)):
        for j in range(i + 1, len(component_pixels)):
            pi = component_pixels[i]
            pj = component_pixels[j]
            if len(pi) > 200:
                pts_i = pi[np.linspace(0, len(pi) - 1, 50, dtype=int)]
            else:
                pts_i = pi
            if len(pj) > 200:
                pts_j = pj[np.linspace(0, len(pj) - 1, 50, dtype=int)]
            else:
                pts_j = pj

            min_dist = float('inf')
            best_pi, best_pj = pts_i[0], pts_j[0]
            for pi_pt in pts_i:
                dists = np.sqrt(((pts_j - pi_pt) ** 2).sum(axis=1))
                idx = np.argmin(dists)
                if dists[idx] < min_dist:
                    min_dist = dists[idx]
                    best_pi = pi_pt
                    best_pj = pts_j[idx]

            if min_dist <= MAX_GAP_PX:
                r1, c1 = int(best_pi[0]), int(best_pi[1])
                r2, c2 = int(best_pj[0]), int(best_pj[1])
                cv2.line(bridged, (c1, r1), (c2, r2), 1, 1)
                gap_px += int(min_dist)

    return bridged.astype(bool), gap_px


def _extract_markers_from_skeleton(
    skeleton: np.ndarray,
    color_mask: np.ndarray,
    series_id: int,
) -> List[LineMarker]:
    """Detect markers via local width spikes (Harris corners removed)."""
    skel_u8 = skeleton.astype(np.uint8)
    dist = cv2.distanceTransform(color_mask, cv2.DIST_L2, 5)

    skel_pixels = np.argwhere(skel_u8 > 0)
    if len(skel_pixels) == 0:
        return []

    widths = dist[skel_pixels[:, 0], skel_pixels[:, 1]]
    if len(widths) == 0:
        return []

    median_width = float(np.median(widths))
    if median_width < 0.5:
        median_width = 0.5

    spike_mask = np.zeros_like(skel_u8)

    # 1. Width spikes ONLY
    for ratio in (MARKER_WIDTH_RATIO, MARKER_WIDTH_RATIO_MIN):
        spike_pixels = skel_pixels[widths > ratio * median_width]
        if len(spike_pixels) > 0:
            break
    for r, c in spike_pixels:
        spike_mask[r, c] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spike_dilated = cv2.dilate(spike_mask, kernel)
    spike_dilated = cv2.bitwise_and(spike_dilated, color_mask)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spike_dilated)
    markers: List[LineMarker] = []
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area >= MIN_MARKER_AREA_PX2:
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            markers.append(LineMarker(
                x_px=cx, y_px=cy,
                series_id=series_id,
                blob_area_px2=area,
            ))

    markers.sort(key=lambda m: m.x_px)
    for i, m in enumerate(markers):
        m.skeleton_arc_length = float(i)

    return markers


# Branch B: isolated blobs → direct centroid detection
def _detect_isolated_blobs(mask: np.ndarray, series_id: int) -> List[LineMarker]:
    """Detect discrete data markers as isolated blobs in a color mask."""
    mask_u8 = (mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)
    markers: List[LineMarker] = []
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < ISOLATED_BLOB_MIN_PX:
            continue

        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 1.8:
            continue

        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            (cx, cy), _radius = cv2.minEnclosingCircle(contours[0])
            cx, cy = float(cx), float(cy)
        else:
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

        markers.append(LineMarker(
            x_px=cx, y_px=cy,
            series_id=series_id,
            blob_area_px2=area,
        ))

    markers.sort(key=lambda m: m.x_px)
    for i, m in enumerate(markers):
        m.skeleton_arc_length = float(i)

    return markers


# Utilities
def _deduplicate_markers(
    markers: List[LineMarker],
    threshold_px: float = 8.0,
) -> List[LineMarker]:
    """Remove spatially duplicate markers; keep the one with the largest blob area."""
    if len(markers) <= 1:
        return markers
    coords = np.array([[m.x_px, m.y_px] for m in markers], dtype=float)
    kept = []
    used = np.zeros(len(markers), dtype=bool)
    for i in range(len(markers)):
        if used[i]:
            continue
        dists = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        cluster = np.where((dists < threshold_px) & (~used))[0]
        best = max(cluster, key=lambda j: markers[j].blob_area_px2)
        kept.append(markers[best])
        used[cluster] = True
    return kept

def _dedupe_x_column(
    markers: List[LineMarker],
    x_tol: float = 10.0,
) -> List[LineMarker]:
    """Drop whisker-cap artifacts: when two markers in the same series share
    an x-column (|Δx| <= x_tol), keep only the one with the largest blob area.

    Rationale: error-bar caps produce width spikes on the skeleton directly
    above/below the real marker. They share the data point's x but have far
    smaller area. Euclidean dedupe misses them because the y-gap exceeds
    `_deduplicate_markers`' threshold.
    """
    if len(markers) <= 1:
        return markers

    from collections import defaultdict
    by_series: dict[int, List[LineMarker]] = defaultdict(list)
    for m in markers:
        by_series[m.series_id].append(m)

    kept: List[LineMarker] = []
    for series_markers in by_series.values():
        series_markers.sort(key=lambda m: m.x_px)
        cluster: List[LineMarker] = []
        for m in series_markers:
            if cluster and (m.x_px - cluster[-1].x_px) <= x_tol:
                cluster.append(m)
            else:
                if cluster:
                    kept.append(max(cluster, key=lambda c: c.blob_area_px2))
                cluster = [m]
        if cluster:
            kept.append(max(cluster, key=lambda c: c.blob_area_px2))
    return kept


def _crop_to_plot_region(img: BGRImage, plot_region: Tuple[int, int, int, int]) -> BGRImage:
    x_min, y_min, x_max, y_max = plot_region
    return img[y_min:y_max, x_min:x_max]


def _renumber_series(markers: List[LineMarker]) -> None:
    """Remap series IDs to contiguous 0-based integers."""
    seen: dict = {}
    counter = 0
    for m in markers:
        if m.series_id not in seen:
            seen[m.series_id] = counter
            counter += 1
        m.series_id = seen[m.series_id]

# def _merge_interleaved_line_series(markers: List[LineMarker],
#                                     smooth_tol_ratio: float = 0.25) -> None:
#     """In-place: merge series_ids when two color groups sit on the same path."""
#     if len(markers) < 4:
#         return
#     coords = np.array([[m.x_px, m.y_px] for m in markers])
#     labels = np.array([m.series_id for m in markers])
#     unique = sorted(set(labels.tolist()))
#     changed = True
#     while changed:
#         changed = False
#         for a in unique:
#             for b in unique:
#                 if a >= b:
#                     continue
#                 idx_a = np.where(labels == a)[0]
#                 idx_b = np.where(labels == b)[0]
#                 if len(idx_a) < 2 or len(idx_b) < 2:
#                     continue
#                 merged_idx = np.concatenate([idx_a, idx_b])
#                 order = merged_idx[np.argsort(coords[merged_idx, 0])]
#                 ys = coords[order, 1]
#                 if len(ys) < 4:
#                     continue
#                 d2 = np.abs(np.diff(ys, n=2))
#                 yrange = max(1.0, ys.max() - ys.min())
#                 seq = labels[order]
#                 flips = int(np.sum(seq[:-1] != seq[1:]))
#                 if np.median(d2) <= smooth_tol_ratio * yrange and flips >= 2:
#                     labels[labels == b] = a
#                     changed = True
#                     break
#             if changed:
#                 break
#         unique = sorted(set(labels.tolist()))
#     for m, lb in zip(markers, labels):
#         m.series_id = int(lb)

def _merge_interleaved_line_series(markers: List[LineMarker],
                                    smooth_tol_ratio: float = 0.25,
                                    parallel_gap_ratio: float = 0.3,
                                    same_column_px: float = 30.0) -> None:
    """In-place: merge series_ids when two color groups sit on the same path.

    Refuses to merge when the two series are *parallel* curves (similar shape
    but at different y levels) by checking the median y-gap between markers
    that share an x-column. If that gap is large relative to each series' own
    y-range, the series are stacked, not interleaved.
    """
    if len(markers) < 4:
        return
    coords = np.array([[m.x_px, m.y_px] for m in markers])
    labels = np.array([m.series_id for m in markers])
    unique = sorted(set(labels.tolist()))
    changed = True
    while changed:
        changed = False
        for a in unique:
            for b in unique:
                if a >= b:
                    continue
                idx_a = np.where(labels == a)[0]
                idx_b = np.where(labels == b)[0]
                if len(idx_a) < 2 or len(idx_b) < 2:
                    continue
                merged_idx = np.concatenate([idx_a, idx_b])
                order = merged_idx[np.argsort(coords[merged_idx, 0])]
                ys = coords[order, 1]
                if len(ys) < 4:
                    continue
                d2 = np.abs(np.diff(ys, n=2))
                yrange = max(1.0, ys.max() - ys.min())
                seq = labels[order]
                flips = int(np.sum(seq[:-1] != seq[1:]))

                # NEW: refuse to merge parallel curves (stacked at different y).
                xa, ya = coords[idx_a, 0], coords[idx_a, 1]
                xb, yb = coords[idx_b, 0], coords[idx_b, 1]
                gaps = []
                for xi, yi in zip(xa, ya):
                    j = int(np.argmin(np.abs(xb - xi)))
                    if abs(xb[j] - xi) < same_column_px:
                        gaps.append(abs(yb[j] - yi))
                if gaps:
                    ya_range = max(1.0, ya.max() - ya.min())
                    yb_range = max(1.0, yb.max() - yb.min())
                    own_range = min(ya_range, yb_range)
                    if np.median(gaps) > parallel_gap_ratio * own_range:
                        continue  # parallel curves -> skip merge

                if np.median(d2) <= smooth_tol_ratio * yrange and flips >= 2:
                    labels[labels == b] = a
                    changed = True
                    break
            if changed:
                break
        unique = sorted(set(labels.tolist()))
    for m, lb in zip(markers, labels):
        m.series_id = int(lb)

def _compute_confidence(total_skel_px: int, connected_skel_px: int) -> float:
    if total_skel_px == 0:
        return 0.0
    continuity = min(1.0, connected_skel_px / total_skel_px)
    raw = total_skel_px * continuity
    return float(min(raw / CONFIDENCE_SATURATION, 1.0))