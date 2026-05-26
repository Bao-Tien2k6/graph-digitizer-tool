"""
pipeline/scatter_detector.py
============================
Stage 4a — Scatter Point Detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.axes_detector import AxesInfo
from pipeline.parallel_router import DetectionResult
from pipeline.preprocess import BGRImage
from pipeline.text_mask import build_text_mask


# Module-level parameters
BLOB_MIN_THRESHOLD    = 10
BLOB_MAX_THRESHOLD    = 220
BLOB_MIN_AREA         = 20
BLOB_MAX_AREA         = 5000
BLOB_MIN_CIRCULARITY  = 0.4
BLOB_MIN_CONVEXITY    = 0.70
BLOB_MIN_INERTIA      = 0.20

N_SERIES_MAX          = 8
MIN_SEPARATION_PX     = 5
MIN_SERIES_SIZE       = 3       # drop color clusters with fewer
CONFIDENCE_SATURATION = 200



# Public data structure
@dataclass
class ScatterPoint:
    x_px: float
    y_px: float
    series_id: int = 0
    shape: str = "circle"
    circularity: float = 1.0
    blob_area_px2: float = 100.0


# Main entry point
def detect(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray = None) -> DetectionResult:
    """Full scatter detection pipeline."""
    roi = _crop_to_plot_region(img, axes.plot_region)
    if roi.size == 0:
        return DetectionResult(chart_type="scatter", confidence=0.0)

    # Text exclusion (uses pre-built mask from router; fall back to empty)
    x0, y0, x1, y1 = axes.plot_region
    if text_mask is None:
        text_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    text_mask_roi = text_mask[y0:y1, x0:x1]
    roi_clean = roi.copy()
    roi_clean[text_mask_roi > 0] = 255   # paint text pixels white

    blobs = _detect_blobs(roi_clean)
    blobs = _deduplicate(blobs)
    # blobs = _drop_whisker_caps(blobs)

    if len(blobs) > 1:
        labels = _separate_series_by_color(roi, blobs)
        labels = _merge_interleaved_series(blobs, labels)   # Hình 24
        blobs, labels = _drop_small_color_series(blobs, labels)  # noise filter
        for i, pt in enumerate(blobs):
            pt.series_id = int(labels[i])

    confidence = _compute_confidence(blobs)

    pixel_points = [
        {"series": pt.series_id, "x": pt.x_px + axes.plot_region[0],
         "y": pt.y_px + axes.plot_region[1],
         "shape": pt.shape, "area": pt.blob_area_px2}
        for pt in blobs
    ]

    return DetectionResult(
        chart_type="scatter",
        confidence=confidence,
        pixel_points=pixel_points,
        detector_meta={"n_blobs": len(blobs)},
    )


# Step helpers
def _crop_to_plot_region(img: BGRImage, plot_region: Tuple[int, int, int, int]) -> BGRImage:
    x_min, y_min, x_max, y_max = plot_region
    return img[y_min:y_max, x_min:x_max]


def _detect_blobs(roi: BGRImage) -> List[ScatterPoint]:
    """Detect blobs via SimpleBlobDetector on grayscale inverted image."""
    grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = BLOB_MIN_THRESHOLD
    params.maxThreshold = BLOB_MAX_THRESHOLD
    params.filterByArea = True
    params.minArea = BLOB_MIN_AREA
    params.maxArea = BLOB_MAX_AREA
    params.filterByCircularity = True
    params.minCircularity = BLOB_MIN_CIRCULARITY
    params.filterByConvexity = True
    params.minConvexity = BLOB_MIN_CONVEXITY
    params.filterByInertia = True
    params.minInertiaRatio = BLOB_MIN_INERTIA

    detector = cv2.SimpleBlobDetector_create(params)

    # Try on inverted image (dark blobs on white background)
    inv = cv2.bitwise_not(grey)
    kps = detector.detect(inv)

    # Also try on direct image
    kps2 = detector.detect(grey)
    all_kps = list(kps) + list(kps2)

    points: List[ScatterPoint] = []
    for kp in all_kps:
        x, y = kp.pt
        r = kp.size / 2.0
        area = np.pi * r * r
        # Compute circularity from a circle approximation
        circ = min(1.0, BLOB_MIN_CIRCULARITY + 0.2)  # approximate
        points.append(ScatterPoint(
            x_px=float(x), y_px=float(y),
            blob_area_px2=float(area),
            circularity=circ,
        ))

    # Also use contour-based detection for colored markers
    points += _detect_colored_blobs(roi)
    return points


def _detect_colored_blobs(roi: BGRImage) -> List[ScatterPoint]:
    """Detect colored marker blobs by thresholding non-white, non-black regions."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Mask: non-white (S > 50, V < 240) and non-black (V > 40)
    mask = cv2.inRange(hsv,
                       np.array([0, 50, 40], dtype=np.uint8),
                       np.array([180, 255, 240], dtype=np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    points: List[ScatterPoint] = []
    for i in range(1, n_labels):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (BLOB_MIN_AREA <= area <= BLOB_MAX_AREA):
            continue
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            (cx, cy), _r = cv2.minEnclosingCircle(contours[0])
            cx, cy = float(cx), float(cy)
        else:
            cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Circularity from aspect of the bounding box (cheap proxy)
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        points.append(ScatterPoint(
            x_px=cx, y_px=cy,
            blob_area_px2=area,
            circularity=float(aspect),
        ))
    return points


def _deduplicate(points: List[ScatterPoint]) -> List[ScatterPoint]:
    """Remove spatial duplicates closer than MIN_SEPARATION_PX."""
    if len(points) <= 1:
        return points
    kept: List[ScatterPoint] = []
    coords = np.array([[p.x_px, p.y_px] for p in points])
    used = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        if used[i]:
            continue
        dists = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        nearby = np.where((dists < MIN_SEPARATION_PX) & (~used))[0]
        # Keep the one with highest circularity
        best = max(nearby, key=lambda j: points[j].circularity)
        kept.append(points[best])
        used[nearby] = True
    return kept


def _separate_series_by_color(roi: BGRImage, points: List[ScatterPoint]) -> np.ndarray:
    """Assign series labels by sampling HSV color at each point centroid."""
    if not points:
        return np.array([], dtype=int)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_img = roi.shape[0]
    w_img = roi.shape[1]
    features = []
    for pt in points:
        xi, yi = int(pt.x_px), int(pt.y_px)
        x1, x2 = max(0, xi - 3), min(w_img, xi + 4)
        y1, y2 = max(0, yi - 3), min(h_img, yi + 4)
        patch = hsv[y1:y2, x1:x2]
        if patch.size > 0:
            median_hs = np.median(patch.reshape(-1, 3), axis=0)[:2]
        else:
            median_hs = np.array([0.0, 0.0])
        features.append(median_hs)
    features = np.array(features, dtype=float)

    # Choose K by elbow (up to N_SERIES_MAX, max 4 for small sets)
    n = len(features)
    max_k = min(N_SERIES_MAX, n, 4)
    if max_k <= 1:
        return np.zeros(n, dtype=int)

    from sklearn.cluster import KMeans
    best_k, best_inertia = 1, float('inf')
    prev_inertia = float('inf')
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(features)
        inertia = km.inertia_
        if k > 1 and prev_inertia > 0:
            improvement = (prev_inertia - inertia) / prev_inertia
            if improvement < 0.05:
                break
        best_k = k
        best_inertia = inertia
        prev_inertia = inertia
        labels = km.labels_

    if best_k == 1:
        return np.zeros(n, dtype=int)
    return labels.astype(int)

def _drop_small_color_series(points: List[ScatterPoint],
                              labels: np.ndarray,
                              min_size: int = MIN_SERIES_SIZE
                              ) -> Tuple[List[ScatterPoint], np.ndarray]:
    """Drop members of color clusters smaller than `min_size`.

    Real data series in a scientific plot are uniform glyphs that produce one
    tight HSV cluster. Whisker caps, curve fragments, and panel-label letters
    each shift the centroid's HSV signature differently and get pushed into
    their own tiny clusters. Filtering by cluster size cleans them out in bulk.
    """
    if len(points) == 0:
        return points, labels
    from collections import Counter
    counts = Counter(labels.tolist())
    keep_labels = {lbl for lbl, n in counts.items() if n >= min_size}
    # Safety net: if every cluster is small (e.g. user has only 2 real points),
    # keep the largest one rather than wiping everything.
    if not keep_labels:
        biggest = max(counts.items(), key=lambda kv: kv[1])[0]
        keep_labels = {biggest}
    keep_idx = [i for i, lbl in enumerate(labels) if lbl in keep_labels]
    return [points[i] for i in keep_idx], labels[keep_idx]

def _merge_interleaved_series(points: List[ScatterPoint],
                               labels: np.ndarray,
                               smooth_tol_ratio: float = 0.25) -> np.ndarray:
    """
    Merge two color clusters when they sit on the same smooth path.
    For each pair of clusters (a, b), sort their union by x and check
    whether the polyline alternates between them while staying smooth
    in y. If so, relabel b -> a.
    """
    if len(points) < 4 or labels.max() < 1:
        return labels
    labels = labels.copy()
    unique = sorted(set(labels.tolist()))
    coords = np.array([[p.x_px, p.y_px] for p in points])

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
                # Smoothness: median |Δ²y| small vs. y range
                d2 = np.abs(np.diff(ys, n=2))
                yrange = max(1.0, ys.max() - ys.min())
                if np.median(d2) <= smooth_tol_ratio * yrange:
                    # interleavedness: at least one a-b-a or b-a-b pattern
                    seq = labels[order]
                    flips = int(np.sum(seq[:-1] != seq[1:]))
                    if flips >= 2:
                        labels[labels == b] = a
                        changed = True
                        break
            if changed:
                break
        unique = sorted(set(labels.tolist()))
    return labels


def _compute_confidence(points: List[ScatterPoint]) -> float:
    if not points:
        return 0.0
    mean_circ = np.mean([p.circularity for p in points])
    raw = len(points) * mean_circ
    return float(min(raw / CONFIDENCE_SATURATION, 1.0))
