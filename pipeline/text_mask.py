"""
pipeline/text_mask.py
=====================
Build a binary mask of text-like regions INSIDE the plot area (the rectangle
bounded by the detected x- and y-axes) so that detectors can exclude legend /
annotation pixels before blob detection.

Method:
  1. Run EasyOCR on the plot-region ROI only.
  2. Every multi-character OCR token -> mask its bbox.
  3. Single-char tokens that look like 'o' / 'O' / '0' / 'Q' / 'D' get an
     extra disambiguation step so we don't accidentally mask open-circle
     data markers as text.
"""
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np

from pipeline.preprocess import BGRImage

# Tunables
OCR_MIN_CONF       = 0.40   
TEXT_DILATE_PX     = 1 
SAT_TEXT_MAX       = 120

# o-vs-marker disambiguation
AMBIGUOUS_GLYPHS   = set("oO0QDc")          # single-char tokens needing a check
NEIGHBOR_DX_FACTOR = 2.0       # horizontal search radius (in glyph widths)
NEIGHBOR_DY_FACTOR = 1.0       # vertical search radius (in glyph heights)
PUNCT_CHARS        = set("().,[]|:;-_=+/\\'\"")
CIRCULARITY_MARKER = 0.82      # >= this AND hollow => open-circle marker
STROKE_CV_MARKER   = 0.35      # ring stroke thickness CV below this => marker
FILLED_INTERIOR_FR = 0.55      # >= this fraction of interior is ink => filled marker


def build_text_mask(img: BGRImage,
                    plot_region: Tuple[int, int, int, int],
                    global_ocr_results: list = None) -> np.ndarray:
    """Return a uint8 mask (H, W) where 255 = text pixel to exclude."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x0, y0, x1, y1 = plot_region
    roi = img[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape[:2]) < 5:
            return mask

    if not global_ocr_results:
        return mask

    grey_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_roi  = hsv_roi[:, :, 1]

    # Normalize global OCR boxes -> ROI relative pixel rects
    boxes: List[Tuple[str, float, int, int, int, int]] = [] 
    
    for line in global_ocr_results:
        if not line or len(line) != 2:
            continue
            
        box_points = np.array(line[0], dtype=float)
        text = line[1][0]
        conf = line[1][1]

        if float(conf) < OCR_MIN_CONF:
            continue
        if not text or not text.strip():
            continue

        min_x = int(round(box_points[:, 0].min()))
        max_x = int(round(box_points[:, 0].max()))
        min_y = int(round(box_points[:, 1].min()))
        max_y = int(round(box_points[:, 1].max()))

        # Check if text bbox intersects the plot region
        if max_x > x0 and min_x < x1 and max_y > y0 and min_y < y1:
            # Map absolute to relative coordinates
            rx = min_x - x0
            ry = min_y - y0
            rw = max_x - min_x
            rh = max_y - min_y
            
            # Clamp to ROI bounds
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, (x1 - x0) - rx)
            rh = min(rh, (y1 - y0) - ry)

            if rw <= 0 or rh <= 0:
                continue
                
            boxes.append((text.strip(), float(conf), rx, ry, rw, rh))

    if not boxes:
        return mask

    # Decide which boxes to mask
    local = np.zeros(roi.shape[:2], dtype=np.uint8)
    for i, (text, conf, rx, ry, rw, rh) in enumerate(boxes):
        if _is_text_box(text, rx, ry, rw, rh, i, boxes,
                        grey_roi, sat_roi):
            local[ry:ry + rh, rx:rx + rw] = 255

    if TEXT_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (TEXT_DILATE_PX * 2 + 1, TEXT_DILATE_PX * 2 + 1),
        )
        local = cv2.dilate(local, k, iterations=1)

    mask[y0:y1, x0:x1] = local
    return mask


# Per-box decision

def _is_text_box(text: str,
                 rx: int, ry: int, rw: int, rh: int,
                 idx: int,
                 boxes: List[Tuple[str, float, int, int, int, int]],
                 grey_roi: np.ndarray,
                 sat_roi: np.ndarray) -> bool:
    """True if this OCR box should be treated as text and masked."""
    # Highly saturated ink => colored data marker, not text
    patch_sat = sat_roi[ry:ry + rh, rx:rx + rw]
    if patch_sat.size and float(np.median(patch_sat)) > SAT_TEXT_MAX:
        return False

    # Multi-character tokens: trust OCR, no marker reads as >=2 chars confidently
    if len(text) >= 2:
        return True

    # Single character — only the ambiguous ones need extra checks
    if text not in AMBIGUOUS_GLYPHS:
        return True

    # 1) Neighbor on the same text line?  => part of a word, mask it.
    if _has_text_neighbor(idx, boxes):
        return True

    # 2) Punctuation/brackets adjacent in the pixel strip? => mask it.
    if _has_adjacent_punctuation(rx, ry, rw, rh, grey_roi):
        return True

    # 3) Shape signature — looks like a marker (filled disc OR clean open ring)?
    if _looks_like_marker(rx, ry, rw, rh, grey_roi):
        return False

    # Default: when in doubt for a lone 'o' with no context, leave it alone.
    # Better to under-mask (a stray glyph leaks into detection) than over-mask
    # (a real data point gets erased).
    return False


def _has_text_neighbor(idx: int,
                       boxes: List[Tuple[str, float, int, int, int, int]]
                       ) -> bool:
    """True if another OCR box sits within one text-line of this one."""
    _, _, rx, ry, rw, rh = boxes[idx]
    cx, cy = rx + rw / 2.0, ry + rh / 2.0
    dx_lim = NEIGHBOR_DX_FACTOR * max(rw, rh)
    dy_lim = NEIGHBOR_DY_FACTOR * rh
    for j, (_, _, ox, oy, ow, oh) in enumerate(boxes):
        if j == idx:
            continue
        ocx, ocy = ox + ow / 2.0, oy + oh / 2.0
        if abs(ocy - cy) <= dy_lim and abs(ocx - cx) <= dx_lim:
            return True
    return False


def _has_adjacent_punctuation(rx: int, ry: int, rw: int, rh: int,
                              grey_roi: np.ndarray) -> bool:
    """
    Look for a thin dark vertical bar (parenthesis / bracket / bar) or a small
    dot just outside the glyph bbox on the left or right. Catches '(o)', 'o.',
    'a, o, b' where EasyOCR split the punctuation into its own glyph cluster
    that fell below confidence.
    """
    H, W = grey_roi.shape
    pad_x = max(2, rw // 2)
    # left strip
    lx0 = max(0, rx - pad_x); lx1 = max(0, rx - 1)
    ly0 = max(0, ry - 1);     ly1 = min(H, ry + rh + 1)
    # right strip
    rx0 = min(W, rx + rw + 1); rx1 = min(W, rx + rw + pad_x)
    ry0 = ly0;                 ry1 = ly1

    for sx0, sx1 in ((lx0, lx1), (rx0, rx1)):
        if sx1 <= sx0:
            continue
        strip = grey_roi[ly0:ly1, sx0:sx1]
        if strip.size == 0:
            continue
        ink = strip < 120
        if ink.sum() < 3:
            continue
        # Vertical bar test: at least 60% of rows have an ink pixel
        rows_with_ink = (ink.sum(axis=1) > 0).sum()
        if rows_with_ink >= 0.6 * strip.shape[0]:
            return True
        # Dot test: a tight cluster of ink near the baseline
        ys, _ = np.where(ink)
        if len(ys) >= 2 and ys.max() >= 0.7 * strip.shape[0] \
                and (ys.max() - ys.min()) <= max(2, rh // 4):
            return True
    return False


def _looks_like_marker(rx: int, ry: int, rw: int, rh: int,
                       grey_roi: np.ndarray) -> bool:
    """
    Distinguish a marker symbol from a letter 'o' inside the bbox.

    Returns True for shapes that look like data markers:
      - Filled disc: interior is mostly ink.
      - Open circle: ring is highly circular AND stroke thickness is uniform.
    Returns False for shapes that look like letter glyphs:
      - Hollow but non-circular (taller than wide, or uneven stroke).
    """
    H, W = grey_roi.shape
    rx = max(0, rx); ry = max(0, ry)
    rw = min(W - rx, rw); rh = min(H - ry, rh)
    if rw < 3 or rh < 3:
        return False

    patch = grey_roi[ry:ry + rh, rx:rx + rw]
    ink = patch < 140
    if ink.sum() < 4:
        return False

    # Filled-disc test
    # Erode the bbox interior by ~25% and check the ink fraction there.
    iy0 = int(0.25 * rh); iy1 = int(0.75 * rh)
    ix0 = int(0.25 * rw); ix1 = int(0.75 * rw)
    interior = ink[iy0:iy1, ix0:ix1]
    if interior.size:
        interior_frac = float(interior.sum()) / interior.size
        if interior_frac >= FILLED_INTERIOR_FR:
            return True   # solid blob => filled marker

    # Open-ring test (largest external contour)
    ink_u8 = ink.astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, closed=True))
    if area < 4 or perim < 4:
        return False
    circularity = 4.0 * np.pi * area / (perim * perim)

    # Stroke-thickness uniformity via distance transform on the ring
    dist = cv2.distanceTransform(ink_u8, cv2.DIST_L2, 3)
    stroke = dist[ink]
    if stroke.size < 4:
        return False
    stroke_mean = float(stroke.mean())
    stroke_cv   = float(stroke.std()) / stroke_mean if stroke_mean > 0 else 1.0

    if circularity >= CIRCULARITY_MARKER and stroke_cv <= STROKE_CV_MARKER:
        return True   # clean uniform ring => open-circle marker

    return False