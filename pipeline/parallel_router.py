"""
pipeline/parallel_router.py
===========================
Stage 3 — Parallel Detection & Routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pipeline.axes_detector import AxesInfo
from pipeline.preprocess import BGRImage
from pipeline.text_mask import build_text_mask

log = logging.getLogger(__name__)

# Public data structures
@dataclass
class DetectionResult:
    """Output of a single detector."""
    chart_type: str
    confidence: float
    pixel_points: List[Dict] = field(default_factory=list)
    detector_meta: Dict = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Final output of Stage 3."""
    primary_chart_type: str
    winning_detections: DetectionResult
    is_mixed: bool = False
    secondary_detections: Optional[DetectionResult] = None
    all_results: Dict[str, DetectionResult] = field(default_factory=dict)


# Main entry point
def route(img: BGRImage, axes: AxesInfo, global_ocr_results: Optional[list] = None) -> RoutingResult:
    """Run all three detectors and return the routing decision."""
    work_img = axes.inpainted_image if axes.inpainted_image is not None else img
    # Build the text mask for all detectors ---
    if global_ocr_results is None:
        global_ocr_results = []
        
    text_mask = build_text_mask(
        img=work_img, 
        plot_region=axes.plot_region, 
        global_ocr_results=global_ocr_results
    )
    # Pass the mask down to the runners
    results = _run_detectors(work_img, axes, text_mask)
    return _select_winner(results)


# Step helpers
def _run_detectors(
    img: BGRImage,
    axes: AxesInfo,
    text_mask: np.ndarray,
) -> Dict[str, DetectionResult]:
    """Run scatter, bar, and line detectors and collect their results."""
    tasks = {
        "scatter": _run_scatter_detector_task,
        "bar":     _run_bar_detector_task,
        "line":    _run_line_detector_task,
    }
    results: Dict[str, DetectionResult] = {}

    for name, fn in tasks.items():
        try:
            # Pass the mask to each task
            results[name] = fn(img, axes, text_mask)
        except Exception as exc:
            log.warning("Detector '%s' failed: %s", name, exc)
            results[name] = DetectionResult(
                chart_type=name, confidence=0.0,
                detector_meta={"error": str(exc)}
            )

    return results


def _run_scatter_detector_task(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray) -> DetectionResult:
    from pipeline.scatter_detector import detect
    return detect(img, axes, text_mask)  


def _run_bar_detector_task(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray) -> DetectionResult:
    from pipeline.bar_detector import detect
    return detect(img, axes, text_mask) 


def _run_line_detector_task(img: BGRImage, axes: AxesInfo, text_mask: np.ndarray) -> DetectionResult:
    from pipeline.line_detector import detect
    return detect(img, axes, text_mask) 


def _select_winner(results: Dict[str, DetectionResult]) -> RoutingResult:
    """
    Choose the primary chart type by confidence vote.
    Tiebreaker: if skeleton_continuity > 0.8 and N_scatter ≤ 15, prefer line.
    """
    if not results:
        raise AllDetectorsFailed("No detector results available.")

    if all(r.confidence == 0.0 for r in results.values()):
        raise AllDetectorsFailed("All detectors returned confidence=0.")

    # Sort by confidence descending; tiebreak order: scatter > line > bar
    order = {"scatter": 2, "line": 1, "bar": 0}
    sorted_results = sorted(
        results.values(),
        key=lambda r: (r.confidence, order.get(r.chart_type, 0)),
        reverse=True,
    )

    winner = sorted_results[0]
    second = sorted_results[1] if len(sorted_results) > 1 else None

    # If line_detector is the winner but its confidence is low, fall back to scatter.
    # if (winner.chart_type == "line"
    #         and winner.confidence <= 0.5
    #         and "scatter" in results
    #         and len(results["scatter"].pixel_points) > 0):
    #     log.info("Line confidence %.3f <= 0.5; falling back to scatter "
    #              "(n_scatter=%d)", winner.confidence,
    #              len(results["scatter"].pixel_points))
    #     winner = results["scatter"]
    #     second = results["line"]
    if (winner.chart_type == "line"
            and "scatter" in results
            and results["scatter"].confidence > winner.confidence
            and len(results["scatter"].pixel_points) > 0):
        log.info("Scatter conf %.3f > line conf %.3f; falling back to scatter "
                 "(n_scatter=%d)",
                 results["scatter"].confidence, winner.confidence,
                 len(results["scatter"].pixel_points))
        winner = results["scatter"]
        second = results["line"]
    
    # prefer bar when bar detection is strong, even if line scored higher.
    bar_r = results.get("bar")
    if (bar_r is not None
            and bar_r.confidence >= 0.7
            and bar_r.detector_meta.get("n_bars", 0) >= 3
            and winner.chart_type == "line"):
        winner = bar_r
        second = sorted_results[0]
        
    is_mixed = second is not None and second.confidence > 0.5

    return RoutingResult(
        primary_chart_type=winner.chart_type,
        winning_detections=winner,
        is_mixed=is_mixed,
        secondary_detections=second if is_mixed else None,
        all_results=results,
    )

# Custom exceptions
class AllDetectorsFailed(RuntimeError):
    """Raised when every detector returns confidence == 0.0."""
