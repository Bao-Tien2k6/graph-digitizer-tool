"""FastAPI app for PlotDigitizer.

Exposes the existing pipeline (preprocess → OCR → axes → router → transform)
as JSON endpoints. The frontend (Vite + React) owns all interactive editing —
once /calibrate returns the affine matrix and points, dragging, deleting, and
exporting happen entirely client-side.

IMPORTANT — single-process only: session state (decoded images + detection
results) lives in the in-process ``SESSIONS`` dict. Run with ONE worker
(``uvicorn api.main:app`` without ``--workers``). Under multiple workers a
``/calibrate`` request can land on a worker that never handled the matching
``/digitize`` and will 404. Use an external store (Redis / disk) before
scaling out.
"""

from __future__ import annotations

import asyncio
import io
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from paddleocr import PaddleOCR

from api.schemas import (
    AxesPayload,
    AxisPayload,
    CalibrateRequest,
    CalibrateResponse,
    DigitizeResponse,
    HealthResponse,
    PointPayload,
    TickPayload,
)
from pipeline.axes_detector import AxesInfo, AxisInfo, TickInfo, _fit_scale, detect_axes
from pipeline.coordinate_transform import (
    _build_affine_matrix,
    _pixel_to_data,
    _propagate_uncertainty,
)
from pipeline.parallel_router import RoutingResult, route
from pipeline.preprocess import load_image_from_bytes, preprocess_image

SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 64               # hard cap; oldest are evicted past this
SWEEP_INTERVAL_SECONDS = 5 * 60  # background reclaim cadence
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # reject uploads larger than 25 MB
ALLOWED_UPLOAD_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@dataclass
class SessionState:
    image: np.ndarray
    global_ocr: list
    axes_info: AxesInfo
    routing_result: RoutingResult | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


SESSIONS: dict[str, SessionState] = {}
_OCR_ENGINE: PaddleOCR | None = None


def _sweep() -> None:
    """Drop sessions older than the TTL. Called opportunistically."""
    now = time.time()
    stale = [
        sid for sid, s in SESSIONS.items()
        if now - s.last_accessed > SESSION_TTL_SECONDS
    ]
    for sid in stale:
        SESSIONS.pop(sid, None)


def _evict_if_full() -> None:
    """Bound memory: drop the least-recently-accessed sessions past the cap."""
    while len(SESSIONS) >= MAX_SESSIONS:
        oldest = min(SESSIONS, key=lambda sid: SESSIONS[sid].last_accessed)
        SESSIONS.pop(oldest, None)


def _get_session(session_id: str) -> SessionState:
    _sweep()
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    session.last_accessed = time.time()
    return session


async def _periodic_sweep() -> None:
    """Reclaim expired sessions even while the server is idle (no traffic)."""
    while True:
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
        _sweep()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _OCR_ENGINE
    _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="en")
    sweeper = asyncio.create_task(_periodic_sweep())
    try:
        yield
    finally:
        sweeper.cancel()
        SESSIONS.clear()


app = FastAPI(title="PlotDigitizer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Conversion helpers -------------------------------------------------------

def _tick_to_payload(t: TickInfo) -> TickPayload:
    return TickPayload(
        pixel_pos=int(t.pixel_pos),
        label_value=float(t.label_value),
        ocr_confidence=float(t.ocr_confidence),
        raw_text=str(t.raw_text or ""),
    )


def _axis_to_payload(a: AxisInfo) -> AxisPayload:
    return AxisPayload(
        line_pixel=int(a.line_pixel),
        scale_type=a.scale_type.value,
        scale_r2=float(a.scale_r2),
        ticks=[_tick_to_payload(t) for t in a.ticks],
    )


def _axes_to_payload(a: AxesInfo) -> AxesPayload:
    return AxesPayload(
        x_axis=_axis_to_payload(a.x_axis),
        y_axis=_axis_to_payload(a.y_axis),
        plot_region=tuple(int(v) for v in a.plot_region),
    )


def _apply_tick_edits(axis: AxisInfo, edits: list[Any]) -> AxisInfo:
    """Overwrite tick label_values from user edits; refit scale."""
    edit_map = {int(e.pixel_pos): float(e.label_value) for e in edits}
    new_ticks: list[TickInfo] = []
    for t in axis.ticks:
        new_value = edit_map.get(int(t.pixel_pos), t.label_value)
        new_ticks.append(TickInfo(
            pixel_pos=int(t.pixel_pos),
            label_value=float(new_value),
            ocr_confidence=float(t.ocr_confidence),
            raw_text=str(t.raw_text or ""),
        ))
    return _fit_scale(new_ticks, axis.line_pixel)


# Routes -------------------------------------------------------------------

@app.get("/api/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse()


@app.post("/api/digitize", response_model=DigitizeResponse)
async def digitize(image: UploadFile = File(...)) -> DigitizeResponse:
    _sweep()

    # Don't trust the client: validate content type and bound the read so a
    # huge or malicious upload cannot OOM the worker.
    if image.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type {image.content_type!r}; expected PNG or JPEG",
        )
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    try:
        img = load_image_from_bytes(raw)
        img = preprocess_image(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Preprocess failed: {exc}") from exc

    assert _OCR_ENGINE is not None, "OCR engine not initialized"
    try:
        # PaddleOCR 2.x .ocr() returns a list-per-image; index [0] is this
        # single image's lines. Each line is [bbox, (text, conf)] — the shape
        # the axes detector and text-mask builder downstream rely on.
        raw_ocr = _OCR_ENGINE.ocr(img, cls=True)
        global_ocr = raw_ocr[0] if raw_ocr and raw_ocr[0] else []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    try:
        axes_info = detect_axes(img, global_ocr_results=global_ocr)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Axes detection failed: {exc}") from exc

    _evict_if_full()
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = SessionState(
        image=img,
        global_ocr=global_ocr,
        axes_info=axes_info,
    )

    height, width = img.shape[:2]
    return DigitizeResponse(
        session_id=session_id,
        image_width=int(width),
        image_height=int(height),
        image_url=f"/api/image/{session_id}",
        axes=_axes_to_payload(axes_info),
    )


@app.get("/api/image/{session_id}")
def get_image(session_id: str) -> StreamingResponse:
    session = _get_session(session_id)
    ok, buf = cv2.imencode(".png", session.image)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest) -> CalibrateResponse:
    session = _get_session(req.session_id)

    new_x = _apply_tick_edits(session.axes_info.x_axis, req.x_ticks)
    new_y = _apply_tick_edits(session.axes_info.y_axis, req.y_ticks)
    session.axes_info = AxesInfo(
        x_axis=new_x,
        y_axis=new_y,
        plot_region=session.axes_info.plot_region,
        gridline_mask=session.axes_info.gridline_mask,
        inpainted_image=session.axes_info.inpainted_image,
    )

    try:
        routing = route(
            session.image,
            session.axes_info,
            global_ocr_results=session.global_ocr,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Routing failed: {exc}") from exc
    session.routing_result = routing

    affine = _build_affine_matrix(session.axes_info)
    points = _pixel_to_data(
        routing.winning_detections.pixel_points, affine, session.axes_info
    )
    points = _propagate_uncertainty(points, session.axes_info, routing.winning_detections)

    payloads = [
        PointPayload(
            id=uuid.uuid4().hex,
            series_id=int(p.series_id),
            x=float(p.x),
            y=float(p.y),
            delta_x=float(p.delta_x),
            delta_y=float(p.delta_y),
            pixel_x=float(p.pixel_x),
            pixel_y=float(p.pixel_y),
        )
        for p in points
    ]

    return CalibrateResponse(
        chart_type=routing.primary_chart_type,
        points=payloads,
        affine=[[float(v) for v in row] for row in affine.tolist()],
        axes=_axes_to_payload(session.axes_info),
    )


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> dict:
    existed = SESSIONS.pop(session_id, None) is not None
    return {"deleted": existed}
