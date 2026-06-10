"""Pydantic request/response models for the PlotDigitizer API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TickPayload(BaseModel):
    pixel_pos: int
    label_value: float
    ocr_confidence: float
    raw_text: str = ""


class AxisPayload(BaseModel):
    line_pixel: int
    scale_type: Literal["linear", "log10"]
    scale_r2: float
    ticks: list[TickPayload]


class AxesPayload(BaseModel):
    x_axis: AxisPayload
    y_axis: AxisPayload
    plot_region: tuple[int, int, int, int]


class DigitizeResponse(BaseModel):
    session_id: str
    image_width: int
    image_height: int
    image_url: str
    axes: AxesPayload


class TickEdit(BaseModel):
    pixel_pos: int
    label_value: float


class CalibrateRequest(BaseModel):
    session_id: str
    x_ticks: list[TickEdit] = Field(default_factory=list)
    y_ticks: list[TickEdit] = Field(default_factory=list)


class PointPayload(BaseModel):
    id: str
    series_id: int
    x: float
    y: float
    delta_x: float = 0.0
    delta_y: float = 0.0
    pixel_x: float = 0.0
    pixel_y: float = 0.0


class CalibrateResponse(BaseModel):
    chart_type: str
    points: list[PointPayload]
    affine: list[list[float]]
    axes: AxesPayload


class HealthResponse(BaseModel):
    status: str = "ok"
