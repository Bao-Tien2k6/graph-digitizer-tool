/**
 * Zustand store — owns all client-side state for PlotDigitizer.
 *
 * Once /calibrate has returned points + the affine matrix, every interactive
 * edit (drag a marker, edit a table cell, delete a row) is pure JS state.
 * The backend only sees /digitize and /calibrate.
 *
 * Affine matrix shape (2×3) from the backend:
 *   [[ax, 0, bx],   -> data_x = ax*pixel_x + bx (linear)
 *    [0, ay, by]]   -> data_y = ay*pixel_y + by (linear)
 * For log10 axes, the formula is data = 10**(a*pixel + b), so the inverse
 * needs the scale_type. We keep the scale flags alongside the matrix.
 */

import { create } from 'zustand';

function pixelToData(px, py, affine, scales) {
  const [[ax, , bx], [, ay, by]] = affine;
  const xLin = ax * px + bx;
  const yLin = ay * py + by;
  return {
    x: scales.x === 'log10' ? Math.pow(10, xLin) : xLin,
    y: scales.y === 'log10' ? Math.pow(10, yLin) : yLin,
  };
}

function dataToPixel(x, y, affine, scales) {
  const [[ax, , bx], [, ay, by]] = affine;
  const xLin = scales.x === 'log10' ? Math.log10(Math.max(x, 1e-12)) : x;
  const yLin = scales.y === 'log10' ? Math.log10(Math.max(y, 1e-12)) : y;
  return {
    pixel_x: ax !== 0 ? (xLin - bx) / ax : 0,
    pixel_y: ay !== 0 ? (yLin - by) / ay : 0,
  };
}

function getScales(axes) {
  return {
    x: axes?.x_axis?.scale_type || 'linear',
    y: axes?.y_axis?.scale_type || 'linear',
  };
}

const initialState = {
  sessionId: null,
  imageUrl: null,
  imageSize: { width: 0, height: 0 },
  axes: null,
  chartType: null,
  points: [],
  affine: null,
  selectedPointId: null,
  history: [],
  isBusy: false,
};

export const useDigitizerStore = create((set, get) => ({
  ...initialState,

  setBusy(isBusy) {
    set({ isBusy });
  },

  setDigitizeResult(resp) {
    set({
      sessionId: resp.session_id,
      imageUrl: resp.image_url,
      imageSize: { width: resp.image_width, height: resp.image_height },
      axes: resp.axes,
      chartType: null,
      points: [],
      affine: null,
      selectedPointId: null,
      history: [],
    });
  },

  updateTick(axis, pixelPos, newValue) {
    set((state) => {
      if (!state.axes) return {};
      const axisKey = axis === 'x' ? 'x_axis' : 'y_axis';
      const updatedAxis = {
        ...state.axes[axisKey],
        ticks: state.axes[axisKey].ticks.map((t) =>
          t.pixel_pos === pixelPos ? { ...t, label_value: newValue } : t,
        ),
      };
      return { axes: { ...state.axes, [axisKey]: updatedAxis } };
    });
  },

  setCalibrateResult(resp) {
    set({
      chartType: resp.chart_type,
      points: resp.points,
      affine: resp.affine,
      axes: resp.axes,
      selectedPointId: null,
      history: [],
    });
  },

  pushHistory() {
    const { points, history } = get();
    set({ history: [...history.slice(-19), points] });
  },

  undo() {
    const { history } = get();
    if (history.length === 0) return;
    const prev = history[history.length - 1];
    set({ points: prev, history: history.slice(0, -1) });
  },

  movePoint(id, newPixelX, newPixelY) {
    const { affine, axes, points } = get();
    if (!affine || !axes) return;
    get().pushHistory();
    const scales = getScales(axes);
    const { x, y } = pixelToData(newPixelX, newPixelY, affine, scales);
    set({
      points: points.map((p) =>
        p.id === id
          ? { ...p, pixel_x: newPixelX, pixel_y: newPixelY, x, y }
          : p,
      ),
    });
  },

  updatePointField(id, field, value) {
    const { affine, axes, points } = get();
    if (!points.some((p) => p.id === id)) return;
    get().pushHistory();
    set({
      points: points.map((p) => {
        if (p.id !== id) return p;
        const next = { ...p, [field]: value };
        if ((field === 'x' || field === 'y') && affine && axes) {
          const { pixel_x, pixel_y } = dataToPixel(
            field === 'x' ? value : p.x,
            field === 'y' ? value : p.y,
            affine,
            getScales(axes),
          );
          next.pixel_x = pixel_x;
          next.pixel_y = pixel_y;
        }
        return next;
      }),
    });
  },

  addPoint(pixelX, pixelY, seriesId = 0) {
    const { affine, axes, points } = get();
    if (!affine || !axes) return;
    get().pushHistory();
    const scales = getScales(axes);
    const { x, y } = pixelToData(pixelX, pixelY, affine, scales);
    const id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `p-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    set({
      points: [
        ...points,
        {
          id,
          series_id: seriesId,
          x, y,
          delta_x: 0, delta_y: 0,
          pixel_x: pixelX, pixel_y: pixelY,
        },
        ],
      selectedPointId: id,
    });
  },

  deletePoint(id) {
    get().pushHistory();
    set((state) => ({
      points: state.points.filter((p) => p.id !== id),
      selectedPointId: state.selectedPointId === id ? null : state.selectedPointId,
    }));
  },

  setSelectedPointId(id) {
    set({ selectedPointId: id });
  },

  reset() {
    set({ ...initialState });
  },
}));
