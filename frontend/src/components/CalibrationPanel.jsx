import { useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import * as api from '@/lib/api';
import { useDigitizerStore } from '@/store/useDigitizerStore';

function ScaleChip({ axisLabel, axis }) {
  if (!axis) return null;
  const ok = axis.scale_r2 >= 0.95;
  return (
    <div className="flex items-center justify-between rounded-md border bg-background px-3 py-2 text-xs">
      <span className="font-medium">{axisLabel}</span>
      <span className="text-muted-foreground">{axis.scale_type}</span>
      <span className={ok ? 'text-emerald-600' : 'text-amber-600'}>
        R²={axis.scale_r2.toFixed(4)}
      </span>
    </div>
  );
}

function TickRow({ axis, tick }) {
  const updateTick = useDigitizerStore((s) => s.updateTick);
  const conf = tick.ocr_confidence;
  const dot = conf >= 0.7 ? '🟢' : '🔴';
  return (
    <div className="grid grid-cols-[1fr_auto] gap-3 items-center py-1">
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">
          {dot} px {tick.pixel_pos} · OCR "{tick.raw_text}" · conf {conf.toFixed(2)}
        </Label>
        <Input
          type="number"
          step="any"
          value={Number.isFinite(tick.label_value) ? tick.label_value : 0}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            updateTick(axis, tick.pixel_pos, Number.isFinite(v) ? v : 0);
          }}
          className="h-8 text-sm"
        />
      </div>
      <span className="text-xs text-muted-foreground">px {tick.pixel_pos}</span>
    </div>
  );
}

export default function CalibrationPanel() {
  const sessionId = useDigitizerStore((s) => s.sessionId);
  const axes = useDigitizerStore((s) => s.axes);
  const setCalibrateResult = useDigitizerStore((s) => s.setCalibrateResult);
  const setBusy = useDigitizerStore((s) => s.setBusy);
  const [busy, setLocalBusy] = useState(false);

  if (!axes) return null;

  async function onRecalibrate() {
    setLocalBusy(true);
    setBusy(true);
    try {
      const resp = await api.calibrate(
        sessionId,
        axes.x_axis.ticks.map((t) => ({
          pixel_pos: t.pixel_pos,
          label_value: t.label_value,
        })),
        axes.y_axis.ticks.map((t) => ({
          pixel_pos: t.pixel_pos,
          label_value: t.label_value,
        })),
      );
      setCalibrateResult(resp);
      toast.success(`Extracted ${resp.points.length} point(s) as ${resp.chart_type}.`);
    } catch (e) {
      toast.error(`Calibration failed: ${e.message}`);
    } finally {
      setLocalBusy(false);
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Step 2 — Verify axis calibration</CardTitle>
        <CardDescription>
          Correct any tick values that OCR misread, then re-detect points.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <ScaleChip axisLabel="X-axis" axis={axes.x_axis} />
          <ScaleChip axisLabel="Y-axis" axis={axes.y_axis} />
        </div>

        <details className="rounded-md border bg-background" open>
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
            X-axis ticks ({axes.x_axis.ticks.length})
          </summary>
          <div className="px-3 pb-3">
            {axes.x_axis.ticks.length === 0 ? (
              <p className="text-xs text-muted-foreground">No ticks detected.</p>
            ) : (
              axes.x_axis.ticks.map((t) => (
                <TickRow key={`x-${t.pixel_pos}`} axis="x" tick={t} />
              ))
            )}
          </div>
        </details>

        <details className="rounded-md border bg-background" open>
          <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium">
            Y-axis ticks ({axes.y_axis.ticks.length})
          </summary>
          <div className="px-3 pb-3">
            {axes.y_axis.ticks.length === 0 ? (
              <p className="text-xs text-muted-foreground">No ticks detected.</p>
            ) : (
              axes.y_axis.ticks.map((t) => (
                <TickRow key={`y-${t.pixel_pos}`} axis="y" tick={t} />
              ))
            )}
          </div>
        </details>

        <Button
          onClick={onRecalibrate}
          disabled={busy}
          className="w-full"
        >
          {busy ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Detecting points…
            </>
          ) : (
            <>
              <RefreshCw className="h-4 w-4" /> Detect / re-detect points
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
