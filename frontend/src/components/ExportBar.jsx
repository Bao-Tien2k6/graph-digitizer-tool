import { useCallback } from 'react';
import { FileJson, FileSpreadsheet, FileText } from 'lucide-react';
import * as XLSX from 'xlsx';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { useDigitizerStore } from '@/store/useDigitizerStore';

function pointsToRows(points) {
  return [...points]
    .sort((a, b) => a.series_id - b.series_id || a.x - b.x)
    .map((p) => ({
      series: p.series_id,
      x: Number(p.x.toFixed(4)),
      y: Number(p.y.toFixed(4)),
      delta_x: Number(p.delta_x.toFixed(4)),
      delta_y: Number(p.delta_y.toFixed(4)),
    }));
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function ExportBar() {
  const points = useDigitizerStore((s) => s.points);
  const chartType = useDigitizerStore((s) => s.chartType);

  const disabled = points.length === 0;
  const stem = chartType ? `plotdigitizer_${chartType}` : 'plotdigitizer';

  const onCsv = useCallback(() => {
    const rows = pointsToRows(points);
    const header = 'series,x,y,delta_x,delta_y';
    const body = rows
      .map((r) => `${r.series},${r.x},${r.y},${r.delta_x},${r.delta_y}`)
      .join('\n');
    download(new Blob([`${header}\n${body}\n`], { type: 'text/csv' }), `${stem}.csv`);
    toast.success('CSV downloaded');
  }, [points, stem]);

  const onJson = useCallback(() => {
    const rows = pointsToRows(points);
    download(
      new Blob([JSON.stringify({ chart_type: chartType, points: rows }, null, 2)], {
        type: 'application/json',
      }),
      `${stem}.json`,
    );
    toast.success('JSON downloaded');
  }, [points, chartType, stem]);

  const onXlsx = useCallback(() => {
    const rows = pointsToRows(points);
    const ws = XLSX.utils.json_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'points');
    XLSX.writeFile(wb, `${stem}.xlsx`);
    toast.success('XLSX downloaded');
  }, [points, stem]);

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-card px-3 py-2">
      <p className="text-sm font-medium mr-auto">Export</p>
      <Button variant="outline" size="sm" onClick={onCsv} disabled={disabled}>
        <FileText className="h-4 w-4" /> CSV
      </Button>
      <Button variant="outline" size="sm" onClick={onJson} disabled={disabled}>
        <FileJson className="h-4 w-4" /> JSON
      </Button>
      <Button variant="outline" size="sm" onClick={onXlsx} disabled={disabled}>
        <FileSpreadsheet className="h-4 w-4" /> XLSX
      </Button>
    </div>
  );
}
