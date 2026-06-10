import { LineChart, RotateCcw } from 'lucide-react';

import CalibrationPanel from '@/components/CalibrationPanel';
import DataTable from '@/components/DataTable';
import ExportBar from '@/components/ExportBar';
import FileUploadZone from '@/components/FileUploadZone';
import ImageCanvas from '@/components/ImageCanvas';
import { Button } from '@/components/ui/button';
import * as api from '@/lib/api';
import { useDigitizerStore } from '@/store/useDigitizerStore';

export default function App() {
  const sessionId = useDigitizerStore((s) => s.sessionId);
  const points = useDigitizerStore((s) => s.points);
  const reset = useDigitizerStore((s) => s.reset);

  const hasSession = Boolean(sessionId);
  const hasPoints = points.length > 0;

  function onNewImage() {
    void api.deleteSession(sessionId);
    reset();
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <LineChart className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-lg font-semibold">PlotDigitizer</h1>
              <p className="text-xs text-muted-foreground">
                Extract numerical data from scientific figures.
              </p>
            </div>
          </div>
          {hasSession && (
            <Button variant="outline" size="sm" onClick={onNewImage}>
              <RotateCcw className="h-4 w-4" /> New image
            </Button>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        {!hasSession ? (
          <div className="mx-auto max-w-xl">
            <FileUploadZone />
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
            <section className="space-y-4">
              <div className="h-[60vh] min-h-[420px]">
                <ImageCanvas />
              </div>
            </section>
            <aside className="space-y-4">
              {!hasPoints ? (
                <CalibrationPanel />
              ) : (
                <>
                  <DataTable />
                  <ExportBar />
                  <details className="rounded-lg border bg-card">
                    <summary className="cursor-pointer select-none px-4 py-2 text-sm font-medium">
                      Re-edit axis ticks
                    </summary>
                    <div className="p-4">
                      <CalibrationPanel />
                    </div>
                  </details>
                </>
              )}
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
