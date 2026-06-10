import { useCallback, useRef, useState } from 'react';
import { Upload, Loader2 } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import * as api from '@/lib/api';
import { useDigitizerStore } from '@/store/useDigitizerStore';
import { cn } from '@/lib/utils';

const ACCEPTED = ['image/png', 'image/jpeg'];

export default function FileUploadZone() {
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);
  const setDigitizeResult = useDigitizerStore((s) => s.setDigitizeResult);
  const setBusyGlobal = useDigitizerStore((s) => s.setBusy);

  const handleFile = useCallback(
    async (file) => {
      if (!file) return;
      if (!ACCEPTED.includes(file.type)) {
        toast.error('Please upload a PNG or JPG image.');
        return;
      }
      setBusy(true);
      setBusyGlobal(true);
      try {
        const resp = await api.digitize(file);
        setDigitizeResult(resp);
        toast.success('Axes detected — verify the calibration on the right.');
      } catch (e) {
        toast.error(`Upload failed: ${e.message}`);
      } finally {
        setBusy(false);
        setBusyGlobal(false);
      }
    },
    [setDigitizeResult, setBusyGlobal],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer?.files?.[0];
        if (f) void handleFile(f);
      }}
      className={cn(
        'rounded-lg border-2 border-dashed p-8 text-center transition-colors',
        dragOver ? 'border-primary bg-accent' : 'border-border bg-muted/30',
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleFile(f);
          e.target.value = '';
        }}
      />
      <div className="flex flex-col items-center gap-3">
        {busy ? (
          <Loader2 className="h-10 w-10 animate-spin text-muted-foreground" />
        ) : (
          <Upload className="h-10 w-10 text-muted-foreground" />
        )}
        <p className="text-sm font-medium">
          {busy ? 'Detecting axes…' : 'Drag and drop a chart image'}
        </p>
        <p className="text-xs text-muted-foreground">PNG or JPG · single panel</p>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          Browse files
        </Button>
      </div>
    </div>
  );
}
