import { useEffect, useMemo, useRef, useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Circle, Line, Group, Rect } from 'react-konva';
import useImage from 'use-image';

import { useDigitizerStore } from '@/store/useDigitizerStore';

const SERIES_PALETTE = [
  '#ef4444', '#3b82f6', '#10b981', '#f59e0b',
  '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16',
];

function colorFor(seriesId) {
  return SERIES_PALETTE[seriesId % SERIES_PALETTE.length];
}

export default function ImageCanvas() {
  const containerRef = useRef(null);
  const [container, setContainer] = useState({ width: 600, height: 400 });

  const imageUrl = useDigitizerStore((s) => s.imageUrl);
  const imageSize = useDigitizerStore((s) => s.imageSize);
  const axes = useDigitizerStore((s) => s.axes);
  const points = useDigitizerStore((s) => s.points);
  const selectedPointId = useDigitizerStore((s) => s.selectedPointId);
  const setSelectedPointId = useDigitizerStore((s) => s.setSelectedPointId);
  const movePoint = useDigitizerStore((s) => s.movePoint);
  const deletePoint = useDigitizerStore((s) => s.deletePoint);

  const [image] = useImage(imageUrl || '', 'anonymous');

  // Track container size for responsive stage.
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        setContainer({ width: Math.max(200, width), height: Math.max(200, height) });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Fit the image into the container while preserving aspect ratio.
  const fitScale = useMemo(() => {
    if (!imageSize.width || !imageSize.height) return 1;
    return Math.min(
      container.width / imageSize.width,
      container.height / imageSize.height,
      1,
    );
  }, [container, imageSize]);

  const [scale, setScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    setScale(fitScale);
    setStagePos({ x: 0, y: 0 });
  }, [fitScale]);

  const stageWidth = container.width;
  const stageHeight = container.height;

  function handleWheel(e) {
    e.evt.preventDefault();
    const stage = e.target.getStage();
    const oldScale = scale;
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stagePos.x) / oldScale,
      y: (pointer.y - stagePos.y) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const factor = 1.1;
    const newScale = Math.max(0.1, Math.min(8, direction > 0 ? oldScale * factor : oldScale / factor));
    setScale(newScale);
    setStagePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  }

  const axisLines = useMemo(() => {
    if (!axes) return null;
    const { x_axis, y_axis, plot_region } = axes;
    const [x0, y0, x1, y1] = plot_region;
    return (
      <Group listening={false}>
        <Line
          points={[x0, x_axis.line_pixel, x1, x_axis.line_pixel]}
          stroke="rgba(16, 185, 129, 0.8)"
          strokeWidth={1.5 / scale}
        />
        <Line
          points={[y_axis.line_pixel, y0, y_axis.line_pixel, y1]}
          stroke="rgba(16, 185, 129, 0.8)"
          strokeWidth={1.5 / scale}
        />
        {x_axis.ticks.map((t, i) => (
          <Line
            key={`xt-${i}`}
            points={[t.pixel_pos, x_axis.line_pixel - 6 / scale, t.pixel_pos, x_axis.line_pixel + 6 / scale]}
            stroke="rgba(245, 158, 11, 0.9)"
            strokeWidth={1.5 / scale}
          />
        ))}
        {y_axis.ticks.map((t, i) => (
          <Line
            key={`yt-${i}`}
            points={[y_axis.line_pixel - 6 / scale, t.pixel_pos, y_axis.line_pixel + 6 / scale, t.pixel_pos]}
            stroke="rgba(245, 158, 11, 0.9)"
            strokeWidth={1.5 / scale}
          />
        ))}
      </Group>
    );
  }, [axes, scale]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[420px] rounded-lg border bg-muted/30 overflow-hidden"
    >
      {!imageUrl ? (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          Upload an image to begin.
        </div>
      ) : (
        <Stage
          width={stageWidth}
          height={stageHeight}
          scaleX={scale}
          scaleY={scale}
          x={stagePos.x}
          y={stagePos.y}
          draggable
          onDragEnd={(e) => {
            if (e.target == e.target.getStage()) {
              setStagePos({x: e.target.x(), y: e.target.y()})}
            }
          }
          onWheel={handleWheel}
          onMouseDown={(e) => {
            // Click on empty stage clears selection.
            if (e.target === e.target.getStage() || e.target.attrs?.name === 'bg') {
              setSelectedPointId(null);
            }
          }}
        >
          <Layer>
            <Rect
              name="bg"
              x={0}
              y={0}
              width={imageSize.width}
              height={imageSize.height}
              fill="white"
              listening={false}
            />
            {image && (
              <KonvaImage
                image={image}
                x={0}
                y={0}
                width={imageSize.width}
                height={imageSize.height}
                listening={false}
              />
            )}
            {axisLines}
            {points.map((p) => {
              const isSelected = selectedPointId === p.id;
              const radius = (isSelected ? 9 : 6) / scale;
              return (
                <Circle
                  key={p.id}
                  x={p.pixel_x}
                  y={p.pixel_y}
                  radius={radius}
                  fill={colorFor(p.series_id)}
                  stroke={isSelected ? '#1e293b' : 'white'}
                  strokeWidth={(isSelected ? 2 : 1) / scale}
                  draggable
                  onClick={(e) => {
                    e.cancelBubble = true;
                    setSelectedPointId(p.id);
                  }}
                  onDblClick={(e) => {
                    e.cancelBubble = true;
                    deletePoint(p.id);
                  }}
                  onContextMenu={(e) => {
                    e.evt.preventDefault();
                    deletePoint(p.id);
                  }}
                  onDragEnd={(e) => {
                    movePoint(p.id, e.target.x(), e.target.y());
                  }}
                />
              );
            })}
          </Layer>
        </Stage>
      )}
      {imageUrl && (
        <div className="pointer-events-none absolute bottom-2 right-2 rounded bg-background/85 px-2 py-1 text-[11px] text-muted-foreground">
          drag to pan · scroll to zoom · drag marker to move · right-click / dbl-click to delete
        </div>
      )}
    </div>
  );
}
