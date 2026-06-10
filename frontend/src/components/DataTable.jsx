import { useMemo, useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import { ArrowDown, ArrowUp, ArrowUpDown, Trash2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';
import { useDigitizerStore } from '@/store/useDigitizerStore';

function EditableNumberCell({ value, onCommit }) {
  const [draft, setDraft] = useState(String(value ?? ''));
  const [editing, setEditing] = useState(false);

  if (!editing) {
    return (
      <button
        type="button"
        className="w-full text-left px-2 py-1 hover:bg-accent rounded"
        onClick={() => {
          setDraft(String(value ?? ''));
          setEditing(true);
        }}
      >
        {Number(value).toFixed(4)}
      </button>
    );
  }

  const commit = () => {
    const n = parseFloat(draft);
    if (Number.isFinite(n)) onCommit(n);
    setEditing(false);
  };

  return (
    <Input
      type="number"
      step="any"
      autoFocus
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit();
        if (e.key === 'Escape') setEditing(false);
      }}
      className="h-8 text-sm"
    />
  );
}

export default function DataTable() {
  const points = useDigitizerStore((s) => s.points);
  const selectedPointId = useDigitizerStore((s) => s.selectedPointId);
  const setSelectedPointId = useDigitizerStore((s) => s.setSelectedPointId);
  const updatePointField = useDigitizerStore((s) => s.updatePointField);
  const deletePoint = useDigitizerStore((s) => s.deletePoint);
  const undo = useDigitizerStore((s) => s.undo);
  const historyLen = useDigitizerStore((s) => s.history.length);

  const [sorting, setSorting] = useState([{ id: 'series_id', desc: false }]);

  const columns = useMemo(
    () => [
      {
        accessorKey: 'series_id',
        header: 'Series',
        cell: ({ row }) => (
          <span className="font-medium">{row.original.series_id}</span>
        ),
      },
      {
        accessorKey: 'x',
        header: 'x',
        cell: ({ row }) => (
          <EditableNumberCell
            value={row.original.x}
            onCommit={(v) => updatePointField(row.original.id, 'x', v)}
          />
        ),
      },
      {
        accessorKey: 'y',
        header: 'y',
        cell: ({ row }) => (
          <EditableNumberCell
            value={row.original.y}
            onCommit={(v) => updatePointField(row.original.id, 'y', v)}
          />
        ),
      },
      {
        accessorKey: 'delta_x',
        header: 'Δx',
        cell: ({ row }) => (
          <span className="text-xs text-muted-foreground">
            {Number(row.original.delta_x).toFixed(3)}
          </span>
        ),
      },
      {
        accessorKey: 'delta_y',
        header: 'Δy',
        cell: ({ row }) => (
          <span className="text-xs text-muted-foreground">
            {Number(row.original.delta_y).toFixed(3)}
          </span>
        ),
      },
      {
        id: 'delete',
        header: '',
        enableSorting: false,
        cell: ({ row }) => (
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="h-7 w-7"
            onClick={(e) => {
              e.stopPropagation();
              deletePoint(row.original.id);
            }}
          >
            <Trash2 className="h-4 w-4 text-destructive" />
          </Button>
        ),
      },
    ],
    [deletePoint, updatePointField],
  );

  const table = useReactTable({
    data: points,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: (row) => row.id,
  });

  return (
    <div className="rounded-lg border bg-card">
      <div className="flex items-center justify-between border-b px-3 py-2">
        <p className="text-sm font-medium">
          {points.length} point{points.length === 1 ? '' : 's'}
        </p>
        <Button
          variant="ghost"
          size="sm"
          onClick={undo}
          disabled={historyLen === 0}
        >
          Undo
        </Button>
      </div>
      <div className="max-h-[420px] overflow-auto">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id}>
                {hg.headers.map((header) => {
                  const sorted = header.column.getIsSorted();
                  const Icon =
                    sorted === 'asc'
                      ? ArrowUp
                      : sorted === 'desc'
                      ? ArrowDown
                      : ArrowUpDown;
                  return (
                    <TableHead
                      key={header.id}
                      onClick={
                        header.column.getCanSort()
                          ? header.column.getToggleSortingHandler()
                          : undefined
                      }
                      className={cn(
                        header.column.getCanSort() &&
                          'cursor-pointer select-none hover:text-foreground',
                      )}
                    >
                      <span className="inline-flex items-center gap-1">
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getCanSort() && (
                          <Icon className="h-3 w-3 opacity-60" />
                        )}
                      </span>
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="text-center text-muted-foreground py-6">
                  No points yet. Run "Detect points" on the left.
                </TableCell>
              </TableRow>
            ) : (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  data-state={selectedPointId === row.id ? 'selected' : undefined}
                  onMouseEnter={() => setSelectedPointId(row.id)}
                  onMouseLeave={() => setSelectedPointId(null)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
