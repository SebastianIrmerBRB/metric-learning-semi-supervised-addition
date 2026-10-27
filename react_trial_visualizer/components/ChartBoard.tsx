"use client";

import { useEffect, useMemo, useRef, useState, type DragEvent, type ReactNode } from "react";
import { DiagramCard } from "@/components/Charts";

export interface ChartBoardItem {
  id: string;
  title: string;
  description: string;
  exportName: string;
  content: ReactNode;
  className?: string;
}

interface StoredChartLayout {
  order: string[];
  collapsed: Record<string, boolean>;
}

function normalizeLayout(value: Partial<StoredChartLayout> | null, ids: string[]): StoredChartLayout {
  const storedOrder = Array.isArray(value?.order)
    ? value.order.filter((id): id is string => typeof id === "string" && ids.includes(id))
    : [];
  const order = [...storedOrder, ...ids.filter((id) => !storedOrder.includes(id))];
  const storedCollapsed = value?.collapsed && typeof value.collapsed === "object" ? value.collapsed : {};
  const collapsed = Object.fromEntries(
    ids.map((id) => [id, storedCollapsed[id] === true]),
  );
  return { order, collapsed };
}

export function ChartBoard({
  storageKey,
  items,
  className = "grid gap-6 xl:grid-cols-2",
}: {
  storageKey: string;
  items: ChartBoardItem[];
  className?: string;
}) {
  const ids = useMemo(() => items.map((item) => item.id), [items]);
  const idsSignature = ids.join("\u001f");
  const [layout, setLayout] = useState<StoredChartLayout>(() => normalizeLayout(null, ids));
  const [storageReady, setStorageReady] = useState(false);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const draggingIdRef = useRef<string | null>(null);

  useEffect(() => {
    setStorageReady(false);
    let stored: Partial<StoredChartLayout> | null = null;
    try {
      const raw = window.localStorage.getItem(storageKey);
      stored = raw ? (JSON.parse(raw) as Partial<StoredChartLayout>) : null;
    } catch {
      stored = null;
    }
    setLayout(normalizeLayout(stored, ids));
    setStorageReady(true);
  }, [idsSignature, storageKey]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(layout));
    } catch {
      // The dashboard still works when private browsing or browser policy blocks storage.
    }
  }, [layout, storageKey, storageReady]);

  const orderedItems = layout.order
    .map((id) => items.find((item) => item.id === id))
    .filter((item): item is ChartBoardItem => item !== undefined);

  const move = (id: string, direction: -1 | 1) => {
    setLayout((current) => {
      const index = current.order.indexOf(id);
      const destination = index + direction;
      if (index < 0 || destination < 0 || destination >= current.order.length) return current;
      const order = [...current.order];
      [order[index], order[destination]] = [order[destination], order[index]];
      return { ...current, order };
    });
  };

  const startDrag = (event: DragEvent<HTMLDivElement>, id: string) => {
    const target = event.target as HTMLElement;
    if (!target.closest("[data-chart-drag-handle]")) {
      event.preventDefault();
      return;
    }
    draggingIdRef.current = id;
    setDraggingId(id);
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", id);
    event.dataTransfer.setDragImage(event.currentTarget, 28, 24);
  };

  const drop = (event: DragEvent<HTMLDivElement>, targetId: string) => {
    event.preventDefault();
    const sourceId = draggingIdRef.current ?? event.dataTransfer.getData("text/plain");
    if (!sourceId || sourceId === targetId) {
      draggingIdRef.current = null;
      setDraggingId(null);
      setDropTargetId(null);
      return;
    }
    setLayout((current) => {
      const order = current.order.filter((id) => id !== sourceId);
      const targetIndex = order.indexOf(targetId);
      order.splice(targetIndex < 0 ? order.length : targetIndex, 0, sourceId);
      return { ...current, order };
    });
    draggingIdRef.current = null;
    setDraggingId(null);
    setDropTargetId(null);
  };

  return (
    <div className={className}>
      {orderedItems.map((item, index) => (
        <div
          key={item.id}
          onDragStart={(event) => startDrag(event, item.id)}
          onDragOver={(event) => {
            if (!draggingIdRef.current || draggingIdRef.current === item.id) return;
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
            setDropTargetId(item.id);
          }}
          onDrop={(event) => drop(event, item.id)}
          onDragEnd={() => {
            draggingIdRef.current = null;
            setDraggingId(null);
            setDropTargetId(null);
          }}
          className={`min-w-0 rounded-[1.25rem] transition ${item.className ?? ""} ${
            draggingId === item.id ? "opacity-45" : "opacity-100"
          } ${dropTargetId === item.id ? "ring-2 ring-[#9b8cff]/70 ring-offset-4 ring-offset-[#080d18]" : ""}`}
        >
          <DiagramCard
            title={item.title}
            description={item.description}
            exportName={item.exportName}
            collapsed={layout.collapsed[item.id] === true}
            onCollapsedChange={(collapsed) =>
              setLayout((current) => ({
                ...current,
                collapsed: { ...current.collapsed, [item.id]: collapsed },
              }))
            }
            reorderable={items.length > 1}
            onMove={(direction) => move(item.id, direction)}
            canMoveUp={index > 0}
            canMoveDown={index < orderedItems.length - 1}
          >
            {item.content}
          </DiagramCard>
        </div>
      ))}
    </div>
  );
}
