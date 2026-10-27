"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import {
  CANVAS_COLORS,
  LABELED_RANK,
  RANK_SHAPES,
  SURFACE,
  subsetStyle,
  type MarkerShape,
  type PreparedEmbedding,
} from "@/lib/embedding-views";

export type ViewMode = "2d" | "3d";

export interface EmbeddingCanvasHandle {
  resetView(): void;
  /**
   * The current view redrawn off screen for a figure: ``print`` on white with
   * dark inks, no hover/selection chrome; ``scale`` times the on-screen pixels.
   */
  renderImage(options?: { scale?: number; print?: boolean }): HTMLCanvasElement | null;
  focusClass(classIndex: number): void;
  focusPoint(index: number): void;
}

export interface CanvasEdge {
  from: number;
  to: number;
  color: string;
  width: number;
}

export interface TooltipContent {
  title: string;
  lines: string[];
  swatch?: string;
}

interface EmbeddingCanvasProps {
  data: PreparedEmbedding;
  mode: ViewMode;
  /** The 2D positions to draw: the view's own, or one tracked epoch's. */
  layout2d: Float32Array;
  /** The 3D positions, or null when the view has none (3D mode then draws 2D). */
  layout3d: Float32Array | null;
  /** 2D only: lines from each row's position in a reference layout to where it is now. */
  trails: { rows: Int32Array; from: Float32Array } | null;
  /** One flag per entry of `data.subsets`. */
  visibleSubsets: boolean[];
  /** Per point: an index into CANVAS_COLORS. */
  pointColors: Uint8Array;
  /** Per point: 0 dimmed context, 1 normal, 2 emphasized. */
  pointTiers: Uint8Array;
  /** Outline labeled diamonds so they read in any color mode. */
  outlineLabeled: boolean;
  pointSize: number;
  selectedIndex: number | null;
  /** Lines drawn over the points, e.g. a selected point's neighbors. */
  edges: CanvasEdge[];
  /** Per point: the group whose center is drawn (-1 for none), with the group count. */
  centerGroups: Int32Array | null;
  centerCount: number;
  describePoint(index: number): TooltipContent;
  describeCenter(group: number, members: number): TooltipContent;
  onSelect(index: number | null): void;
  onBoxSelect(indices: Int32Array): void;
}

// A style is rank * CANVAS_COLORS.length + color, so buckets drawn in ascending
// style order paint validation first and labeled points last, whatever their color.
const COLOR_COUNT = CANVAS_COLORS.length;
const STYLE_COUNT = RANK_SHAPES.length * COLOR_COUNT;
// Dimmed context, normal, emphasized. Drawn in this order so emphasis lands on top.
const TIER_ALPHA = [0.14, 0.8, 1];
const TIER_COUNT = TIER_ALPHA.length;
const DEPTH_SLABS = 32;
const HIT_RADIUS = 14;
const CENTER_HIT_RADIUS = 10;
const FOCAL = 3.2;
const TEXT_PRIMARY = "#f7f8fc";
const TEXT_MUTED = "#8f9bb2";
const LABELED_OUTLINE = "rgba(247, 248, 252, 0.9)";

interface Inks {
  surface: string;
  primary: string;
  muted: string;
  outline: string;
}

const SCREEN_INKS: Inks = { surface: SURFACE, primary: TEXT_PRIMARY, muted: TEXT_MUTED, outline: LABELED_OUTLINE };
const PRINT_INKS: Inks = { surface: "#ffffff", primary: "#111827", muted: "#4b5563", outline: "rgba(17, 24, 39, 0.85)" };

interface RenderTarget {
  canvas: HTMLCanvasElement;
  dpr: number;
  print: boolean;
}

interface Camera2D {
  cx: number;
  cy: number;
  zoom: number;
}

interface Camera3D {
  yaw: number;
  pitch: number;
  distance: number;
  target: [number, number, number];
}

interface DragState {
  pointerId: number;
  kind: "pan" | "rotate" | "box";
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  moved: boolean;
}

type HoverState =
  | { kind: "point"; index: number; x: number; y: number }
  | { kind: "center"; group: number; members: number; x: number; y: number };

const defaultCamera2D = (): Camera2D => ({ cx: 0, cy: 0, zoom: 1 });
const defaultCamera3D = (): Camera3D => ({ yaw: -0.6, pitch: 0.35, distance: FOCAL, target: [0, 0, 0] });

function clamp(value: number, low: number, high: number) {
  return Math.min(high, Math.max(low, value));
}

function addShape(ctx: CanvasRenderingContext2D, shape: MarkerShape, x: number, y: number, r: number) {
  switch (shape) {
    case "circle":
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
      break;
    case "square":
      ctx.rect(x - r * 0.9, y - r * 0.9, r * 1.8, r * 1.8);
      break;
    case "triangle":
      ctx.moveTo(x, y - r * 1.15);
      ctx.lineTo(x + r * 1.05, y + r * 0.8);
      ctx.lineTo(x - r * 1.05, y + r * 0.8);
      ctx.closePath();
      break;
    case "diamond":
      ctx.moveTo(x, y - r * 1.3);
      ctx.lineTo(x + r * 1.3, y);
      ctx.lineTo(x, y + r * 1.3);
      ctx.lineTo(x - r * 1.3, y);
      ctx.closePath();
      break;
  }
}

export const EmbeddingCanvas = forwardRef<EmbeddingCanvasHandle, EmbeddingCanvasProps>(function EmbeddingCanvas(
  props,
  ref,
) {
  const {
    data,
    mode,
    layout2d,
    layout3d,
    visibleSubsets,
    pointColors,
    pointTiers,
    outlineLabeled,
    pointSize,
    selectedIndex,
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sizeRef = useRef({ width: 0, height: 0, dpr: 1 });
  const camera2DRef = useRef<Camera2D>(defaultCamera2D());
  const camera3DRef = useRef<Camera3D>(defaultCamera3D());
  const dragRef = useRef<DragState | null>(null);
  const boxRef = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const frameRef = useRef<number | null>(null);
  const hoverFrameRef = useRef<number | null>(null);
  const hoverRef = useRef<HoverState | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  // Per-point screen state of the last frame, read back by picking.
  const screenRef = useRef({
    x: new Float32Array(0),
    y: new Float32Array(0),
    radius: new Float32Array(0),
    depth: new Float32Array(0),
    key: new Int32Array(0),
    order: new Int32Array(0),
  });
  // Screen positions of the drawn cluster centers: [group, x, y, members] per center.
  const centerScreenRef = useRef<number[]>([]);
  const styleRef = useRef(new Uint16Array(0));

  const propsRef = useRef(props);
  propsRef.current = props;

  const render = useCallback((target?: RenderTarget) => {
    if (!target) frameRef.current = null;
    const canvas = target?.canvas ?? canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const { width, height } = sizeRef.current;
    const dpr = target?.dpr ?? sizeRef.current.dpr;
    const ink = target?.print ? PRINT_INKS : SCREEN_INKS;
    const exporting = target != null;
    if (width === 0 || height === 0) return;
    const current = propsRef.current;
    const view = current.data;
    const count = view.count;
    const screen = screenRef.current;
    if (screen.x.length !== count) {
      screen.x = new Float32Array(count);
      screen.y = new Float32Array(count);
      screen.radius = new Float32Array(count);
      screen.depth = new Float32Array(count);
      screen.key = new Int32Array(count);
      screen.order = new Int32Array(count);
    }
    const style = styleRef.current;
    const tiers = current.pointTiers;
    if (style.length !== count || tiers.length !== count) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = ink.surface;
    ctx.fillRect(0, 0, width, height);

    const base = Math.min(width, height) * 0.46;
    const centerX = width / 2;
    const centerY = height / 2;
    const use3D = current.mode === "3d" && current.layout3d != null;
    const slabs = use3D ? DEPTH_SLABS : 1;
    const keySpan = TIER_COUNT * slabs * STYLE_COUNT;
    const counts = new Int32Array(keySpan + 1);
    const baseRadius = current.pointSize;
    const visible = current.visibleSubsets;

    const camera3 = camera3DRef.current;
    const cosYaw = Math.cos(camera3.yaw);
    const sinYaw = Math.sin(camera3.yaw);
    const cosPitch = Math.cos(camera3.pitch);
    const sinPitch = Math.sin(camera3.pitch);
    const camera2 = camera2DRef.current;
    const scale2 = camera2.zoom * base;

    /** Screen x, y and depth of a layout position; depth <= 0 means behind the camera. */
    const project = (x: number, y: number, z: number): [number, number, number] => {
      if (!use3D) {
        return [centerX + (x - camera2.cx) * scale2, centerY - (y - camera2.cy) * scale2, 1];
      }
      const px = x - camera3.target[0];
      const py = y - camera3.target[1];
      const pz = z - camera3.target[2];
      const x1 = cosYaw * px + sinYaw * pz;
      const z1 = -sinYaw * px + cosYaw * pz;
      const y2 = cosPitch * py - sinPitch * z1;
      const z2 = sinPitch * py + cosPitch * z1;
      const depth = camera3.distance - z2;
      if (depth < 0.05) return [0, 0, -1];
      const scale = (base * FOCAL) / depth;
      return [centerX + x1 * scale, centerY - y2 * scale, depth];
    };

    const positions2 = current.layout2d;
    const positions3 = current.layout3d;
    let minDepth = Infinity;
    let maxDepth = -Infinity;
    for (let index = 0; index < count; index += 1) {
      screen.key[index] = -1;
      if (!visible[view.subset[index]]) continue;
      const [sx, sy, depth] = use3D
        ? project(positions3![index * 3], positions3![index * 3 + 1], positions3![index * 3 + 2])
        : project(positions2[index * 2], positions2[index * 2 + 1], 0);
      if (depth <= 0) continue;
      if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) continue;
      screen.x[index] = sx;
      screen.y[index] = sy;
      screen.depth[index] = depth;
      screen.radius[index] = use3D ? baseRadius * clamp(FOCAL / depth, 0.5, 2.4) : baseRadius;
      screen.key[index] = 0;
      if (depth < minDepth) minDepth = depth;
      if (depth > maxDepth) maxDepth = depth;
    }

    // Counting sort by (tier, far-to-near slab, style): one fill per bucket keeps
    // the canvas state changes to a few hundred however many points there are.
    const depthSpan = maxDepth - minDepth || 1;
    for (let index = 0; index < count; index += 1) {
      if (screen.key[index] < 0) continue;
      const slab = use3D
        ? slabs - 1 - Math.min(slabs - 1, Math.floor(((screen.depth[index] - minDepth) / depthSpan) * slabs))
        : 0;
      const key = (tiers[index] * slabs + slab) * STYLE_COUNT + style[index];
      screen.key[index] = key;
      counts[key + 1] += 1;
    }
    for (let key = 0; key < keySpan; key += 1) counts[key + 1] += counts[key];
    const offsets = counts.slice(0, keySpan);
    for (let index = 0; index < count; index += 1) {
      const key = screen.key[index];
      if (key < 0) continue;
      screen.order[offsets[key]] = index;
      offsets[key] += 1;
    }

    for (let key = 0; key < keySpan; key += 1) {
      const start = counts[key];
      const end = counts[key + 1];
      if (start === end) continue;
      const styleIndex = key % STYLE_COUNT;
      const tierIndex = Math.floor(key / (slabs * STYLE_COUNT));
      const rank = Math.floor(styleIndex / COLOR_COUNT);
      ctx.globalAlpha = TIER_ALPHA[tierIndex];
      ctx.fillStyle = CANVAS_COLORS[styleIndex % COLOR_COUNT];
      ctx.beginPath();
      for (let position = start; position < end; position += 1) {
        const index = screen.order[position];
        addShape(ctx, RANK_SHAPES[rank], screen.x[index], screen.y[index], screen.radius[index]);
      }
      ctx.fill();
      if (current.outlineLabeled && rank === LABELED_RANK) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = ink.outline;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Movement trails: where a point sat in the reference layout, to where it is now.
    if (current.trails && !use3D) {
      const { rows, from } = current.trails;
      ctx.lineWidth = 1;
      ctx.strokeStyle = ink.muted;
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      for (let offset = 0; offset < rows.length; offset += 1) {
        const row = rows[offset];
        if (screen.key[row] < 0) continue;
        const [fromX, fromY] = project(from[row * 2], from[row * 2 + 1], 0);
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(screen.x[row], screen.y[row]);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Edges between drawn points, over the cloud but under the rings.
    for (const edge of current.edges) {
      if (screen.key[edge.from] < 0 || screen.key[edge.to] < 0) continue;
      ctx.beginPath();
      ctx.moveTo(screen.x[edge.from], screen.y[edge.from]);
      ctx.lineTo(screen.x[edge.to], screen.y[edge.to]);
      ctx.lineWidth = edge.width;
      ctx.strokeStyle = edge.color;
      ctx.globalAlpha = 0.85;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // Cluster centers: each group's visible member nearest the group's mean. A
    // mean alone can land in empty space when a cluster is split across islands;
    // the nearest member always marks a real sample of it.
    const centerScreen: number[] = [];
    if (current.centerGroups && current.centerCount > 0) {
      const groups = current.centerGroups;
      const dimensions = use3D ? 3 : 2;
      const positions = use3D ? positions3! : positions2;
      const sums = new Float64Array(current.centerCount * 3);
      const members = new Int32Array(current.centerCount);
      for (let index = 0; index < count; index += 1) {
        const group = groups[index];
        if (group < 0 || !visible[view.subset[index]]) continue;
        members[group] += 1;
        for (let axis = 0; axis < dimensions; axis += 1) sums[group * 3 + axis] += positions[index * dimensions + axis];
      }
      const nearest = new Int32Array(current.centerCount).fill(-1);
      const nearestDistance = new Float64Array(current.centerCount).fill(Infinity);
      for (let index = 0; index < count; index += 1) {
        const group = groups[index];
        if (group < 0 || !visible[view.subset[index]]) continue;
        let distance = 0;
        for (let axis = 0; axis < dimensions; axis += 1) {
          const delta = positions[index * dimensions + axis] - sums[group * 3 + axis] / members[group];
          distance += delta * delta;
        }
        if (distance < nearestDistance[group]) {
          nearestDistance[group] = distance;
          nearest[group] = index;
        }
      }
      for (let group = 0; group < current.centerCount; group += 1) {
        const member = nearest[group];
        if (member < 0 || screen.key[member] < 0) continue;
        const sx = screen.x[member];
        const sy = screen.y[member];
        const arm = clamp(2.5 + Math.sqrt(members[group]) * 0.6, 3, 7);
        ctx.beginPath();
        ctx.moveTo(sx - arm, sy);
        ctx.lineTo(sx + arm, sy);
        ctx.moveTo(sx, sy - arm);
        ctx.lineTo(sx, sy + arm);
        ctx.strokeStyle = ink.surface;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.strokeStyle = ink.primary;
        ctx.lineWidth = 1.25;
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1;
        centerScreen.push(group, sx, sy, members[group]);
      }
    }
    if (!exporting) centerScreenRef.current = centerScreen;

    const ring = (index: number, color: string, extra: number) => {
      if (index < 0 || index >= count || screen.key[index] < 0) return;
      const radius = screen.radius[index] * 1.3 + extra + 2;
      ctx.beginPath();
      ctx.arc(screen.x[index], screen.y[index], radius, 0, Math.PI * 2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = ink.surface;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(screen.x[index], screen.y[index], radius, 0, Math.PI * 2);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    const hovered = exporting ? null : hoverRef.current;
    if (hovered?.kind === "point" && hovered.index !== current.selectedIndex) ring(hovered.index, ink.muted, 1);
    if (current.selectedIndex != null && !target?.print) ring(current.selectedIndex, ink.primary, 3);

    const box = exporting ? null : boxRef.current;
    if (box) {
      const x = Math.min(box.x0, box.x1);
      const y = Math.min(box.y0, box.y1);
      ctx.fillStyle = "rgba(155, 140, 255, 0.08)";
      ctx.fillRect(x, y, Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0));
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#9b8cff";
      ctx.strokeRect(x + 0.5, y + 0.5, Math.abs(box.x1 - box.x0), Math.abs(box.y1 - box.y0));
    }

    if (use3D) {
      // Orientation gizmo: the three projection axes, rotated but not projected.
      const originX = 40;
      const originY = height - 40;
      ctx.font = "10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const [label, x, y, z] of [
        ["1", 1, 0, 0],
        ["2", 0, 1, 0],
        ["3", 0, 0, 1],
      ] as [string, number, number, number][]) {
        const x1 = cosYaw * x + sinYaw * z;
        const z1 = -sinYaw * x + cosYaw * z;
        const y2 = cosPitch * y - sinPitch * z1;
        ctx.beginPath();
        ctx.moveTo(originX, originY);
        ctx.lineTo(originX + x1 * 22, originY - y2 * 22);
        ctx.lineWidth = 1;
        ctx.strokeStyle = ink.muted;
        ctx.stroke();
        ctx.fillStyle = ink.muted;
        ctx.fillText(label, originX + x1 * 30, originY - y2 * 30);
      }
    }
  }, []);

  const draw = useCallback(() => render(), [render]);

  const requestDraw = useCallback(() => {
    if (frameRef.current == null) frameRef.current = requestAnimationFrame(draw);
  }, [draw]);

  // Style = draw rank of the subset * colors + color; recomputed when colors change.
  useEffect(() => {
    const ranks = data.subsets.map((name) => subsetStyle(name).rank);
    const style = new Uint16Array(data.count);
    for (let index = 0; index < data.count; index += 1) {
      style[index] = ranks[data.subset[index]] * COLOR_COUNT + pointColors[index];
    }
    styleRef.current = style;
    requestDraw();
  }, [data, pointColors, requestDraw]);

  useEffect(() => {
    requestDraw();
  }, [
    mode,
    layout2d,
    layout3d,
    props.trails,
    visibleSubsets,
    pointTiers,
    outlineLabeled,
    pointSize,
    selectedIndex,
    props.edges,
    props.centerGroups,
    requestDraw,
  ]);

  const resetView = useCallback(() => {
    camera2DRef.current = defaultCamera2D();
    camera3DRef.current = defaultCamera3D();
    requestDraw();
  }, [requestDraw]);

  // A different view starts from the default camera. A live view that only
  // gained an epoch keeps the user's pan and zoom.
  const viewKey = `${data.id}|${data.count}`;
  useEffect(() => {
    hoverRef.current = null;
    setHover(null);
    resetView();
  }, [viewKey, resetView]);

  const focusOn = useCallback(
    (includes: (index: number) => boolean, minimumExtent: number) => {
      const { data: view, mode: viewMode, layout2d: positions2, layout3d: positions3 } = propsRef.current;
      const use3D = viewMode === "3d" && positions3 != null;
      const dimensions = use3D ? 3 : 2;
      const positions = use3D ? positions3 : positions2;
      const low = new Array<number>(dimensions).fill(Infinity);
      const high = new Array<number>(dimensions).fill(-Infinity);
      for (let index = 0; index < view.count; index += 1) {
        if (!includes(index)) continue;
        for (let axis = 0; axis < dimensions; axis += 1) {
          const value = positions[index * dimensions + axis];
          low[axis] = Math.min(low[axis], value);
          high[axis] = Math.max(high[axis], value);
        }
      }
      if (!Number.isFinite(low[0])) return;
      const center = low.map((value, axis) => (value + high[axis]) / 2);
      const extent = Math.max(minimumExtent, ...high.map((value, axis) => value - low[axis]));
      if (use3D) {
        camera3DRef.current = {
          ...camera3DRef.current,
          target: [center[0], center[1], center[2]],
          distance: clamp(extent * 2.6, 0.4, 30),
        };
      } else {
        camera2DRef.current = { cx: center[0], cy: center[1], zoom: clamp(1.6 / extent, 0.2, 400) };
      }
      requestDraw();
    },
    [requestDraw],
  );

  useImperativeHandle(
    ref,
    () => ({
      resetView,
      renderImage({ scale = 4, print = true } = {}) {
        const { width, height } = sizeRef.current;
        if (width === 0 || height === 0) return null;
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(width * scale);
        canvas.height = Math.round(height * scale);
        render({ canvas, dpr: scale, print });
        return canvas;
      },
      focusClass(classIndex: number) {
        const view = propsRef.current.data;
        focusOn((index) => view.classIndex[index] === classIndex, 0.05);
      },
      focusPoint(target: number) {
        focusOn((index) => index === target, 0.3);
      },
    }),
    [focusOn, render, resetView],
  );

  // Size the canvas to its container at device resolution.
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;
    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { width: rect.width, height: rect.height, dpr };
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      requestDraw();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [requestDraw]);

  useEffect(
    () => () => {
      // Clear the ids too: a remount (React's strict-mode double effect included)
      // would otherwise see a "pending" frame that was cancelled and never draw.
      if (frameRef.current != null) cancelAnimationFrame(frameRef.current);
      if (hoverFrameRef.current != null) cancelAnimationFrame(hoverFrameRef.current);
      frameRef.current = null;
      hoverFrameRef.current = null;
    },
    [],
  );

  /** A cluster center within reach wins; otherwise the nearest emphasized-first point. */
  const pick = useCallback((x: number, y: number): HoverState | null => {
    const centers = centerScreenRef.current;
    let bestCenter = -1;
    let bestCenterDistance = CENTER_HIT_RADIUS * CENTER_HIT_RADIUS;
    for (let offset = 0; offset < centers.length; offset += 4) {
      const dx = centers[offset + 1] - x;
      const dy = centers[offset + 2] - y;
      const distance = dx * dx + dy * dy;
      if (distance <= bestCenterDistance) {
        bestCenter = offset;
        bestCenterDistance = distance;
      }
    }
    if (bestCenter >= 0) {
      return { kind: "center", group: centers[bestCenter], members: centers[bestCenter + 3], x, y };
    }
    const screen = screenRef.current;
    const tiers = propsRef.current.pointTiers;
    let best = -1;
    let bestDistance = HIT_RADIUS * HIT_RADIUS;
    let bestDepth = Infinity;
    let bestTier = -1;
    for (let index = 0; index < screen.key.length; index += 1) {
      if (screen.key[index] < 0) continue;
      const dx = screen.x[index] - x;
      const dy = screen.y[index] - y;
      const distance = dx * dx + dy * dy;
      if (distance > HIT_RADIUS * HIT_RADIUS) continue;
      const pointTier = tiers[index] ?? 0;
      const closer = distance < bestDistance - 4;
      const tie = Math.abs(distance - bestDistance) <= 4;
      if (
        best < 0 ||
        (pointTier > bestTier && distance < bestDistance + 36) ||
        (pointTier === bestTier && (closer || (tie && screen.depth[index] < bestDepth)))
      ) {
        best = index;
        bestDistance = distance;
        bestDepth = screen.depth[index];
        bestTier = pointTier;
      }
    }
    return best < 0 ? null : { kind: "point", index: best, x, y };
  }, []);

  const localPoint = (event: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    return rect ? { x: event.clientX - rect.left, y: event.clientY - rect.top } : { x: 0, y: 0 };
  };

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
    const { x, y } = localPoint(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    const panModifier = event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey;
    dragRef.current = {
      pointerId: event.pointerId,
      kind: event.shiftKey ? "box" : propsRef.current.mode === "3d" && !panModifier ? "rotate" : "pan",
      startX: x,
      startY: y,
      lastX: x,
      lastY: y,
      moved: false,
    };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = localPoint(event);
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId) {
      const dx = x - drag.lastX;
      const dy = y - drag.lastY;
      drag.lastX = x;
      drag.lastY = y;
      if (Math.abs(x - drag.startX) + Math.abs(y - drag.startY) > 3) drag.moved = true;
      if (!drag.moved) return;
      const { width, height } = sizeRef.current;
      const base = Math.min(width, height) * 0.46;
      if (drag.kind === "box") {
        boxRef.current = { x0: drag.startX, y0: drag.startY, x1: x, y1: y };
      } else if (drag.kind === "rotate") {
        const camera = camera3DRef.current;
        camera.yaw += dx * 0.008;
        camera.pitch = clamp(camera.pitch + dy * 0.008, -1.55, 1.55);
      } else if (propsRef.current.mode === "3d") {
        const camera = camera3DRef.current;
        const perPixel = camera.distance / (base * FOCAL);
        const cosYaw = Math.cos(camera.yaw);
        const sinYaw = Math.sin(camera.yaw);
        const cosPitch = Math.cos(camera.pitch);
        const sinPitch = Math.sin(camera.pitch);
        const right = [cosYaw, 0, sinYaw];
        const up = [sinPitch * sinYaw, cosPitch, -sinPitch * cosYaw];
        camera.target = camera.target.map(
          (value, axis) => value - right[axis] * dx * perPixel + up[axis] * dy * perPixel,
        ) as Camera3D["target"];
      } else {
        const camera = camera2DRef.current;
        camera.cx -= dx / (camera.zoom * base);
        camera.cy += dy / (camera.zoom * base);
      }
      if (hoverRef.current) {
        hoverRef.current = null;
        setHover(null);
      }
      requestDraw();
      return;
    }
    if (hoverFrameRef.current != null) return;
    hoverFrameRef.current = requestAnimationFrame(() => {
      hoverFrameRef.current = null;
      const next = pick(x, y);
      const previous = hoverRef.current;
      const changed =
        next?.kind !== previous?.kind ||
        (next?.kind === "point" && previous?.kind === "point" && next.index !== previous.index) ||
        (next?.kind === "center" && previous?.kind === "center" && next.group !== previous.group);
      if (changed) requestDraw();
      hoverRef.current = next;
      setHover(next);
    });
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    const { x, y } = localPoint(event);
    if (drag.kind === "box" && drag.moved) {
      const box = boxRef.current;
      boxRef.current = null;
      if (box) {
        const screen = screenRef.current;
        const left = Math.min(box.x0, box.x1);
        const right = Math.max(box.x0, box.x1);
        const top = Math.min(box.y0, box.y1);
        const bottom = Math.max(box.y0, box.y1);
        const picked: number[] = [];
        for (let index = 0; index < screen.key.length; index += 1) {
          if (screen.key[index] < 0) continue;
          const px = screen.x[index];
          const py = screen.y[index];
          if (px >= left && px <= right && py >= top && py <= bottom) picked.push(index);
        }
        propsRef.current.onBoxSelect(Int32Array.from(picked));
      }
      requestDraw();
      return;
    }
    if (!drag.moved) {
      const target = pick(x, y);
      propsRef.current.onSelect(target?.kind === "point" ? target.index : null);
    }
  };

  // Wheel zoom needs a non-passive listener to keep the page from scrolling.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.0015);
      if (propsRef.current.mode === "3d") {
        const camera = camera3DRef.current;
        camera.distance = clamp(camera.distance / factor, 0.25, 40);
      } else {
        const rect = canvas.getBoundingClientRect();
        const { width, height } = sizeRef.current;
        const base = Math.min(width, height) * 0.46;
        const camera = camera2DRef.current;
        const mouseX = event.clientX - rect.left - width / 2;
        const mouseY = event.clientY - rect.top - height / 2;
        const worldX = camera.cx + mouseX / (camera.zoom * base);
        const worldY = camera.cy - mouseY / (camera.zoom * base);
        const zoom = clamp(camera.zoom * factor, 0.2, 400);
        camera2DRef.current = {
          zoom,
          cx: worldX - mouseX / (zoom * base),
          cy: worldY + mouseY / (zoom * base),
        };
      }
      requestDraw();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [requestDraw]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLCanvasElement>) => {
    const step = event.shiftKey ? 0.2 : 0.06;
    const camera2 = camera2DRef.current;
    const camera3 = camera3DRef.current;
    const is3D = propsRef.current.mode === "3d";
    switch (event.key) {
      case "r":
      case "R":
        resetView();
        break;
      case "Escape":
        propsRef.current.onSelect(null);
        break;
      case "+":
      case "=":
        if (is3D) camera3.distance = clamp(camera3.distance / 1.2, 0.25, 40);
        else camera2.zoom = clamp(camera2.zoom * 1.2, 0.2, 400);
        break;
      case "-":
      case "_":
        if (is3D) camera3.distance = clamp(camera3.distance * 1.2, 0.25, 40);
        else camera2.zoom = clamp(camera2.zoom / 1.2, 0.2, 400);
        break;
      case "ArrowLeft":
        if (is3D) camera3.yaw -= step * 2;
        else camera2.cx -= step / camera2.zoom;
        break;
      case "ArrowRight":
        if (is3D) camera3.yaw += step * 2;
        else camera2.cx += step / camera2.zoom;
        break;
      case "ArrowUp":
        if (is3D) camera3.pitch = clamp(camera3.pitch - step * 2, -1.55, 1.55);
        else camera2.cy += step / camera2.zoom;
        break;
      case "ArrowDown":
        if (is3D) camera3.pitch = clamp(camera3.pitch + step * 2, -1.55, 1.55);
        else camera2.cy -= step / camera2.zoom;
        break;
      default:
        return;
    }
    event.preventDefault();
    requestDraw();
  };

  let tooltip: TooltipContent | null = null;
  if (hover?.kind === "point" && hover.index < data.count) tooltip = props.describePoint(hover.index);
  if (hover?.kind === "center") tooltip = props.describeCenter(hover.group, hover.members);
  const tooltipLeft = hover ? (hover.x > sizeRef.current.width - 260 ? hover.x - 252 : hover.x + 14) : 0;
  const tooltipTop = hover ? (hover.y > sizeRef.current.height - 140 ? hover.y - 132 : hover.y + 14) : 0;

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-2xl">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        role="img"
        aria-label={`${mode === "3d" ? "3D" : "2D"} embedding projection of ${data.count} points. Drag to ${
          mode === "3d" ? "rotate" : "pan"
        }, scroll to zoom, shift-drag to select a region, click a point to select it, R to reset.`}
        className="focus-ring block touch-none select-none"
        style={{ cursor: hover ? "pointer" : mode === "3d" ? "grab" : "move" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => {
          dragRef.current = null;
          boxRef.current = null;
          requestDraw();
        }}
        onPointerLeave={() => {
          if (dragRef.current) return;
          hoverRef.current = null;
          setHover(null);
          requestDraw();
        }}
        onDoubleClick={resetView}
        onContextMenu={(event) => event.preventDefault()}
        onKeyDown={onKeyDown}
      />
      {tooltip && hover && (
        <div
          className="pointer-events-none absolute z-10 w-[240px] rounded-xl border border-white/[0.08] bg-[#10182a]/95 px-3 py-2 shadow-xl"
          style={{ left: tooltipLeft, top: tooltipTop }}
        >
          <p className="flex items-center gap-2 truncate text-sm font-semibold text-[#f7f8fc]">
            {tooltip.swatch && (
              <span aria-hidden className="inline-block h-[2px] w-3 shrink-0 rounded-full" style={{ background: tooltip.swatch }} />
            )}
            <span className="truncate">{tooltip.title}</span>
          </p>
          {tooltip.lines.map((line) => (
            <p key={line} className="mt-0.5 text-[11px] leading-snug text-[#98a4ba]">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  );
});
