/**
 * Types and pure helpers for the epoch-tracking snapshot.
 *
 * The snapshot is written by `evaluate/export_react.py`, which shares its
 * discovery and metric grouping with the CSV and HTML readers, so this view and
 * the thesis tables can never disagree about what a run is called or what its
 * baseline was.
 *
 * Series arrive pre-aggregated over folds as `[epoch, mean, min, max]`: the
 * browser has no reason to carry per-fold rows it would only average again.
 */

/** `[epoch, mean, min, max]` for one tracked epoch. */
export type SeriesTuple = [number, number, number, number];

export interface EpochTrackingSummaryRow {
  metric: string;
  label: string;
  baseline: number | null;
  final: number | null;
  delta: number | null;
  folds: number;
  baselineEpoch: number;
}

export interface EpochTrackingRun {
  id: string;
  label: string;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  seed: number | null;
  variant: string | null;
  layout: "pca" | "pacmap" | "tsne" | null;
  folds: number[];
  epochs: number[];
  series: Record<string, SeriesTuple[]>;
  summary: EpochTrackingSummaryRow[];
  filmstrips: EpochTrackingFilmstrip[];
}

export interface EpochTrackingFilmstrip {
  /** `-1` denotes a run without cross-validation folds. */
  fold: number;
  graph?: string;
  embeddings?: string;
}

export interface OverviewFigure {
  name: string;
  file: string;
  caption: string;
}

export interface EpochTrackingSnapshot {
  generatedAt: string;
  root: string;
  runCount: number;
  foldCount: number;
  metrics: string[];
  metricLabels: Record<string, string>;
  metricGroups: Record<string, string[]>;
  higherIsBetter: Record<string, boolean>;
  baselineEpoch: number;
  overviewFigures: OverviewFigure[];
  runs: EpochTrackingRun[];
}

export const METRIC_GROUP_LABELS: Record<string, string> = {
  test: "Test performance",
  embedding: "Embedding geometry",
  graph: "Graph structure",
};

export const METRIC_GROUP_BLURBS: Record<string, string> = {
  test:
    "D_test measured off the selection path. Nothing here was read back into training, early stopping, or Optuna.",
  embedding:
    "Whether the representation is spreading out or collapsing. Effective rank near 1 means the variance has fallen into a single direction.",
  graph:
    "The kNN graph over the SSL pool, rebuilt from the current embeddings with the method's own parameters. Purity is the share of edges joining two samples of the same true class.",
};

/** Whether a change is an improvement, for metrics that have a direction. */
export type DeltaDirection = "better" | "worse" | "neutral";

export function deltaDirection(
  metric: string,
  delta: number | null,
  higherIsBetter: Record<string, boolean>,
): DeltaDirection {
  if (delta == null || Math.abs(delta) < 1e-12) return "neutral";
  const better = higherIsBetter[metric];
  if (better == null) return "neutral";
  return (better ? delta > 0 : delta < 0) ? "better" : "worse";
}

export function formatValue(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "--";
  if (Math.abs(value) >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return value.toFixed(digits);
}

export function formatDelta(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatValue(value, digits)}`;
}

export interface Extent {
  min: number;
  max: number;
}

/** The value range of a set of series, padded so a flat line is still visible. */
export function seriesExtent(series: SeriesTuple[][], useBand = true): Extent {
  const values: number[] = [];
  for (const points of series) {
    for (const [, mean, low, high] of points) {
      values.push(mean);
      if (useBand) {
        values.push(low, high);
      }
    }
  }
  if (values.length === 0) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const delta = Math.abs(min) * 0.05 || 1;
    return { min: min - delta, max: max + delta };
  }
  const padding = (max - min) * 0.08;
  min -= padding;
  max += padding;
  return { min, max };
}

export function epochExtent(series: SeriesTuple[][]): Extent {
  const epochs = series.flatMap((points) => points.map(([epoch]) => epoch));
  if (epochs.length === 0) return { min: -1, max: 1 };
  const min = Math.min(...epochs);
  const max = Math.max(...epochs);
  return min === max ? { min: min - 1, max: max + 1 } : { min, max };
}

export function scale(value: number, from: Extent, min: number, max: number): number {
  if (from.max === from.min) return (min + max) / 2;
  return min + ((value - from.min) / (from.max - from.min)) * (max - min);
}

/** Ticks that always include the endpoints, for a small axis. */
export function ticks(extent: Extent, count = 4): number[] {
  if (count < 2) return [extent.min, extent.max];
  const step = (extent.max - extent.min) / (count - 1);
  return Array.from({ length: count }, (_, index) => extent.min + step * index);
}

export function runMatchesQuery(run: EpochTrackingRun, query: string): boolean {
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return [run.label, run.dataset, run.method, run.loss, run.variant]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toLowerCase().includes(needle));
}
