/** Phase-2 snapshots written by main_eval/ablation_report.py and main_eval/geometry_tracking.py. */

export interface PairedDiff {
  n: number;
  mean?: number;
  sd?: number;
  se?: number;
  p?: number | null;
  positive?: number;
  seeds?: number[];
}

export interface ValueSummary {
  n: number;
  mean: number;
  sd?: number;
  per_seed: Record<string, number>;
}

export type Aggregation = "mean" | "concat";

export interface AblationVariant {
  dataset: string;
  k: number;
  loss: string;
  method: string;
  variant: string;
  key: string;
  label_de: string;
  label_en: string;
  group: string;
  cell: string | null;
  uses_hidden_labels: boolean;
  description: string;
  mean?: ValueSummary;
  concat?: ValueSummary;
  mean_vs_baseline?: PairedDiff;
  concat_vs_baseline?: PairedDiff;
  mean_vs_supervised?: PairedDiff;
  concat_vs_supervised?: PairedDiff;
  /** One-at-a-time sweep point: parameter, its value in this variant and the tuned value of the baseline. */
  sweep?: { param: string; value: number; tuned: number | null };
  /** LP only: accepted pseudo-label accuracy and coverage, measured with the hidden labels (diagnostic). */
  pseudo_label_quality?: {
    accuracy_first: number | null;
    accuracy_last: number | null;
    accepted_share_last: number | null;
    pair_precision_last: number | null;
    seeds: number;
  };
}

export interface TwoByTwoAggregate {
  cells: Record<string, ValueSummary>;
  total: PairedDiff;
  data_effect_ssl_hp: PairedDiff;
  data_effect_sup_hp: PairedDiff;
  hp_effect_no_data: PairedDiff;
  hp_effect_with_data: PairedDiff;
  interaction: PairedDiff;
}

export interface TwoByTwoBlock {
  dataset: string;
  k: number;
  loss: string;
  method: string;
  mean?: TwoByTwoAggregate;
  concat?: TwoByTwoAggregate;
}

export interface AblationSnapshot {
  generatedAt: string;
  metric: string;
  metricLabel: string;
  roots: string[];
  labels: { method: Record<string, string>; loss: Record<string, string>; dataset: Record<string, string> };
  variants: AblationVariant[];
  twoByTwo: TwoByTwoBlock[];
}

export interface GeometryCurve {
  root: string;
  combo: string;
  method: string;
  loss: string;
  k: number;
  seed: number;
  split: string;
  epochs: number[];
  folds: number;
  selectedEpoch: number | null;
  series: Record<string, (number | null)[]>;
  backbone: Record<string, number | null>;
}

export interface GeometryHypothesisRow {
  root: string;
  loss: string;
  k: number;
  split: string;
  method: string;
  folds: number;
  [key: string]: string | number | null;
}

export interface GeometrySpectrum {
  root: string;
  combo: string;
  k: number;
  split: string;
  shares: number[];
}

export interface GeometrySnapshot {
  generatedAt: string;
  roots: string[];
  metrics: string[];
  metricLabels: Record<string, string>;
  splits: string[];
  splitLabels: Record<string, string>;
  curves: GeometryCurve[];
  hypothesis: GeometryHypothesisRow[];
  spectra: GeometrySpectrum[];
  /** Roth et al.'s rho_spectrum@0: normalized singular values, uncentered (their Fig. 7). */
  spectraRoth?: GeometrySpectrum[];
  /** Roth et al.'s class-conditioned spectrum: first 35 per-class singular values, class mean (supp. E, Fig. 11). */
  classSpectraRoth?: GeometrySpectrum[];
  /** Roth et al.'s Fig. 5: correlation of a measure with test R@1 across runs. */
  correlation?: GeometryCorrelationRow[];
  /** The points behind `correlation`: one per run and split. */
  correlationPoints?: GeometryCorrelationPoint[];
  /** Correlation targets: key -> label, e.g. { map_r: "MAP@R", r1: "R@1" }. */
  targets?: Record<string, string>;
}

export interface GeometryCorrelationPoint {
  root: string;
  combo: string;
  method: string;
  k: number;
  seed: number;
  split: string;
  r1: number | null;
  map_r?: number | null;
  [metric: string]: string | number | null | undefined;
}

export interface GeometryCorrelationRow {
  root: string;
  split: string;
  k: number | "all";
  /** "map_r" (thesis primary) or "r1" (Roth et al.). Older snapshots have no target: R@1. */
  target?: string;
  metric: string;
  n: number;
  pearson: number | null;
  spearman: number | null;
}

export type EffectKey = Exclude<keyof TwoByTwoAggregate, "cells">;

export const EFFECTS: { key: EffectKey; label: string; hint: string }[] = [
  { key: "total", label: "Gesamtgewinn", hint: "SSL (eigene HPs) − überwacht (eigene HPs)" },
  { key: "data_effect_ssl_hp", label: "Daten | SSL-HPs", hint: "mit − ohne ungelabelte Daten, SSL-HPs" },
  { key: "data_effect_sup_hp", label: "Daten | überwachte HPs", hint: "mit − ohne ungelabelte Daten, überwachte HPs" },
  { key: "hp_effect_no_data", label: "HPs | ohne Daten", hint: "SSL-HPs − überwachte HPs, ohne ungelabelte Daten" },
  { key: "hp_effect_with_data", label: "HPs | mit Daten", hint: "SSL-HPs − überwachte HPs, mit ungelabelten Daten" },
  { key: "interaction", label: "Interaktion", hint: "Daten-Effekt(SSL-HPs) − Daten-Effekt(überwachte HPs)" },
];

export const CELLS: { key: string; row: "sup_hp" | "ssl_hp"; col: "no_data" | "with_data" }[] = [
  { key: "sup_hp/no_data", row: "sup_hp", col: "no_data" },
  { key: "sup_hp/with_data", row: "sup_hp", col: "with_data" },
  { key: "ssl_hp/no_data", row: "ssl_hp", col: "no_data" },
  { key: "ssl_hp/with_data", row: "ssl_hp", col: "with_data" },
];

export function fmt(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return value.toFixed(digits);
}

export function fmtSigned(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function fmtP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "";
  return p < 0.001 ? "p<0.001" : `p=${p.toFixed(3)}`;
}
