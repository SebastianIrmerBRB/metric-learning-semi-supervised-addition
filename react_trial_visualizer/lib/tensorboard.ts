import { normalizeColumn, type Study, type TrialRow } from "@/lib/trials";

export type ScalarPoint = [step: number, value: number, wallTime: number];

export interface TensorboardSeries {
  tag: string;
  label: string;
  x_label: string;
  point_count: number;
  sampled_count: number;
  points: ScalarPoint[];
}

export interface TensorboardFold {
  id: string;
  label: string;
  event_file_count: number;
  series: {
    validation: TensorboardSeries | null;
    batch_loss: TensorboardSeries | null;
    epoch_loss: TensorboardSeries | null;
  };
}

export interface TensorboardTrialData {
  status: "ok";
  log_dir: string;
  resolved_path: string;
  event_file_count: number;
  folds: TensorboardFold[];
  warnings: string[];
}

interface TensorboardError {
  status: "error";
  message: string;
}

export function trialLogDirectory(study: Study, row: TrialRow | null): string | null {
  if (!row) return null;
  const column = study.columns.find((candidate) =>
    ["attr_log_dir", "log_dir", "tensorboard_log_dir"].includes(normalizeColumn(candidate)),
  );
  if (!column) return null;
  const value = row[column]?.trim();
  if (!value) return null;
  return value.replace(/^(["'])|(["'])$/g, "");
}

function apiOrigin(): string {
  const configured = process.env.NEXT_PUBLIC_TENSORBOARD_API?.replace(/\/$/, "");
  if (configured) return configured;
  if (typeof window === "undefined") return "http://127.0.0.1:8765";
  const host = window.location.hostname || "127.0.0.1";
  return `http://${host}:8765`;
}

export async function fetchTensorboardTrial(
  logDirectory: string,
  signal?: AbortSignal,
): Promise<TensorboardTrialData> {
  const query = new URLSearchParams({ log_dir: logDirectory, max_points: "2000" });
  const response = await fetch(`${apiOrigin()}/api/tensorboard/trial?${query}`, {
    signal,
    cache: "no-store",
  });

  let payload: TensorboardTrialData | TensorboardError;
  try {
    payload = (await response.json()) as TensorboardTrialData | TensorboardError;
  } catch {
    throw new Error("The TensorBoard service returned an unreadable response.");
  }
  if (!response.ok || payload.status === "error") {
    throw new Error(payload.status === "error" ? payload.message : `TensorBoard request failed (${response.status}).`);
  }
  return payload;
}
