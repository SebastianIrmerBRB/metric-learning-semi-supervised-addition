import { useCallback, useEffect, useRef, useState } from "react";

import {
  prepareEmbedding,
  type EdgeListFile,
  type EmbeddingViewFile,
  type EvidenceFile,
  type MinedPairsFile,
  type PreparedEmbedding,
  type PropagationFile,
  type PseudoLabelFile,
} from "@/lib/embedding-views";
import { alignSeries } from "@/lib/procrustes";

/**
 * Epoch tracking watched while the run trains.
 *
 * The dev server (app/api/live-tracking/) finds every epoch_tracking/ below
 * logs/ and serves it in three pieces: a manifest of file stats, polled every
 * couple of seconds; the base (tracked points, classes, subsets), fetched once;
 * and one frame per epoch, fetched only when that epoch's files changed. The
 * page aligns the frames and hands the result to the same viewer that draws
 * exported views, so a live view looks and behaves like an exported one and
 * grows an epoch as soon as the tracker writes its projection.
 */

export interface LiveStatus {
  state: "running" | "finished";
  epoch: number | null;
  stage: string | null;
  lastCompletedEpoch: number | null;
  epochs: number;
  interval: number;
  layout: string;
  updatedAt: string;
}

export interface LiveRunEntry {
  id: string;
  dir: string;
  runDir: string;
  label: string;
  runLabel: string;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  seed: number | null;
  fold: number | null;
  labelBudget: string | null;
  epochCount: number;
  modifiedAt: number;
  status: LiveStatus | null;
  running: boolean;
}

export interface LiveEpochEntry {
  epoch: number;
  slug: string;
  fingerprint: string;
  has2d: boolean;
  has3d: boolean;
  hasGraph: boolean;
  hasEvidence: boolean;
  hasPropagation: boolean;
}

export interface LiveManifest {
  dir: string;
  runDir: string;
  baseFingerprint: string | null;
  epochs: LiveEpochEntry[];
  epochMetrics: Record<string, number | null>[];
  status: LiveStatus | null;
  checkedAt: string;
}

export interface LiveBase {
  version: number;
  id: string;
  baseKey: string;
  source: "epoch_tracking";
  sourcePath: string;
  projection: string;
  referenceEpoch: number;
  nodeEpoch: number | null;
  label: string;
  runLabel: string;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  seed: number | null;
  fold: number | null;
  labelBudget: string | null;
  runDir: string;
  subsets: string[];
  classNames: string[];
  points: { subset: number[]; class: number[]; sample: number[] };
}

export interface LiveFrame {
  epoch: number;
  fingerprint?: string;
  /** Set while the epoch's 2D layout is not complete yet; nothing else is. */
  pending?: string;
  /** Base64 little-endian float32, n x 2 (and n x 3), unaligned. */
  layout: string;
  layout3d: string | null;
  nodeMetrics: Record<string, (number | null)[]> | null;
  edges: EdgeListFile | null;
  evidence: EvidenceFile | null;
  propagation: PropagationFile | null;
  pseudoLabels: PseudoLabelFile | null;
  clusters: number[] | null;
  clusterK: number | null;
  clusterRefreshEpoch: number | null;
  centerSimilarity: string | null;
  minedPairs: MinedPairsFile | null;
  /** Parts that exist but could not be read yet (still being written). */
  incomplete?: string[];
}

export interface LiveFramesResponse {
  version: number;
  baseKey: string;
  extraLabels: number[];
  extraClassNames: string[];
  frames: LiveFrame[];
}

const STAGE_LABELS: Record<string, string> = {
  waiting: "training",
  embedding: "embedding the pool",
  projection: "fitting the 2D projection",
  projection_3d: "fitting the 3D projection",
  graph: "building the graph",
  class_evidence: "computing class evidence",
};

export function describeStage(status: LiveStatus | null): string | null {
  if (!status) return null;
  if (status.state === "finished") return "finished";
  const stage = status.stage ? STAGE_LABELS[status.stage] ?? status.stage : null;
  if (status.epoch != null && stage) return `epoch ${status.epoch}: ${stage}`;
  if (status.lastCompletedEpoch != null) return `training (last tracked epoch ${status.lastCompletedEpoch})`;
  return stage;
}

function decodeFloat32(text: string): Float32Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Base + frames → the same view file the exporter writes, aligned across epochs. */
export function assembleLiveView(
  base: LiveBase,
  frames: LiveFrame[],
  epochMetrics: Record<string, number | null>[],
  extraClassNames: string[],
  dir: string,
): EmbeddingViewFile | null {
  const ready = frames.filter((frame) => !frame.pending && frame.layout).sort((left, right) => left.epoch - right.epoch);
  if (ready.length === 0) return null;
  const epochs = ready.map((frame) => frame.epoch);
  const aligned2 = alignSeries(ready.map((frame) => decodeFloat32(frame.layout)), 2);
  const with3d = ready.filter((frame) => frame.layout3d);
  const aligned3 = with3d.length ? alignSeries(with3d.map((frame) => decodeFloat32(frame.layout3d!)), 3) : [];

  const byEpoch = <T>(pick: (frame: LiveFrame) => T | null | undefined): Record<string, T> | null => {
    const entries = ready.map((frame) => [String(frame.epoch), pick(frame)] as const).filter((entry) => entry[1] != null);
    return entries.length ? (Object.fromEntries(entries) as Record<string, T>) : null;
  };
  // The typed arrays stand in for number[]: prepareEmbedding only reads length and copies them.
  const layouts = Object.fromEntries(epochs.map((epoch, index) => [String(epoch), aligned2[index] as unknown as number[]]));
  const layouts3d = aligned3.length
    ? Object.fromEntries(with3d.map((frame, index) => [String(frame.epoch), aligned3[index] as unknown as number[]]))
    : null;
  const clusterK = [...ready].reverse().find((frame) => frame.clusterK != null)?.clusterK ?? null;
  const latest = ready[ready.length - 1];

  return {
    version: base.version,
    id: `live:${dir}`,
    label: base.label,
    runLabel: base.runLabel,
    source: "epoch_tracking",
    projection: base.projection,
    dataset: base.dataset,
    method: base.method,
    loss: base.loss,
    seed: base.seed,
    fold: base.fold,
    labelBudget: base.labelBudget,
    runDir: base.runDir,
    sourcePath: base.sourcePath,
    selectedEpoch: null,
    subsets: base.subsets,
    classNames: [...base.classNames, ...extraClassNames],
    points: base.points,
    layouts: {
      "2d": layouts[String(epochs[0])],
      ...(layouts3d ? { "3d": layouts3d[String(with3d[0].epoch)] } : {}),
    },
    tracking: {
      epochs,
      layouts,
      layouts3d,
      nodeMetrics: byEpoch((frame) => frame.nodeMetrics) ?? {},
      epochMetrics,
      clusters: byEpoch((frame) => frame.clusters),
      clusterK,
      clusterRefreshEpochs: byEpoch((frame) => frame.clusterRefreshEpoch),
      edges: byEpoch((frame) => frame.edges),
      evidence: byEpoch((frame) => frame.evidence),
      propagation: byEpoch((frame) => frame.propagation),
      pseudoLabels: byEpoch((frame) => frame.pseudoLabels),
      centerSimilarity: byEpoch((frame) => frame.centerSimilarity),
      minedPairs: byEpoch((frame) => frame.minedPairs),
    },
    notes: [
      `Live: read straight from ${dir} while the run trains. A new epoch appears as soon as its 2D projection is written; its graph, evidence and 3D projection follow when the tracker has them.`,
      `Each epoch's ${base.projection} fit is aligned to the previous one (rotation, reflection, scale) and all share one scale, recomputed as epochs arrive.`,
      latest.edges
        ? "Edges: every edge of the run's own graph at that epoch (graph_edges.npz)."
        : "Edges appear once the tracker has written the epoch's graph.",
    ],
  };
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: "no-store", signal });
  const body = (await response.json().catch(() => null)) as (T & { error?: string }) | null;
  if (!response.ok || body == null) throw new Error(body?.error || `${url} answered ${response.status}`);
  return body;
}

/**
 * Tracking directories the dev server can see; `available` is false on the static site.
 * `enabled = false` (the hosted build) skips the probe entirely.
 */
export function useLiveRuns(intervalMs = 10_000, enabled = true) {
  const [runs, setRuns] = useState<LiveRunEntry[]>([]);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setAvailable(false);
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const result = await fetchJson<{ runs: LiveRunEntry[] }>(`/api/live-tracking/runs/${refreshToken ? "?refresh=1" : ""}`);
        if (cancelled) return;
        setRuns(result.runs);
        setAvailable(true);
      } catch {
        if (cancelled) return;
        // The static export has no route handlers: live runs simply are not offered.
        setAvailable((current) => current ?? false);
      }
      if (!cancelled) timer = window.setTimeout(load, document.visibilityState === "visible" ? intervalMs : intervalMs * 3);
    };
    void load();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [intervalMs, refreshToken, enabled]);

  const refresh = useCallback(() => setRefreshToken((token) => token + 1), []);
  return { runs, available, refresh };
}

export interface LiveViewState {
  data: PreparedEmbedding | null;
  manifest: LiveManifest | null;
  /** Epochs with a layout on disk vs. epochs loaded into the view. */
  progress: { loaded: number; available: number; fetching: boolean };
  error: string | null;
  lastUpdate: number | null;
}

const EMPTY_STATE: LiveViewState = {
  data: null,
  manifest: null,
  progress: { loaded: 0, available: 0, fetching: false },
  error: null,
  lastUpdate: null,
};
// Frames per request while a long run is opened; each batch is drawn as it lands.
const FRAME_BATCH = 8;

/**
 * One tracking directory, kept current: polls the manifest and fetches the
 * frames whose files changed. `paused` stops polling without dropping the view.
 */
export function useLiveTrackingView(dir: string | null, { intervalMs = 2_000, paused = false } = {}): LiveViewState {
  const [state, setState] = useState<LiveViewState>(EMPTY_STATE);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  useEffect(() => {
    setState(EMPTY_STATE);
    if (!dir) return;
    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const query = `dir=${encodeURIComponent(dir)}`;
    let base: LiveBase | null = null;
    let baseFingerprint: string | null = null;
    let extras: number[] = [];
    let extraNames: string[] = [];
    // epoch -> the frame and the fingerprint it was fetched at (pending ones too,
    // so an epoch stuck half-written is not asked for again until it changes).
    const frames = new Map<number, LiveFrame>();
    let lastMetrics = "";

    const publish = (manifest: LiveManifest, fetching: boolean) => {
      const available = manifest.epochs.filter((entry) => entry.has2d).length;
      const loadedFrames = [...frames.values()].filter((frame) => !frame.pending);
      let data: PreparedEmbedding | null = null;
      let error: string | null = null;
      if (base && loadedFrames.length) {
        try {
          const file = assembleLiveView(base, loadedFrames, manifest.epochMetrics, extraNames, dir);
          data = file ? prepareEmbedding(file) : null;
        } catch (reason) {
          error = reason instanceof Error ? reason.message : String(reason);
        }
      }
      setState((current) => ({
        data: data ?? current.data,
        manifest,
        progress: { loaded: loadedFrames.length, available, fetching },
        error,
        lastUpdate: data ? Date.now() : current.lastUpdate,
      }));
    };

    const tick = async () => {
      try {
        const manifest = await fetchJson<LiveManifest>(`/api/live-tracking/manifest/?${query}`, controller.signal);
        if (cancelled) return;
        let changed = false;
        if (manifest.baseFingerprint !== baseFingerprint) {
          baseFingerprint = manifest.baseFingerprint;
          base = null;
          frames.clear();
          extras = [];
          extraNames = [];
          if (baseFingerprint) {
            const fetched = await fetchJson<LiveBase | { pending: string }>(`/api/live-tracking/base/?${query}`, controller.signal);
            if (cancelled) return;
            if (!("pending" in fetched)) base = fetched;
          }
          changed = true;
        }
        // Rows are only ever appended, so the count and the newest row say whether it changed.
        const metricsKey = `${manifest.epochMetrics.length}|${JSON.stringify(manifest.epochMetrics.at(-1) ?? null)}`;
        if (metricsKey !== lastMetrics) {
          lastMetrics = metricsKey;
          changed = true;
        }
        // Drop frames whose epoch directory disappeared (a fold restarted in place).
        const present = new Set(manifest.epochs.map((entry) => entry.epoch));
        for (const epoch of [...frames.keys()]) if (!present.has(epoch)) frames.delete(epoch);
        const stale = base
          ? manifest.epochs.filter((entry) => entry.has2d && frames.get(entry.epoch)?.fingerprint !== entry.fingerprint).map((entry) => entry.epoch)
          : [];
        if (stale.length) {
          setState((current) => ({ ...current, manifest, progress: { ...current.progress, fetching: true } }));
        }
        for (let start = 0; start < stale.length && base && !cancelled; start += FRAME_BATCH) {
          const batch = stale.slice(start, start + FRAME_BATCH);
          const response = await fetchJson<LiveFramesResponse>(
            `/api/live-tracking/frames/?${query}&epochs=${batch.join(",")}&extra=${extras.join(",")}`,
            controller.signal,
          );
          if (cancelled) return;
          if (response.baseKey && response.baseKey !== base.baseKey) {
            // The base changed under us; start over on the next tick.
            baseFingerprint = null;
            break;
          }
          extras = response.extraLabels;
          extraNames = response.extraClassNames;
          response.frames.forEach((frame) => frames.set(frame.epoch, frame));
          changed = true;
          publish(manifest, start + FRAME_BATCH < stale.length);
        }
        if (changed && !stale.length) publish(manifest, false);
        else if (!changed) setState((current) => ({ ...current, manifest, error: null }));
      } catch (reason) {
        if (cancelled || controller.signal.aborted) return;
        setState((current) => ({ ...current, error: reason instanceof Error ? reason.message : String(reason) }));
      }
      if (!cancelled) {
        const hidden = typeof document !== "undefined" && document.visibilityState !== "visible";
        timer = window.setTimeout(() => void loop(), pausedRef.current ? 1_000 : hidden ? intervalMs * 5 : intervalMs);
      }
    };
    const loop = async () => {
      if (pausedRef.current) {
        timer = window.setTimeout(() => void loop(), 1_000);
        return;
      }
      await tick();
    };
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [dir, intervalMs]);

  return state;
}
