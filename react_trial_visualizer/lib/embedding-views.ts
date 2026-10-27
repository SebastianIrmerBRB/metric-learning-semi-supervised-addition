/**
 * Types and pure helpers for the embedding viewer.
 *
 * Views are written by `evaluate/export_embedding_views.py` into
 * `public/embedding-views/`: one `index.json` listing them, and one JSON file
 * per view holding its points. A view from a saved head carries the train split
 * (labeled / unlabeled), validation and D_test in 2D and 3D; a view from a
 * `test_tsne.csv` carries D_test in 2D only. A `test_tsne.csv` can also be
 * dropped straight into the page, which parses it here.
 *
 * Coordinates arrive rescaled into [-1, 1]; only relative positions carry
 * meaning in a t-SNE or PaCMAP projection.
 */

import { parseCsvRecords } from "@/lib/test-results";

export type ViewSource = "head_state" | "epoch_tracking" | "projection_csv";

export interface EmbeddingViewIndexEntry {
  id: string;
  file: string;
  /** Hosted build only: the file is stored as `<file>.0` … `<file>.<parts-1>` (per-file size limits). */
  parts?: number;
  label: string;
  /** The label without the fold, shared by every fold of one run. */
  runLabel?: string;
  epochs?: number[] | null;
  hasGraph?: boolean;
  source: ViewSource;
  projection: string;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  seed: number | null;
  fold: number | null;
  /** Set on a projection of every fold's test embeddings joined: how many folds. */
  concatenatedFolds?: number | null;
  labelBudget: string | null;
  runDir: string | null;
  /** Artifact used to create the view (head, tracking directory, or projection CSV). */
  sourcePath?: string | null;
  pointCount: number;
  classCount: number;
  subsetCounts: Record<string, number>;
  has3d: boolean;
  exportedAt: string;
}

export interface EmbeddingViewIndex {
  version: number;
  generatedAt: string;
  views: EmbeddingViewIndexEntry[];
}

export interface GraphMetricRow {
  name: string;
  label: string;
  higherIsBetter: boolean | null;
  description: string;
  trained: number | null;
  pretrained: number | null;
}

export interface EmbeddingViewFile {
  version: number;
  id: string;
  label: string;
  runLabel?: string;
  source: ViewSource;
  projection: string;
  dataset?: string | null;
  method?: string | null;
  loss?: string | null;
  seed?: number | null;
  fold?: number | null;
  labelBudget?: string | null;
  /** The run the points came from; the image server rebuilds its datasets to find each sample's file. */
  runDir?: string | null;
  sourcePath?: string | null;
  selectedEpoch?: number | null;
  subsets: string[];
  classNames: string[];
  points: { subset: number[]; class: number[]; sample: number[] };
  layouts: { "2d": number[]; "3d"?: number[] };
  notes: string[];
  /** Saved-head views: the kNN graph over train points, nearest labeled links and clusters. */
  graph?: {
    k: number;
    knn: number[];
    nearestLabeled: number[];
    nearestLabeledSimilarity: number[];
    cluster: number[] | null;
    clusterK: number | null;
    clusterSource: string | null;
    metrics: GraphMetricRow[];
  };
  /** Epoch-tracking views: one aligned 2D layout and one set of node metrics per tracked epoch. */
  tracking?: {
    epochs: number[];
    layouts: Record<string, number[]>;
    layouts3d?: Record<string, number[]> | null;
    nodeMetrics: Record<string, Record<string, (number | null)[]>>;
    epochMetrics: Record<string, number | null>[];
    /** SLADE's saved clusters: the refresh each tracked epoch was trained against. */
    clusters?: Record<string, number[]> | null;
    clusterK?: number | null;
    clusterRefreshEpochs?: Record<string, number> | null;
    /** Every edge of the run's graph per epoch (graph_edges.npz), where the run saved it. */
    edges?: Record<string, EdgeListFile> | null;
    /** How label propagation and Hoffer's entropy term read each epoch (class_evidence.npz). */
    evidence?: Record<string, EvidenceFile> | null;
    /** The same diffusion step by step (label_propagation.npz), where the run saved it. */
    propagation?: Record<string, PropagationFile> | null;
    /** The run's own pseudo labels: the generation each epoch was trained on. */
    pseudoLabels?: Record<string, PseudoLabelFile> | null;
    /** SLADE: each point's cosine to its cluster's center at the refresh the epoch trained on (byte, 0 none). */
    centerSimilarity?: Record<string, string> | null;
    /** SLADE: a sample of the pairs Eq 7 mined in that epoch. */
    minedPairs?: Record<string, MinedPairsFile> | null;
  };
}

/** One epoch's graph as the exporter packs it: base64 little-endian index pairs. */
export interface EdgeListFile {
  edgeCount: number;
  /** Each undirected edge is listed once. */
  symmetric: boolean;
  indexBytes: 2 | 4;
  pairs: string;
  /** Byte-quantized weights (times weightScale / 255), absent when all edges weigh `weight`. */
  weights?: string;
  weightScale?: number;
  weight?: number | null;
  /** Each row's own neighbor search in rank order: k indices per row (all ones: empty). */
  k?: number;
  knn?: string;
  /** Their cosine similarity, one byte over [-1, 1]. */
  knnSimilarity?: string;
}

export interface EvidenceMapFile {
  topK: number;
  /** topK view class indices per row (-1 empty), classBytes each. */
  topClass: string;
  /** uint16 over [0, 1]. */
  topProbability: string;
  certainty: string;
  /** Label propagation only: Iscen's weight omega, uint16 over [0, 1]. */
  weight?: string;
}

export interface EvidenceFile {
  classBytes: 2 | 4;
  lp?: EvidenceMapFile;
  hoffer?: EvidenceMapFile;
  /** The network's own prediction: softmax over labeled class prototypes. */
  prototype?: EvidenceMapFile;
  hofferDistanceScale?: number;
  prototypeTemperature?: number;
}

/** One epoch's label propagation, step by step, as the exporter packs it. */
export interface PropagationFile {
  /** Recorded steps, counting step 0 (the seeds alone). */
  steps: number;
  /** Which graph the mass moved over: label propagation's own, or the run's tracked graph. */
  graphSource: string;
  /** null where a backfill ran over a saved graph that fixes them. */
  alpha: number | null;
  gamma: number | null;
  nNeighbors: number;
  classBytes: 2 | 4;
  indexBytes: 2 | 4;
  /** Per row, per step: the class held then (-1 while the row holds nothing). */
  stepClass: string;
  /** Per row, per step: the mass held then over the row's own final mass, one byte. */
  stepShare: string;
  /** Per row: the step its first mass arrived (255 never). */
  arrivalStep: string;
  /** Per row: the edge that delivered it, and that edge's share of the arriving mass. */
  parent: string;
  parentShare: string;
  /** Per row: the labeled node at the root of the parent chain, and its distance in edges. */
  source: string;
  hop: string;
  /** Per row: the strongest contributors to the row's final class, influenceK of them. */
  influenceK?: number;
  influence?: string;
  influenceShare?: string;
  /** Per step: accuracy on the unlabeled rows reached so far, and the share of them reached. */
  stepAccuracy: (number | null)[];
  stepReached: (number | null)[];
  /** The epoch's own numbers, as the run (or the backfill) computed them. */
  summary: Record<string, number | null>;
}

export interface MinedPairsFile {
  sampledPairs: number;
  minedPositivePairs: number;
  minedNegativePairs: number;
  thresholds: (number | null)[];
  indexBytes: 2 | 4;
  pairs: string;
  similarity: string;
  /** Bit 0 mined positive, bit 1 same cluster, bit 2 cluster unknown. */
  kind: string;
}

export interface MinedPairs {
  count: number;
  from: Int32Array;
  to: Int32Array;
  similarity: Float32Array;
  positive: Uint8Array;
  /** 1 same cluster, 0 across clusters, -1 unknown. */
  sameCluster: Int8Array;
  sampledPairs: number;
  minedPositivePairs: number;
  minedNegativePairs: number;
  thresholds: (number | null)[];
}

export interface PseudoLabelFile {
  generation: number;
  generationEpoch: number;
  classBytes: 2 | 4;
  /** View class index per row, -1 where the row was not a candidate. */
  label: string;
  confidence: string;
  accepted: string;
}

export interface EvidenceMap {
  topK: number;
  topClass: Int32Array;
  topProbability: Float32Array;
  /** 1 - normalized entropy of the class distribution. */
  certainty: Float32Array;
  weight: Float32Array | null;
}

export interface EpochEvidence {
  lp: EvidenceMap | null;
  hoffer: EvidenceMap | null;
  prototype: EvidenceMap | null;
  hofferDistanceScale: number | null;
  prototypeTemperature: number | null;
}

/** One epoch's propagation, unpacked: the series per row, then one value per row. */
export interface PropagationFrame {
  steps: number;
  graphSource: string;
  alpha: number | null;
  gamma: number | null;
  nNeighbors: number;
  /** Row-major, `steps` values per row. */
  stepClass: Int32Array;
  stepShare: Float32Array;
  /** -1 where the row was never reached. */
  arrivalStep: Int32Array;
  parent: Int32Array;
  parentShare: Float32Array;
  source: Int32Array;
  hop: Int32Array;
  influenceK: number;
  influence: Int32Array | null;
  influenceShare: Float32Array | null;
  stepAccuracy: (number | null)[];
  stepReached: (number | null)[];
  summary: Record<string, number | null>;
}

export interface PseudoLabelFrame {
  generation: number;
  generationEpoch: number;
  label: Int32Array;
  confidence: Float32Array;
  accepted: Uint8Array;
}

/** Adjacency lists: the neighbors of point i are targets[offsets[i]] .. targets[offsets[i + 1] - 1]. */
export interface NeighborLists {
  offsets: Int32Array;
  targets: Int32Array;
  /** Weight per entry, or null when every edge weighs the same. */
  weights: Float32Array | null;
  edgeCount: number;
  /** Each row's ranked neighbor search: k slots per row (-1 empty), with cosine similarities. */
  k: number;
  knn: Int32Array | null;
  knnSimilarity: Float32Array | null;
}

export interface PreparedGraph {
  k: number;
  knn: Int32Array;
  nearestLabeled: Int32Array;
  nearestLabeledSimilarity: Float32Array;
  cluster: Int32Array | null;
  clusterK: number;
  clusterSource: string | null;
  metrics: GraphMetricRow[];
}

export interface PreparedTracking {
  epochs: number[];
  layouts: Float32Array[];
  /** Per epoch; null for an epoch whose 3D projection is not written yet (a live run). */
  layouts3d: (Float32Array | null)[] | null;
  /** Per epoch: the SLADE refresh's cluster per point (-1 none), or null where none was active yet. */
  clusters: (Int32Array | null)[] | null;
  clusterK: number;
  clusterRefreshEpochs: (number | null)[];
  /** Per epoch, per metric: one value per point (NaN where the run recorded none). */
  nodeMetrics: Record<string, Float32Array>[];
  /** Per epoch: the full graph's adjacency lists, or null for an epoch (or a run) that saved none. */
  neighbors: (NeighborLists | null)[] | null;
  /** Per epoch: label propagation and Hoffer evidence, or null where the run computed none. */
  evidence: (EpochEvidence | null)[] | null;
  /** Largest Hoffer certainty of an unlabeled point over all epochs: the ramp's top. */
  hofferCertaintyMax: number;
  /** Per epoch: the label propagation step by step, or null where none was saved. */
  propagation: (PropagationFrame | null)[] | null;
  /** Per epoch: the pseudo-label generation it trained on, or null before the first. */
  pseudoLabels: (PseudoLabelFrame | null)[] | null;
  /** Per epoch: cosine to the own SLADE cluster center (NaN without a cluster). */
  centerSimilarity: (Float32Array | null)[] | null;
  /** Per epoch: SLADE's sampled mined pairs. */
  minedPairs: (MinedPairs | null)[] | null;
  epochMetrics: Record<string, number | null>[];
}

/** A view unpacked into typed arrays, ready to draw. */
export interface PreparedEmbedding {
  id: string;
  label: string;
  source: EmbeddingViewFile["source"];
  projection: string;
  selectedEpoch: number | null;
  count: number;
  subsets: string[];
  classNames: string[];
  subset: Uint8Array;
  classIndex: Int32Array;
  sample: Int32Array;
  pos2: Float32Array;
  pos3: Float32Array | null;
  subsetCounts: number[];
  /** Per class, per subset: how many points of that class fall in that subset. */
  classSubsetCounts: Uint32Array;
  notes: string[];
  graph: PreparedGraph | null;
  tracking: PreparedTracking | null;
  /** Set for exported views: images are read from the data folder through this run. */
  runDir: string | null;
}

export type MarkerShape = "diamond" | "circle" | "square" | "triangle";

export interface SubsetStyle {
  label: string;
  description: string;
  /** Index into CATEGORICAL, or -1 for the neutral context color. */
  slot: number;
  /** Draw priority: higher ranks land on top. Each rank has one shape (RANK_SHAPES). */
  rank: number;
  shape: MarkerShape;
}

/** The marker of each draw rank, lowest (validation) to highest (labeled). */
export const RANK_SHAPES: MarkerShape[] = ["circle", "square", "triangle", "circle", "diamond"];
export const LABELED_RANK = 4;

/**
 * The dark categorical slots 1-3 of the reference palette. A scatter lets any two
 * marks sit side by side, so only the first three slots are used: they clear the
 * all-pairs CVD and normal-vision floors on this app's surface (validated with
 * `validate_palette.js --pairs all --mode dark --surface #0b1222`). Everything
 * beyond three identities is carried by marker shape or folded into the neutral.
 */
export const CATEGORICAL = ["#3987e5", "#d95926", "#199e70"] as const;

/**
 * All eight dark categorical slots, for the class color mode. Their first three
 * are CATEGORICAL. With hundreds or thousands of classes no palette can make
 * every class identifiable, so class colors only separate *neighboring* classes
 * (see assignClassColors); a class's name comes from hover, pin or search.
 */
export const CLASS_PALETTE = [
  "#3987e5",
  "#d95926",
  "#199e70",
  "#c98500",
  "#d55181",
  "#008300",
  "#9085e9",
  "#e66767",
] as const;

/**
 * Slot pairs that fail the scatter (all-pairs) check side by side on this
 * surface -- CVD separation below 8 or unsimulated separation below 15 -- per
 * `validate_palette.js --pairs all --mode dark --surface #0b1222`. Neighboring
 * classes avoid these as well as identical colors.
 */
const UNSAFE_NEIGHBOR_PAIRS: [number, number][] = [
  [0, 6],
  [1, 3],
  [1, 4],
  [1, 5],
  [1, 7],
  [2, 4],
  [2, 5],
  [2, 7],
  [3, 5],
  [3, 7],
  [4, 7],
];
export const NEUTRAL = "#5d6679";
export const SURFACE = "#0b1222";

export const SUBSET_STYLES: Record<string, SubsetStyle> = {
  labeled: {
    label: "Labeled",
    description: "Train samples whose labels the run trained on",
    slot: 0,
    rank: 4,
    shape: "diamond",
  },
  train: {
    label: "Train",
    description: "Every train sample (a fully supervised run has no split)",
    slot: 0,
    rank: 4,
    shape: "diamond",
  },
  unlabeled: {
    label: "Unlabeled",
    description: "Train samples the SSL method saw without labels",
    slot: 1,
    rank: 3,
    shape: "circle",
  },
  validation: {
    label: "Validation",
    description: "The fold's validation set",
    slot: -1,
    rank: 0,
    shape: "circle",
  },
  test_query: {
    label: "Test queries",
    description: "D_test query side",
    slot: 2,
    rank: 2,
    shape: "triangle",
  },
  test_gallery: {
    label: "Test gallery",
    description: "D_test gallery side",
    slot: 2,
    rank: 1,
    shape: "square",
  },
  test: {
    label: "Test",
    description: "D_test",
    slot: 2,
    rank: 1,
    shape: "square",
  },
};

export function subsetStyle(name: string): SubsetStyle {
  return (
    SUBSET_STYLES[name] ?? {
      label: name,
      description: name,
      slot: -1,
      rank: 0,
      shape: "circle",
    }
  );
}

export function subsetColor(name: string): string {
  const slot = subsetStyle(name).slot;
  return slot < 0 ? NEUTRAL : CATEGORICAL[slot];
}

export const MAX_PINNED_CLASSES = CATEGORICAL.length;

function assertLength(name: string, actual: number, expected: number) {
  if (actual !== expected) {
    throw new Error(`The view's ${name} holds ${actual} values; ${expected} were expected.`);
  }
}

function decodeBase64(text: string): Uint8Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function decodeIntegers(text: string, bytes: number, signed: boolean): Int32Array {
  const raw = decodeBase64(text);
  const reader = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const values = new Int32Array(raw.byteLength / bytes);
  for (let index = 0; index < values.length; index += 1) {
    const offset = index * bytes;
    values[index] =
      bytes === 1
        ? reader.getUint8(offset)
        : bytes === 2
          ? signed
            ? reader.getInt16(offset, true)
            : reader.getUint16(offset, true)
          : reader.getInt32(offset, true);
  }
  return values;
}

function decodeUnit16(text: string): Float32Array {
  return Float32Array.from(decodeIntegers(text, 2, false), (value) => value / 65535);
}

function decodeEvidenceMap(file: EvidenceMapFile, classBytes: number, count: number, name: string): EvidenceMap {
  const map = {
    topK: file.topK,
    topClass: decodeIntegers(file.topClass, classBytes, true),
    topProbability: decodeUnit16(file.topProbability),
    certainty: decodeUnit16(file.certainty),
    weight: file.weight ? decodeUnit16(file.weight) : null,
  };
  assertLength(`${name} classes`, map.topClass.length, count * file.topK);
  assertLength(`${name} certainty`, map.certainty.length, count);
  return map;
}

export function decodeEvidence(file: EvidenceFile, count: number): EpochEvidence {
  return {
    lp: file.lp ? decodeEvidenceMap(file.lp, file.classBytes, count, "label propagation") : null,
    hoffer: file.hoffer ? decodeEvidenceMap(file.hoffer, file.classBytes, count, "Hoffer") : null,
    prototype: file.prototype ? decodeEvidenceMap(file.prototype, file.classBytes, count, "network prediction") : null,
    hofferDistanceScale: file.hofferDistanceScale ?? null,
    prototypeTemperature: file.prototypeTemperature ?? null,
  };
}

function decodeByteUnit(text: string): Float32Array {
  return Float32Array.from(decodeIntegers(text, 1, false), (value) => value / 255);
}

export function decodePropagation(file: PropagationFile, count: number): PropagationFrame {
  // An index the exporter had no row for is all ones, as in the edge lists.
  const missing = file.indexBytes === 2 ? 0xffff : -1;
  const rows = (text: string) => decodeIntegers(text, file.indexBytes, false).map((value) => (value === missing ? -1 : value));
  // 255 is "never reached" for the step columns, which count far below it.
  const steps = (text: string) => decodeIntegers(text, 1, false).map((value) => (value === 255 ? -1 : value));
  const frame: PropagationFrame = {
    steps: file.steps,
    graphSource: file.graphSource,
    alpha: file.alpha,
    gamma: file.gamma,
    nNeighbors: file.nNeighbors,
    stepClass: decodeIntegers(file.stepClass, file.classBytes, true),
    stepShare: decodeByteUnit(file.stepShare),
    arrivalStep: steps(file.arrivalStep),
    parent: rows(file.parent),
    parentShare: decodeByteUnit(file.parentShare),
    source: rows(file.source),
    hop: steps(file.hop),
    influenceK: file.influenceK ?? 0,
    influence: file.influence ? rows(file.influence) : null,
    influenceShare: file.influenceShare ? decodeByteUnit(file.influenceShare) : null,
    stepAccuracy: file.stepAccuracy ?? [],
    stepReached: file.stepReached ?? [],
    summary: file.summary ?? {},
  };
  assertLength("propagation series", frame.stepClass.length, count * file.steps);
  assertLength("propagation arrival steps", frame.arrivalStep.length, count);
  if (frame.influence) assertLength("propagation contributors", frame.influence.length, count * frame.influenceK);
  return frame;
}

export function decodeMinedPairs(file: MinedPairsFile): MinedPairs {
  const flat = decodeIntegers(file.pairs, file.indexBytes, false);
  const count = flat.length / 2;
  const from = new Int32Array(count);
  const to = new Int32Array(count);
  for (let pair = 0; pair < count; pair += 1) {
    from[pair] = flat[2 * pair];
    to[pair] = flat[2 * pair + 1];
  }
  const kind = decodeIntegers(file.kind, 1, false);
  assertLength("mined pair kinds", kind.length, count);
  return {
    count,
    from,
    to,
    similarity: Float32Array.from(decodeIntegers(file.similarity, 1, false), (value) => value / 127.5 - 1),
    positive: Uint8Array.from(kind, (value) => value & 1),
    sameCluster: Int8Array.from(kind, (value) => (value & 4 ? -1 : value & 2 ? 1 : 0)),
    sampledPairs: file.sampledPairs,
    minedPositivePairs: file.minedPositivePairs,
    minedNegativePairs: file.minedNegativePairs,
    thresholds: file.thresholds,
  };
}

function decodeCenterSimilarity(text: string, count: number): Float32Array {
  const values = Float32Array.from(decodeIntegers(text, 1, false), (value) => (value === 0 ? Number.NaN : (value - 1) / 127 - 1));
  assertLength("center similarity", values.length, count);
  return values;
}

export function decodePseudoLabels(file: PseudoLabelFile, count: number): PseudoLabelFrame {
  const frame = {
    generation: file.generation,
    generationEpoch: file.generationEpoch,
    label: decodeIntegers(file.label, file.classBytes, true),
    confidence: decodeUnit16(file.confidence),
    accepted: Uint8Array.from(decodeIntegers(file.accepted, 1, false)),
  };
  assertLength("pseudo labels", frame.label.length, count);
  return frame;
}

export function decodeNeighborLists(file: EdgeListFile, count: number): NeighborLists {
  const bytes = decodeBase64(file.pairs);
  const reader = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const edgeCount = bytes.byteLength / (2 * file.indexBytes);
  assertLength("edge list", edgeCount, file.edgeCount);
  const read = (slot: number) =>
    file.indexBytes === 2 ? reader.getUint16(slot * 2, true) : reader.getUint32(slot * 4, true);
  const quantized = file.weights ? decodeBase64(file.weights) : null;
  if (quantized) assertLength("edge weights", quantized.length, edgeCount);

  const offsets = new Int32Array(count + 1);
  for (let edge = 0; edge < edgeCount; edge += 1) {
    offsets[read(2 * edge) + 1] += 1;
    if (file.symmetric) offsets[read(2 * edge + 1) + 1] += 1;
  }
  for (let row = 0; row < count; row += 1) offsets[row + 1] += offsets[row];
  const fill = offsets.slice(0, count);
  const targets = new Int32Array(offsets[count]);
  const weights = quantized ? new Float32Array(offsets[count]) : null;
  const scale = (file.weightScale ?? 1) / 255;
  const place = (from: number, to: number, edge: number) => {
    const slot = fill[from];
    fill[from] += 1;
    targets[slot] = to;
    if (weights && quantized) weights[slot] = quantized[edge] * scale;
  };
  for (let edge = 0; edge < edgeCount; edge += 1) {
    const from = read(2 * edge);
    const to = read(2 * edge + 1);
    place(from, to, edge);
    if (file.symmetric) place(to, from, edge);
  }
  let knn: Int32Array | null = null;
  let knnSimilarity: Float32Array | null = null;
  const k = file.k ?? 0;
  if (file.knn && k > 0) {
    const empty = file.indexBytes === 2 ? 0xffff : -1;
    knn = decodeIntegers(file.knn, file.indexBytes, false).map((value) => (value === empty ? -1 : value));
    assertLength("ranked neighbors", knn.length, count * k);
    if (file.knnSimilarity) {
      knnSimilarity = Float32Array.from(decodeIntegers(file.knnSimilarity, 1, false), (value) => value / 127.5 - 1);
    }
  }
  return { offsets, targets, weights, edgeCount, k, knn, knnSimilarity };
}

/** The top of the Hoffer certainty ramp: its largest unlabeled value over every epoch. */
function hofferCertaintyMax(view: PreparedEmbedding): number {
  const unlabeledSubset = view.subsets.indexOf("unlabeled");
  let largest = 0;
  view.tracking?.evidence?.forEach((evidence) => {
    const certainty = evidence?.hoffer?.certainty;
    if (!certainty) return;
    for (let row = 0; row < view.count; row += 1) {
      if (view.subset[row] === unlabeledSubset && certainty[row] > largest) largest = certainty[row];
    }
  });
  return largest;
}

export function prepareEmbedding(file: EmbeddingViewFile): PreparedEmbedding {
  const prepared = prepareEmbeddingArrays(file);
  if (prepared.tracking) prepared.tracking.hofferCertaintyMax = hofferCertaintyMax(prepared);
  return prepared;
}

function prepareEmbeddingArrays(file: EmbeddingViewFile): PreparedEmbedding {
  const count = file.points.subset.length;
  assertLength("class column", file.points.class.length, count);
  assertLength("sample column", file.points.sample.length, count);
  assertLength("2D layout", file.layouts["2d"].length, count * 2);
  if (file.layouts["3d"]) assertLength("3D layout", file.layouts["3d"].length, count * 3);

  const subset = Uint8Array.from(file.points.subset);
  const classIndex = Int32Array.from(file.points.class);
  const subsetCounts = new Array<number>(file.subsets.length).fill(0);
  const classSubsetCounts = new Uint32Array(file.classNames.length * file.subsets.length);
  for (let index = 0; index < count; index += 1) {
    subsetCounts[subset[index]] += 1;
    classSubsetCounts[classIndex[index] * file.subsets.length + subset[index]] += 1;
  }

  return {
    id: file.id,
    label: file.label,
    source: file.source,
    projection: file.projection,
    selectedEpoch: file.selectedEpoch ?? null,
    count,
    subsets: file.subsets,
    classNames: file.classNames,
    subset,
    classIndex,
    sample: Int32Array.from(file.points.sample),
    pos2: Float32Array.from(file.layouts["2d"]),
    pos3: file.layouts["3d"] ? Float32Array.from(file.layouts["3d"]) : null,
    subsetCounts,
    classSubsetCounts,
    notes: file.notes ?? [],
    graph: file.graph
      ? {
          k: file.graph.k,
          knn: Int32Array.from(file.graph.knn),
          nearestLabeled: Int32Array.from(file.graph.nearestLabeled),
          nearestLabeledSimilarity: Float32Array.from(file.graph.nearestLabeledSimilarity),
          cluster: file.graph.cluster ? Int32Array.from(file.graph.cluster) : null,
          clusterK: file.graph.clusterK ?? 0,
          clusterSource: file.graph.clusterSource,
          metrics: file.graph.metrics ?? [],
        }
      : null,
    tracking: file.tracking
      ? {
          epochs: file.tracking.epochs,
          layouts: file.tracking.epochs.map((epoch) => {
            const layout = file.tracking?.layouts[String(epoch)] ?? file.layouts["2d"];
            assertLength(`epoch ${epoch} layout`, layout.length, count * 2);
            return Float32Array.from(layout);
          }),
          nodeMetrics: file.tracking.epochs.map((epoch) => {
            const metrics = file.tracking?.nodeMetrics[String(epoch)] ?? {};
            return Object.fromEntries(
              Object.entries(metrics).map(([name, values]) => [
                name,
                Float32Array.from(values, (value) => (value == null ? Number.NaN : value)),
              ]),
            );
          }),
          epochMetrics: file.tracking.epochMetrics ?? [],
          neighbors: file.tracking.edges
            ? file.tracking.epochs.map((epoch) => {
                const edges = file.tracking?.edges?.[String(epoch)];
                return edges ? decodeNeighborLists(edges, count) : null;
              })
            : null,
          evidence: file.tracking.evidence
            ? file.tracking.epochs.map((epoch) => {
                const evidence = file.tracking?.evidence?.[String(epoch)];
                return evidence ? decodeEvidence(evidence, count) : null;
              })
            : null,
          hofferCertaintyMax: 0,
          centerSimilarity: file.tracking.centerSimilarity
            ? file.tracking.epochs.map((epoch) => {
                const text = file.tracking?.centerSimilarity?.[String(epoch)];
                return text ? decodeCenterSimilarity(text, count) : null;
              })
            : null,
          minedPairs: file.tracking.minedPairs
            ? file.tracking.epochs.map((epoch) => {
                const pairs = file.tracking?.minedPairs?.[String(epoch)];
                return pairs ? decodeMinedPairs(pairs) : null;
              })
            : null,
          propagation: file.tracking.propagation
            ? file.tracking.epochs.map((epoch) => {
                const frame = file.tracking?.propagation?.[String(epoch)];
                return frame ? decodePropagation(frame, count) : null;
              })
            : null,
          pseudoLabels: file.tracking.pseudoLabels
            ? file.tracking.epochs.map((epoch) => {
                const frame = file.tracking?.pseudoLabels?.[String(epoch)];
                return frame ? decodePseudoLabels(frame, count) : null;
              })
            : null,
          layouts3d: file.tracking.layouts3d
            ? file.tracking.epochs.map((epoch) => {
                const layout = file.tracking?.layouts3d?.[String(epoch)];
                if (!layout) return null;
                assertLength(`epoch ${epoch} 3D layout`, layout.length, count * 3);
                return Float32Array.from(layout);
              })
            : null,
          clusters: file.tracking.clusters
            ? file.tracking.epochs.map((epoch) => {
                const clusters = file.tracking?.clusters?.[String(epoch)];
                return clusters ? Int32Array.from(clusters) : null;
              })
            : null,
          clusterK: file.tracking.clusterK ?? 0,
          clusterRefreshEpochs: file.tracking.epochs.map(
            (epoch) => file.tracking?.clusterRefreshEpochs?.[String(epoch)] ?? null,
          ),
        }
      : null,
    runDir: file.runDir ?? null,
  };
}

/**
 * The URL of a point's image, or null for a view without a run (an upload). The
 * app looks the file up in the data folder: `sample` indexes the run's train,
 * validation or test dataset, whichever the point's subset belongs to.
 */
export function imageUrl(view: PreparedEmbedding, row: number): string | null {
  if (!view.runDir || row < 0 || row >= view.count) return null;
  const subset = view.subsets[view.subset[row]];
  const set = subset === "validation" ? "validation" : subset.startsWith("test") ? "test" : "train";
  const query = new URLSearchParams({ run: view.runDir, set, sample: String(view.sample[row]) });
  return `/api/embedding-image/?${query.toString()}`;
}

function normalizeLayout(values: number[], dimensions: number): number[] {
  const low = new Array<number>(dimensions).fill(Infinity);
  const high = new Array<number>(dimensions).fill(-Infinity);
  for (let index = 0; index < values.length; index += 1) {
    const axis = index % dimensions;
    low[axis] = Math.min(low[axis], values[index]);
    high[axis] = Math.max(high[axis], values[index]);
  }
  const halfRange = Math.max(...high.map((value, axis) => value - low[axis])) / 2 || 1;
  return values.map((value, index) => {
    const axis = index % dimensions;
    return (value - (low[axis] + high[axis]) / 2) / halfRange;
  });
}

/** Parse a `test_tsne.csv` (or any `<projection>_x/_y` CSV a run writes). */
export function parseProjectionCsv(text: string, name: string): EmbeddingViewFile {
  const { columns, rows } = parseCsvRecords(text);
  const xColumn = columns.find((column) => column.endsWith("_x") && columns.includes(`${column.slice(0, -2)}_y`));
  if (!xColumn) {
    throw new Error("No <projection>_x / <projection>_y columns were found in this CSV.");
  }
  const prefix = xColumn.slice(0, -2);
  const zColumn = columns.includes(`${prefix}_z`) ? `${prefix}_z` : null;
  const subsetColumn = columns.includes("subset") ? "subset" : null;

  const classLookup = new Map<string, number>();
  const subsetNames: string[] = [];
  const subsetLookup = new Map<string, number>();
  const classes: number[] = [];
  const subsets: number[] = [];
  const samples: number[] = [];
  const coordinates2: number[] = [];
  const coordinates3: number[] = [];

  rows.forEach((row, index) => {
    const x = Number(row[xColumn]);
    const y = Number(row[`${prefix}_y`]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const className = row.class_name || `class ${row.label ?? "?"}`;
    if (!classLookup.has(className)) classLookup.set(className, classLookup.size);
    const subsetName = (subsetColumn && row[subsetColumn]) || "test";
    if (!subsetLookup.has(subsetName)) {
      subsetLookup.set(subsetName, subsetNames.length);
      subsetNames.push(subsetName);
    }
    classes.push(classLookup.get(className) as number);
    subsets.push(subsetLookup.get(subsetName) as number);
    const sample = row.sample_position == null ? Number.NaN : Number(row.sample_position);
    samples.push(Number.isFinite(sample) ? sample : index);
    coordinates2.push(x, y);
    if (zColumn) coordinates3.push(x, y, Number(row[zColumn]));
  });
  if (classes.length === 0) throw new Error("The CSV holds no finite coordinates.");

  const layouts: EmbeddingViewFile["layouts"] = { "2d": normalizeLayout(coordinates2, 2) };
  if (zColumn && coordinates3.every(Number.isFinite)) layouts["3d"] = normalizeLayout(coordinates3, 3);

  return {
    version: 1,
    id: `upload:${name}:${classes.length}`,
    label: name,
    source: "projection_csv",
    projection: prefix,
    subsets: subsetNames,
    classNames: [...classLookup.keys()],
    points: { subset: subsets, class: classes, sample: samples },
    layouts,
    notes: [
      subsetColumn
        ? `Uploaded ${name}.`
        : `Uploaded ${name}: D_test only, so there is no labeled/unlabeled split to toggle.`,
      zColumn ? "" : "2D only: this projection was fitted with two components.",
    ].filter(Boolean),
  };
}

/**
 * Give every class one of CLASS_PALETTE's eight colors so that classes whose
 * points sit next to each other in the 2D layout never share a color, and avoid
 * the slot pairs that are hard to tell apart side by side.
 *
 * Adjacency comes from a grid over the layout: two classes are neighbors when
 * their points fall in the same or adjacent cells, weighted by how many points
 * meet there. Classes are colored greedily, most-connected first, each taking the
 * slot with the least weighted clash against its already-colored neighbors
 * (same slot costs 1, an unsafe pair 0.5); ties go to the least-used slot. The 2D
 * layout drives both 2D and 3D so a class keeps its color when switching.
 */
export function assignClassColors(view: PreparedEmbedding): Uint8Array {
  return assignGroupColors(view.classIndex, view.classNames.length, view.pos2, view.count);
}

/** assignClassColors for any grouping (classes, clusters); rows with group -1 are skipped. */
export function assignGroupColors(
  groupIndex: Int32Array,
  groupCount: number,
  pos2: Float32Array,
  pointCount: number,
): Uint8Array {
  const classCount = groupCount;
  const slots = new Uint8Array(classCount);
  const paletteSize = CLASS_PALETTE.length;
  if (classCount <= 1) return slots;

  const unsafe = new Uint8Array(paletteSize * paletteSize);
  for (const [left, right] of UNSAFE_NEIGHBOR_PAIRS) {
    unsafe[left * paletteSize + right] = 1;
    unsafe[right * paletteSize + left] = 1;
  }

  const grid = Math.min(320, Math.max(24, Math.round(Math.sqrt(pointCount) / 1.5)));
  const cells = new Map<number, Map<number, number>>();
  for (let index = 0; index < pointCount; index += 1) {
    if (groupIndex[index] < 0) continue;
    const column = Math.min(grid - 1, Math.max(0, Math.floor(((pos2[index * 2] + 1) / 2) * grid)));
    const row = Math.min(grid - 1, Math.max(0, Math.floor(((pos2[index * 2 + 1] + 1) / 2) * grid)));
    const cell = row * grid + column;
    let counts = cells.get(cell);
    if (!counts) {
      counts = new Map();
      cells.set(cell, counts);
    }
    const classIndex = groupIndex[index];
    counts.set(classIndex, (counts.get(classIndex) ?? 0) + 1);
  }

  const weights = new Map<number, number>();
  const addWeight = (left: number, right: number, weight: number) => {
    if (left === right) return;
    const key = left < right ? left * classCount + right : right * classCount + left;
    weights.set(key, (weights.get(key) ?? 0) + weight);
  };
  // Self plus four forward neighbors visits every adjacent cell pair once.
  const forward: [number, number, number][] = [
    [0, 0, 1],
    [1, 0, 0.5],
    [-1, 1, 0.5],
    [0, 1, 0.5],
    [1, 1, 0.5],
  ];
  for (const [cell, counts] of cells) {
    const row = Math.floor(cell / grid);
    const column = cell % grid;
    for (const [dx, dy, scale] of forward) {
      const otherColumn = column + dx;
      const otherRow = row + dy;
      if (otherColumn < 0 || otherColumn >= grid || otherRow >= grid) continue;
      const other = dx === 0 && dy === 0 ? counts : cells.get(otherRow * grid + otherColumn);
      if (!other) continue;
      for (const [left, leftCount] of counts) {
        for (const [right, rightCount] of other) {
          if (other === counts && right <= left) continue;
          addWeight(left, right, Math.min(leftCount, rightCount) * scale);
        }
      }
    }
  }

  const neighbors: [number, number][][] = Array.from({ length: classCount }, () => []);
  const strength = new Float64Array(classCount);
  for (const [key, weight] of weights) {
    const left = Math.floor(key / classCount);
    const right = key % classCount;
    neighbors[left].push([right, weight]);
    neighbors[right].push([left, weight]);
    strength[left] += weight;
    strength[right] += weight;
  }

  const order = Array.from({ length: classCount }, (_, classIndex) => classIndex).sort(
    (left, right) => strength[right] - strength[left] || left - right,
  );
  const colored = new Uint8Array(classCount);
  const usage = new Float64Array(paletteSize);
  const penalty = new Float64Array(paletteSize);
  for (const classIndex of order) {
    penalty.fill(0);
    for (const [neighbor, weight] of neighbors[classIndex]) {
      if (!colored[neighbor]) continue;
      const taken = slots[neighbor];
      for (let slot = 0; slot < paletteSize; slot += 1) {
        if (slot === taken) penalty[slot] += weight;
        else if (unsafe[slot * paletteSize + taken]) penalty[slot] += weight * 0.5;
      }
    }
    let best = 0;
    for (let slot = 1; slot < paletteSize; slot += 1) {
      if (
        penalty[slot] < penalty[best] - 1e-9 ||
        (Math.abs(penalty[slot] - penalty[best]) <= 1e-9 && usage[slot] < usage[best])
      ) {
        best = slot;
      }
    }
    slots[classIndex] = best;
    colored[classIndex] = 1;
    usage[best] += 1;
  }
  return slots;
}

export type ColorMode =
  | "subset"
  | "class"
  | "pinned"
  | "agreement"
  | "purity"
  | "cluster"
  | "purityChange"
  | "agreementChange"
  | "lpPrediction"
  | "lpWeight"
  | "hofferPrediction"
  | "hofferCertainty"
  | "pseudoLabel"
  | "prototypePrediction"
  | "propagationClass"
  | "propagationCorrect"
  | "propagationHop"
  | "outlier"
  | "centerSimilarity";
export type SplitEmphasis = "none" | "labeled" | "unlabeled";

/**
 * A single-hue blue ramp for a magnitude in [0, 1], dark to light. Validated as an
 * ordinal ramp on this surface (`validate_palette.js --ordinal --mode dark
 * --surface #0b1222`): monotone lightness, visible steps, darkest step 2.3:1.
 */
export const RAMP = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"] as const;

/** The dark diverging midpoint of the reference palette: "no change". */
export const CHANGE_NONE = "#383835";

/** Every color the canvas draws, by index: class slots, neutral, the ramp, then "no change". */
export const CANVAS_COLORS: string[] = [...CLASS_PALETTE, NEUTRAL, ...RAMP, CHANGE_NONE];
export const NEUTRAL_INDEX: number = CLASS_PALETTE.length;
export const RAMP_OFFSET = NEUTRAL_INDEX + 1;
export const CHANGE_NONE_INDEX = RAMP_OFFSET + RAMP.length;
/** Change against the reference epoch: better (blue, slot 1) or worse (red, slot 8). */
export const CHANGE_BETTER_INDEX = 0;
export const CHANGE_WORSE_INDEX = 7;
/** Purity must move by more than this to count as a change. */
export const PURITY_CHANGE_THRESHOLD = 0.1;
/** Agreement: same class as the nearest labeled sample, or not. */
export const AGREE_INDEX = 0;
export const DISAGREE_INDEX = 1;

/** Point-wise quantities derived from a view's graph or tracking data at one epoch. */
export interface DerivedGraph {
  /** Share of a point's graph neighbors with its own class; NaN where undefined. */
  purity: Float32Array | null;
  /** 1 same class as its reference neighbor, 0 different, -1 not applicable. */
  agreement: Int8Array | null;
  /** What `agreement` compares against, for the legend. */
  agreementSource: string | null;
  /** Labeled points: how many unlabeled points have them as nearest labeled, and how many of those share the class. */
  pulls: Int32Array | null;
  pullsSameClass: Int32Array | null;
  /** Unlabeled points: 1 label propagation's top class is the true one, 0 not (or no mass); -1 otherwise. */
  lpCorrect: Int8Array | null;
  /** Iscen's weight omega per point. */
  lpWeight: Float32Array | null;
  /** Unlabeled points: 1 Hoffer's most likely reference class is the true one, 0 not; -1 otherwise. */
  hofferCorrect: Int8Array | null;
  hofferCertainty: Float32Array | null;
  /** 2 accepted and right, 1 accepted and wrong, 0 a rejected candidate, -1 not a candidate. */
  pseudo: Int8Array | null;
  /** Unlabeled points: whether the network's prototype prediction is the true class. */
  prototypeCorrect: Int8Array | null;
  /** The class each point holds after `propagationStep` steps of the diffusion (-1 none). */
  propagationClass: Int32Array | null;
  /** Unlabeled points: 1 that class is the true one, 0 not, -1 nothing held (or labeled). */
  propagationCorrect: Int8Array | null;
  /** Edges between a point and the labeled node its mass came from (NaN never reached). */
  propagationHop: Float32Array | null;
  /** The step `propagationClass` was read at, and the frame it came from. */
  propagationStep: number;
  propagation: PropagationFrame | null;
  /** 1 - mean cosine to the point's own k nearest neighbors at this epoch. */
  outlierScore: Float32Array | null;
  /** 1 for the OUTLIER_SHARE of points with the highest outlier score. */
  outlier: Uint8Array | null;
  centerSimilarity: Float32Array | null;
}

/** The share of points the outlier mode marks. */
export const OUTLIER_SHARE = 0.05;

const NO_EVIDENCE = {
  propagationClass: null,
  propagationCorrect: null,
  propagationHop: null,
  propagationStep: 0,
  propagation: null,
  lpCorrect: null,
  lpWeight: null,
  hofferCorrect: null,
  hofferCertainty: null,
  pseudo: null,
  prototypeCorrect: null,
  outlierScore: null,
  outlier: null,
  centerSimilarity: null,
};

function outlierAt(view: PreparedEmbedding, epochIndex: number) {
  const lists = view.tracking?.neighbors?.[epochIndex];
  if (!lists?.knn || !lists.knnSimilarity || lists.k === 0) return { outlierScore: null, outlier: null };
  const score = new Float32Array(view.count).fill(Number.NaN);
  const finite: number[] = [];
  for (let row = 0; row < view.count; row += 1) {
    let sum = 0;
    let valid = 0;
    for (let slot = 0; slot < lists.k; slot += 1) {
      if (lists.knn[row * lists.k + slot] < 0) continue;
      sum += lists.knnSimilarity[row * lists.k + slot];
      valid += 1;
    }
    if (valid) {
      score[row] = 1 - sum / valid;
      finite.push(score[row]);
    }
  }
  finite.sort((a, b) => b - a);
  const cutoff = finite.length ? finite[Math.max(0, Math.ceil(finite.length * OUTLIER_SHARE) - 1)] : Infinity;
  const outlier = Uint8Array.from(score, (value) => (Number.isFinite(value) && value >= cutoff ? 1 : 0));
  return { outlierScore: score, outlier };
}

function topClassCorrect(view: PreparedEmbedding, map: EvidenceMap): Int8Array {
  const unlabeledSubset = view.subsets.indexOf("unlabeled");
  const correct = new Int8Array(view.count).fill(-1);
  for (let row = 0; row < view.count; row += 1) {
    if (view.subset[row] !== unlabeledSubset) continue;
    correct[row] = map.topClass[row * map.topK] === view.classIndex[row] ? 1 : 0;
  }
  return correct;
}

/**
 * One epoch's propagation read at one step: which class each point holds after
 * that many hops, whether it is the true one, and how far the mass travelled.
 * `step` counts from 0 (the seeds alone); -1 asks for the last recorded step.
 */
export function propagationAt(view: PreparedEmbedding, epochIndex: number, step = -1) {
  const frame = view.tracking?.propagation?.[epochIndex] ?? null;
  if (!frame) return { propagationClass: null, propagationCorrect: null, propagationHop: null, propagationStep: 0, propagation: null };
  const unlabeledSubset = view.subsets.indexOf("unlabeled");
  const chosen = step < 0 ? frame.steps - 1 : Math.max(0, Math.min(step, frame.steps - 1));
  const held = new Int32Array(view.count);
  const correct = new Int8Array(view.count).fill(-1);
  const hop = new Float32Array(view.count).fill(Number.NaN);
  for (let row = 0; row < view.count; row += 1) {
    const classIndex = frame.stepClass[row * frame.steps + chosen];
    held[row] = classIndex;
    if (classIndex >= 0 && view.subset[row] === unlabeledSubset) correct[row] = classIndex === view.classIndex[row] ? 1 : 0;
    if (frame.hop[row] >= 0) hop[row] = frame.hop[row];
  }
  return { propagationClass: held, propagationCorrect: correct, propagationHop: hop, propagationStep: chosen, propagation: frame };
}

function evidenceAt(view: PreparedEmbedding, epochIndex: number) {
  const evidence = view.tracking?.evidence?.[epochIndex] ?? null;
  const frame = view.tracking?.pseudoLabels?.[epochIndex] ?? null;
  let pseudo: Int8Array | null = null;
  if (frame) {
    pseudo = new Int8Array(view.count).fill(-1);
    for (let row = 0; row < view.count; row += 1) {
      if (frame.label[row] < 0) continue;
      pseudo[row] = frame.accepted[row] ? (frame.label[row] === view.classIndex[row] ? 2 : 1) : 0;
    }
  }
  return {
    lpCorrect: evidence?.lp ? topClassCorrect(view, evidence.lp) : null,
    lpWeight: evidence?.lp?.weight ?? null,
    hofferCorrect: evidence?.hoffer ? topClassCorrect(view, evidence.hoffer) : null,
    hofferCertainty: evidence?.hoffer?.certainty ?? null,
    pseudo,
    prototypeCorrect: evidence?.prototype ? topClassCorrect(view, evidence.prototype) : null,
    ...outlierAt(view, epochIndex),
    centerSimilarity: view.tracking?.centerSimilarity?.[epochIndex] ?? null,
  };
}

export function deriveGraph(view: PreparedEmbedding, epochIndex: number, propagationStep = -1): DerivedGraph {
  const count = view.count;
  const labeledSubset = view.subsets.map((name) => name === "labeled");
  const unlabeledSubset = view.subsets.map((name) => name === "unlabeled");
  if (view.graph) {
    const { k, knn, nearestLabeled } = view.graph;
    const purity = new Float32Array(count).fill(Number.NaN);
    for (let index = 0; index < count; index += 1) {
      let valid = 0;
      let same = 0;
      for (let slot = 0; slot < k; slot += 1) {
        const neighbor = knn[index * k + slot];
        if (neighbor < 0) continue;
        valid += 1;
        if (view.classIndex[neighbor] === view.classIndex[index]) same += 1;
      }
      if (valid > 0) purity[index] = same / valid;
    }
    const labeledClasses = new Set<number>();
    for (let index = 0; index < count; index += 1) {
      if (labeledSubset[view.subset[index]]) labeledClasses.add(view.classIndex[index]);
    }
    const agreement = new Int8Array(count).fill(-1);
    const pulls = new Int32Array(count);
    const pullsSameClass = new Int32Array(count);
    for (let index = 0; index < count; index += 1) {
      const nearest = nearestLabeled[index];
      if (nearest < 0 || labeledSubset[view.subset[index]]) continue;
      const same = view.classIndex[nearest] === view.classIndex[index];
      // A class the labeled support never shows (class-disjoint validation/test) cannot agree.
      if (labeledClasses.has(view.classIndex[index])) agreement[index] = same ? 1 : 0;
      if (unlabeledSubset[view.subset[index]]) {
        pulls[nearest] += 1;
        if (same) pullsSameClass[nearest] += 1;
      }
    }
    return {
      purity,
      agreement,
      agreementSource: "the nearest labeled sample in the trained embedding",
      pulls,
      pullsSameClass,
      ...NO_EVIDENCE,
    };
  }
  const metrics = view.tracking?.nodeMetrics[epochIndex];
  if (metrics) {
    const fraction = metrics.same_label_neighbor_fraction ?? null;
    const strongestSame = metrics.strongest_neighbor_same_label ?? null;
    const agreement = strongestSame
      ? Int8Array.from(strongestSame, (value) => (Number.isNaN(value) ? -1 : value > 0.5 ? 1 : 0))
      : null;
    return {
      purity: fraction,
      agreement,
      agreementSource: agreement ? "the strongest neighbor in the run's graph at this epoch" : null,
      pulls: null,
      pullsSameClass: null,
      ...evidenceAt(view, epochIndex),
      ...propagationAt(view, epochIndex, propagationStep),
    };
  }
  return {
    purity: null,
    agreement: null,
    agreementSource: null,
    pulls: null,
    pullsSameClass: null,
    ...(view.tracking ? { ...evidenceAt(view, epochIndex), ...propagationAt(view, epochIndex, propagationStep) } : NO_EVIDENCE),
  };
}

export interface EncodingInput {
  view: PreparedEmbedding;
  colorMode: ColorMode;
  classColors: Uint8Array | null;
  clusterColors: Uint8Array | null;
  /** The cluster per point to color by (-1 none). */
  clusters: Int32Array | null;
  derived: DerivedGraph;
  /** The reference epoch's derived data, for the change modes. */
  reference: DerivedGraph | null;
  pinnedClasses: number[];
  splitEmphasis: SplitEmphasis;
  selectedIndex: number | null;
  selectionMask: Uint8Array | null;
}

/** Each point's color (index into CANVAS_COLORS) and emphasis tier (0 dim, 1 normal, 2 emphasized). */
export function encodePoints(input: EncodingInput): { color: Uint8Array; tier: Uint8Array } {
  const { view, colorMode, classColors, clusterColors, derived, reference, pinnedClasses, splitEmphasis, selectionMask } =
    input;
  const count = view.count;
  const color = new Uint8Array(count);
  const tier = new Uint8Array(count);
  const pinnedSlot = new Map(pinnedClasses.map((classIndex, slot) => [classIndex, slot]));
  const focusClass = input.selectedIndex != null ? view.classIndex[input.selectedIndex] : null;
  const subsetSlot = view.subsets.map((name) => subsetStyle(name).slot);
  const labeledSubset = view.subsets.map((name) => name === "labeled" || name === "train");
  const unlabeledSubset = view.subsets.map((name) => name === "unlabeled");
  const cluster = input.clusters;

  for (let index = 0; index < count; index += 1) {
    const subset = view.subset[index];
    const classIndex = view.classIndex[index];
    let colorIndex: number = NEUTRAL_INDEX;
    let pointTier = 1;
    switch (colorMode) {
      case "pinned": {
        const slot = pinnedSlot.get(classIndex);
        colorIndex = slot == null ? NEUTRAL_INDEX : slot;
        pointTier = slot == null ? 0 : 2;
        break;
      }
      case "class":
        colorIndex = classColors ? classColors[classIndex] : NEUTRAL_INDEX;
        break;
      case "agreement": {
        const value = derived.agreement ? derived.agreement[index] : -1;
        colorIndex = value === 1 ? AGREE_INDEX : value === 0 ? DISAGREE_INDEX : NEUTRAL_INDEX;
        break;
      }
      case "purity": {
        const value = derived.purity ? derived.purity[index] : Number.NaN;
        colorIndex = Number.isNaN(value)
          ? NEUTRAL_INDEX
          : RAMP_OFFSET + Math.min(RAMP.length - 1, Math.floor(value * RAMP.length));
        break;
      }
      case "outlier": {
        const flagged = derived.outlier ? derived.outlier[index] : 0;
        colorIndex = derived.outlier ? (flagged ? DISAGREE_INDEX : NEUTRAL_INDEX) : NEUTRAL_INDEX;
        pointTier = flagged ? 2 : 0;
        break;
      }
      case "centerSimilarity": {
        const value = derived.centerSimilarity ? derived.centerSimilarity[index] : Number.NaN;
        // Cosine to the own center, shown over [0, 1]; a negative cosine is the darkest step.
        colorIndex = Number.isNaN(value)
          ? NEUTRAL_INDEX
          : RAMP_OFFSET + Math.max(0, Math.min(RAMP.length - 1, Math.floor(value * RAMP.length)));
        break;
      }
      case "lpPrediction":
      case "prototypePrediction":
      case "hofferPrediction": {
        const correct =
          colorMode === "lpPrediction"
            ? derived.lpCorrect
            : colorMode === "prototypePrediction"
              ? derived.prototypeCorrect
              : derived.hofferCorrect;
        const value = correct ? correct[index] : -1;
        colorIndex = value === 1 ? AGREE_INDEX : value === 0 ? DISAGREE_INDEX : NEUTRAL_INDEX;
        break;
      }
      case "lpWeight":
      case "hofferCertainty": {
        const values = colorMode === "lpWeight" ? derived.lpWeight : derived.hofferCertainty;
        const top = colorMode === "lpWeight" ? 1 : view.tracking?.hofferCertaintyMax || 1;
        const value = values ? values[index] / top : Number.NaN;
        colorIndex = Number.isNaN(value)
          ? NEUTRAL_INDEX
          : RAMP_OFFSET + Math.max(0, Math.min(RAMP.length - 1, Math.floor(value * RAMP.length)));
        break;
      }
      case "propagationClass": {
        const held = derived.propagationClass ? derived.propagationClass[index] : -1;
        colorIndex = held >= 0 && classColors ? classColors[held] : NEUTRAL_INDEX;
        // A point still holding nothing at this step is context, not a color.
        pointTier = held >= 0 ? pointTier : 0;
        break;
      }
      case "propagationCorrect": {
        const value = derived.propagationCorrect ? derived.propagationCorrect[index] : -1;
        colorIndex = value === 1 ? AGREE_INDEX : value === 0 ? DISAGREE_INDEX : NEUTRAL_INDEX;
        break;
      }
      case "propagationHop": {
        const value = derived.propagationHop ? derived.propagationHop[index] : Number.NaN;
        // Hops are small: 0 is a labeled node, and the ramp's top is "5 or more".
        colorIndex = Number.isNaN(value) ? NEUTRAL_INDEX : RAMP_OFFSET + Math.min(RAMP.length - 1, Math.max(0, value));
        break;
      }
      case "pseudoLabel": {
        const value = derived.pseudo ? derived.pseudo[index] : -1;
        colorIndex = value === 2 ? AGREE_INDEX : value === 1 ? DISAGREE_INDEX : value === 0 ? CHANGE_NONE_INDEX : NEUTRAL_INDEX;
        break;
      }
      case "cluster": {
        const group = cluster ? cluster[index] : -1;
        colorIndex = group >= 0 && clusterColors ? clusterColors[group] : NEUTRAL_INDEX;
        break;
      }
      case "purityChange": {
        const now = derived.purity ? derived.purity[index] : Number.NaN;
        const before = reference?.purity ? reference.purity[index] : Number.NaN;
        if (Number.isNaN(now) || Number.isNaN(before)) colorIndex = NEUTRAL_INDEX;
        else if (now - before > PURITY_CHANGE_THRESHOLD) colorIndex = CHANGE_BETTER_INDEX;
        else if (before - now > PURITY_CHANGE_THRESHOLD) colorIndex = CHANGE_WORSE_INDEX;
        else colorIndex = CHANGE_NONE_INDEX;
        break;
      }
      case "agreementChange": {
        const now = derived.agreement ? derived.agreement[index] : -1;
        const before = reference?.agreement ? reference.agreement[index] : -1;
        if (now < 0 || before < 0) colorIndex = NEUTRAL_INDEX;
        else if (now > before) colorIndex = CHANGE_BETTER_INDEX;
        else if (now < before) colorIndex = CHANGE_WORSE_INDEX;
        else colorIndex = CHANGE_NONE_INDEX;
        break;
      }
      default:
        colorIndex = subsetSlot[subset] < 0 ? NEUTRAL_INDEX : subsetSlot[subset];
    }
    if (splitEmphasis !== "none") {
      const emphasized = splitEmphasis === "labeled" ? labeledSubset[subset] : unlabeledSubset[subset];
      pointTier = emphasized ? (colorMode === "pinned" && colorIndex === NEUTRAL_INDEX ? 1 : 2) : 0;
    }
    if (selectionMask) {
      pointTier = selectionMask[index] ? 2 : 0;
    } else if (focusClass != null) {
      pointTier = classIndex === focusClass ? 2 : pointTier === 2 ? 1 : 0;
    }
    color[index] = colorIndex;
    tier[index] = pointTier;
  }
  return { color, tier };
}

export interface TimelineSeries {
  name: string;
  label: string;
  higherIsBetter: boolean | null;
  /** One value per tracked epoch (null where not recorded). */
  values: (number | null)[];
}

/**
 * Every tracked epoch's summary, so a run can be read as a trajectory rather than
 * as its last frame: the run's own epoch_metrics.csv columns plus quantities
 * derived here from the per-node graph metrics and the saved SLADE clusters.
 */
export function trackingTimeline(view: PreparedEmbedding): TimelineSeries[] {
  const tracking = view.tracking;
  if (!tracking) return [];
  const unlabeledSubset = view.subsets.indexOf("unlabeled");
  const series: TimelineSeries[] = [];
  const derivedPerEpoch = tracking.epochs.map((_, epochIndex) => deriveGraph(view, epochIndex));
  const meanOverUnlabeled = (values: ArrayLike<number> | null, transform = (value: number) => value) => {
    if (!values) return null;
    let sum = 0;
    let count = 0;
    for (let row = 0; row < view.count; row += 1) {
      if (unlabeledSubset >= 0 && view.subset[row] !== unlabeledSubset) continue;
      const value = values[row];
      if (value == null || Number.isNaN(value) || value < 0) continue;
      sum += transform(value);
      count += 1;
    }
    return count ? sum / count : null;
  };
  series.push({
    name: "unlabeled_neighbor_purity",
    label: "Neighbor purity · unlabeled",
    higherIsBetter: true,
    values: derivedPerEpoch.map((derived) => meanOverUnlabeled(derived.purity)),
  });
  series.push({
    name: "unlabeled_strongest_agrees",
    label: "Strongest neighbor same class · unlabeled",
    higherIsBetter: true,
    values: derivedPerEpoch.map((derived) => meanOverUnlabeled(derived.agreement)),
  });
  if (tracking.pseudoLabels) {
    const frames = tracking.pseudoLabels;
    series.push({
      name: "pseudo_label_accepted_accuracy",
      label: "Pseudo labels accepted · correct",
      higherIsBetter: true,
      values: derivedPerEpoch.map((derived) => {
        if (!derived.pseudo) return null;
        let accepted = 0;
        let correct = 0;
        derived.pseudo.forEach((value) => {
          if (value >= 1) accepted += 1;
          if (value === 2) correct += 1;
        });
        return accepted ? correct / accepted : null;
      }),
    });
    series.push({
      name: "pseudo_label_accepted_share",
      label: "Pseudo labels accepted · share of unlabeled",
      higherIsBetter: null,
      values: frames.map((frame) => {
        if (!frame) return null;
        let unlabeled = 0;
        let accepted = 0;
        for (let row = 0; row < view.count; row += 1) {
          if (view.subset[row] !== unlabeledSubset) continue;
          unlabeled += 1;
          if (frame.accepted[row]) accepted += 1;
        }
        return unlabeled ? accepted / unlabeled : null;
      }),
    });
  }
  if (tracking.evidence) {
    const certainHalfError = (correct: Int8Array | null, certainty: Float32Array | null) => {
      if (!correct || !certainty) return null;
      const rows: number[] = [];
      for (let row = 0; row < view.count; row += 1) if (correct[row] >= 0) rows.push(row);
      if (!rows.length) return null;
      rows.sort((a, b) => certainty[b] - certainty[a]);
      const half = rows.slice(0, Math.max(1, Math.floor(rows.length / 2)));
      return half.filter((row) => correct[row] === 0).length / half.length;
    };
    series.push({
      name: "lp_certain_half_error",
      label: "Label propagation · error in its more certain half",
      higherIsBetter: false,
      values: derivedPerEpoch.map((derived) => certainHalfError(derived.lpCorrect, derived.lpWeight)),
    });
    series.push({
      name: "hoffer_certain_half_error",
      label: "Hoffer · error in its more certain half",
      higherIsBetter: false,
      values: derivedPerEpoch.map((derived) => certainHalfError(derived.hofferCorrect, derived.hofferCertainty)),
    });
  }
  if (tracking.propagation) {
    // The run's own numbers for the epoch, which a backfilled epoch has in its
    // file rather than in epoch_metrics.csv.
    const fromSummary = (key: string) =>
      tracking.propagation!.map((frame) => (frame ? frame.summary[key] ?? null : null));
    (
      [
        ["lp_reached_share", "Label propagation · unlabeled reached", null],
        ["lp_tree_edge_purity", "Label propagation · first-arrival edge stays in class", true],
        ["lp_source_accuracy", "Label propagation · mass came from the true class", true],
        ["lp_influence_share", "Label propagation · share carried by the strongest neighbor", null],
      ] as const
    ).forEach(([key, label, higherIsBetter]) => {
      series.push({ name: key, label, higherIsBetter, values: fromSummary(key) });
    });
  }
  if (tracking.clusters) {
    series.push({
      name: "cluster_purity",
      label: "Cluster purity · SLADE refresh",
      higherIsBetter: true,
      values: tracking.clusters.map((clusters) => (clusters ? clusterPurity(view, clusters) : null)),
    });
  }
  const byEpoch = new Map(tracking.epochMetrics.map((row) => [row.epoch, row]));
  for (const metric of TRACKING_METRICS) {
    const values = tracking.epochs.map((epoch) => byEpoch.get(epoch)?.[metric.name] ?? null);
    if (values.some((value) => value != null)) series.push({ ...metric, values });
  }
  return series.filter((entry) => entry.values.some((value) => value != null));
}

/** A class predictor the diagnostics compare: diffusion, the network itself, Hoffer, or the run's own pseudo labels. */
export type Predictor = "lp" | "prototype" | "hoffer" | "pseudo";

export const PREDICTOR_LABELS: Record<Predictor, string> = {
  lp: "Label propagation (diffusion)",
  prototype: "Network (class prototypes)",
  hoffer: "Hoffer reference softmax",
  pseudo: "The run's own pseudo labels",
};

export interface PredictorFrame {
  /** Unlabeled points: 1 right, 0 wrong; -1 not scored (labeled, or not a candidate). */
  correct: Int8Array;
  /** The confidence the predictor attaches (Iscen's ω for diffusion, 1 - normalized entropy otherwise). */
  confidence: Float32Array;
  /** Predicted view class index (-1 none). */
  predicted: Int32Array;
}

/** One predictor at one epoch, or null where the view has none. */
export function predictorAt(view: PreparedEmbedding, epochIndex: number, predictor: Predictor): PredictorFrame | null {
  const unlabeledSubset = view.subsets.indexOf("unlabeled");
  if (unlabeledSubset < 0 || !view.tracking) return null;
  if (predictor === "pseudo") {
    const frame = view.tracking.pseudoLabels?.[epochIndex];
    if (!frame) return null;
    const correct = new Int8Array(view.count).fill(-1);
    for (let row = 0; row < view.count; row += 1) {
      if (view.subset[row] !== unlabeledSubset || frame.label[row] < 0) continue;
      correct[row] = frame.label[row] === view.classIndex[row] ? 1 : 0;
    }
    return { correct, confidence: frame.confidence, predicted: frame.label };
  }
  const map = view.tracking.evidence?.[epochIndex]?.[predictor];
  if (!map) return null;
  const correct = topClassCorrect(view, map);
  const predicted = new Int32Array(view.count);
  for (let row = 0; row < view.count; row += 1) predicted[row] = map.topClass[row * map.topK];
  return { correct, confidence: predictor === "lp" && map.weight ? map.weight : map.certainty, predicted };
}

export function frameAccuracy(frame: PredictorFrame | null): number | null {
  if (!frame) return null;
  let right = 0;
  let total = 0;
  frame.correct.forEach((value) => {
    if (value < 0) return;
    total += 1;
    right += value;
  });
  return total ? right / total : null;
}

/** Accuracy among the most confident `share` of scored points (1 = all of them). */
export function topConfidenceAccuracy(frame: PredictorFrame | null, share: number): number | null {
  if (!frame) return null;
  const rows: number[] = [];
  frame.correct.forEach((value, row) => {
    if (value >= 0) rows.push(row);
  });
  if (!rows.length) return null;
  rows.sort((a, b) => frame.confidence[b] - frame.confidence[a]);
  const kept = rows.slice(0, Math.max(1, Math.round(rows.length * share)));
  return kept.reduce((sum, row) => sum + frame.correct[row], 0) / kept.length;
}

/** Confidence histograms of right and wrong predictions, each as shares of its own group. */
export function confidenceHistogram(frame: PredictorFrame, bins: number, top: number) {
  const right = new Array<number>(bins).fill(0);
  const wrong = new Array<number>(bins).fill(0);
  let rightTotal = 0;
  let wrongTotal = 0;
  frame.correct.forEach((value, row) => {
    if (value < 0) return;
    const bin = Math.max(0, Math.min(bins - 1, Math.floor((frame.confidence[row] / (top || 1)) * bins)));
    if (value) {
      right[bin] += 1;
      rightTotal += 1;
    } else {
      wrong[bin] += 1;
      wrongTotal += 1;
    }
  });
  return {
    right: right.map((count) => (rightTotal ? count / rightTotal : 0)),
    wrong: wrong.map((count) => (wrongTotal ? count / wrongTotal : 0)),
    rightTotal,
    wrongTotal,
  };
}

export interface Confusion {
  trueClass: number;
  predictedClass: number;
  rows: number[];
}

/**
 * Recurring mistakes among the confident: wrong predictions whose confidence is
 * in the top `share` of all scored predictions, grouped by (true, predicted) class.
 */
export function confidentConfusions(frame: PredictorFrame, view: PreparedEmbedding, share: number): Confusion[] {
  const scored: number[] = [];
  frame.correct.forEach((value, row) => {
    if (value >= 0) scored.push(row);
  });
  if (!scored.length) return [];
  const confidences = scored.map((row) => frame.confidence[row]).sort((a, b) => b - a);
  const cutoff = confidences[Math.max(0, Math.round(confidences.length * share) - 1)];
  const groups = new Map<string, Confusion>();
  scored.forEach((row) => {
    if (frame.correct[row] !== 0 || frame.confidence[row] < cutoff || frame.predicted[row] < 0) return;
    const key = `${view.classIndex[row]}:${frame.predicted[row]}`;
    const group = groups.get(key) ?? { trueClass: view.classIndex[row], predictedClass: frame.predicted[row], rows: [] };
    group.rows.push(row);
    groups.set(key, group);
  });
  return [...groups.values()]
    .map((group) => ({ ...group, rows: group.rows.sort((a, b) => frame.confidence[b] - frame.confidence[a]) }))
    .sort((a, b) => b.rows.length - a.rows.length);
}

/** Share of clustered points whose cluster's majority class is their own. */
export function clusterPurity(view: PreparedEmbedding, clusters: Int32Array): number | null {
  const counts = new Map<number, Map<number, number>>();
  let total = 0;
  for (let row = 0; row < view.count; row += 1) {
    const group = clusters[row];
    if (group < 0) continue;
    total += 1;
    let classCounts = counts.get(group);
    if (!classCounts) {
      classCounts = new Map();
      counts.set(group, classCounts);
    }
    classCounts.set(view.classIndex[row], (classCounts.get(view.classIndex[row]) ?? 0) + 1);
  }
  if (!total) return null;
  let majority = 0;
  counts.forEach((classCounts) => {
    majority += Math.max(...classCounts.values());
  });
  return majority / total;
}

/** Epoch-tracking series worth comparing against the epoch -1 baseline. */
export const TRACKING_METRICS: { name: string; label: string; higherIsBetter: boolean | null }[] = [
  { name: "graph_edge_purity", label: "Graph edge purity", higherIsBetter: true },
  { name: "graph_weighted_edge_purity", label: "Weighted edge purity", higherIsBetter: true },
  { name: "graph_nodes_without_labeled_component", label: "Nodes cut off from labeled", higherIsBetter: false },
  { name: "embedding_class_separation", label: "Class separation (cosine)", higherIsBetter: true },
  { name: "embedding_effective_rank", label: "Effective rank", higherIsBetter: null },
  { name: "test_mean_average_precision_at_r", label: "Test MAP@R", higherIsBetter: true },
  { name: "lp_unlabeled_accuracy", label: "Label propagation · accuracy on unlabeled", higherIsBetter: true },
  { name: "lp_weight_auc", label: "Label propagation · weight ranks correctness (AUC)", higherIsBetter: true },
  { name: "lp_unlabeled_certainty_mean", label: "Label propagation · mean certainty", higherIsBetter: null },
  { name: "lp_confident_wrong_share", label: "Label propagation · wrong with weight ≥ 0.5", higherIsBetter: false },
  { name: "hoffer_unlabeled_accuracy", label: "Hoffer · accuracy on unlabeled", higherIsBetter: true },
  { name: "hoffer_certainty_auc", label: "Hoffer · certainty ranks correctness (AUC)", higherIsBetter: true },
  { name: "hoffer_unlabeled_certainty_mean", label: "Hoffer · mean certainty (1 − entropy)", higherIsBetter: null },
  { name: "hoffer_confident_wrong_share", label: "Hoffer · wrong with certainty ≥ 0.5", higherIsBetter: false },
];

export function formatCount(value: number): string {
  return value.toLocaleString("en-US");
}
