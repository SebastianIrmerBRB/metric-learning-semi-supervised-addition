/** Types mirroring the payload of `scripts/studio_service.py`'s `schema` op. */

export interface CliArgument {
  dest: string;
  options: string[];
  type: string | null;
  nargs: string | number | null;
  isFlag: boolean;
  isList: boolean;
  choices: unknown[] | null;
  default: unknown;
  help: string;
  required: boolean;
}

export interface ComponentParam {
  name: string;
  default: unknown;
  hasDefault: boolean;
  declaredBy: string;
}

export interface SslMethodKey {
  key: string;
  value: unknown;
  valueType: string;
}

export interface SslMethod {
  name: string;
  path: string;
  method: string | null;
  keys: SslMethodKey[];
}

export interface HpoField {
  name: string;
  annotation: string;
  default: unknown;
}

export interface StudioSchema {
  repoRoot: string;
  python: string;
  cliArguments: CliArgument[];
  componentParams: {
    loss: Record<string, ComponentParam[]>;
    miner: Record<string, ComponentParam[]>;
  };
  sslMethods: SslMethod[];
  sslConfigFields: { name: string; key: string; annotation: string; default: unknown }[];
  hpoFields: HpoField[];
  usedSpaceKeys: { key: string; count: number }[];
  constants: {
    datasets: string[];
    losses: string[];
    classificationLosses: string[];
    miners: string[];
    objectiveMetrics: string[];
    selectionMetrics: string[];
    samplers: string[];
    pruners: string[];
    directions: string[];
    labelSamplingModes: string[];
    batchSamplerKey: string;
    labeledBatchSizeKeys: string[];
    labeledBatchFractionKey: string;
    labeledBatchFractionAliases: string[];
    maxLabeledBatchFraction: number;
    comparisonForbiddenKeys: string[];
    hpoModeKeys: string[];
    twoStreamSamplerMethods: string[];
  };
  manifest: {
    version: number;
    topLevelKeys: string[];
    runKeys: string[];
    studyReplayKeys: string[];
    studyDirModes: string[];
    defaultStudyDirMode: string;
    comparisonSeedTargets: string[];
    finalTestVisualizationModes: string[];
    schedulerOwnedOptions: string[];
    studyReplayOwnedOptions: string[];
    runNamePattern: string;
    defaults: Record<string, unknown>;
  };
}

export type ConfigKind = "manifest" | "hparam" | "ssl-method" | "experiment" | "unknown";

export interface ConfigFileEntry {
  path: string;
  name: string;
  directory: string;
  size: number;
  modified: string;
  kind: ConfigKind;
}

export type IssueLevel = "error" | "warning" | "info";

export interface Issue {
  level: IssueLevel;
  message: string;
  field?: string;
}

export interface LaunchStatus {
  id: string;
  kind: "main" | "scheduler";
  label: string;
  command: string[];
  pid: number;
  startedAt: string;
  logPath: string;
  dryRun: boolean;
  finishedAt?: string;
  stoppedByUser?: boolean;
  running: boolean;
  logSize: number;
}

/** A single Optuna search dimension, in the shape the JSON configs use. */
export type SpaceSpec =
  | unknown[]
  | {
      type?: "float" | "int" | "categorical";
      low?: number;
      high?: number;
      log?: boolean;
      step?: number | null;
      choices?: unknown[];
    };
