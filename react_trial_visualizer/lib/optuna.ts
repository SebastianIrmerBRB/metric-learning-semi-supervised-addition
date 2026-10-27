export type OptunaParamValue = string | number | boolean | null;

export interface OptunaTrialSnapshot {
  number: number;
  objective: number;
  params: Record<string, OptunaParamValue>;
}

export interface OptunaParameterSnapshot {
  name: string;
  rank: number;
  importance: number;
  kind: "numeric" | "categorical";
  distribution: "float" | "int" | "categorical";
  log: boolean;
  low?: number;
  high?: number;
  step?: number;
  choices?: OptunaParamValue[];
}

export interface OptunaStudySnapshot {
  id: string;
  relativePath: string;
  studyName: string;
  displayName: string;
  dataset: string;
  method: string;
  loss: string;
  labelSetting: string;
  collection: string;
  direction: "maximize" | "minimize";
  target: string;
  sampler: string;
  completeTrials: number;
  reportedCompleteTrials: number;
  bestTrial: {
    number: number;
    objective: number;
  } | null;
  parameters: OptunaParameterSnapshot[];
  trials: OptunaTrialSnapshot[];
}

export interface OptunaSnapshot {
  schemaVersion: 1;
  generatedAt: string;
  source: string;
  totalStudyDirectories: number;
  skippedStudyDirectories: number;
  studies: OptunaStudySnapshot[];
  warning?: string;
  failures?: Array<{
    relativeStudyDir: string;
    error: string;
  }>;
}

export function isOptunaSnapshot(value: unknown): value is OptunaSnapshot {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<OptunaSnapshot>;
  return candidate.schemaVersion === 1 && Array.isArray(candidate.studies);
}
