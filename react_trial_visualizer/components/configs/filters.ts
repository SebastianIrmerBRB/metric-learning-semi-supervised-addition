import { KIND_GROUP, type ConfigEntry, type ConfigPhase, type Phase2Kind, type SectionId } from "@/lib/study-configs";

export const ALL = "*";

export type P2Group = "all" | "ablation" | "geometry";

/** Filters shared by the views, so a check can open the table it is about. */
export interface Filters {
  phase: ConfigPhase;
  dataset: string;
  method: string;
  loss: string;
  p2group: P2Group;
  kind: Phase2Kind | typeof ALL;
  root: string;
}

export const DEFAULT_FILTERS: Filters = {
  phase: "phase1",
  dataset: ALL,
  method: ALL,
  loss: ALL,
  p2group: "all",
  kind: ALL,
  root: ALL,
};

export function matchesPhase2(entry: ConfigEntry, filters: Pick<Filters, "p2group" | "kind" | "root">): boolean {
  if (entry.phase !== "phase2") return true;
  if (filters.p2group !== "all" && KIND_GROUP[entry.kind ?? "other"] !== filters.p2group) return false;
  if (filters.kind !== ALL && entry.kind !== filters.kind) return false;
  if (filters.root !== ALL && entry.root !== filters.root) return false;
  return true;
}

export const DEFAULT_SECTIONS: SectionId[] = ["protocol", "data", "optim", "loss", "ssl", "hpo", "seeds"];
