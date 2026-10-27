"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type DragEvent } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  SlidersVertical,
  ChevronRight,
  Database,
  FileCog,
  FilePlus2,
  FolderOpen,
  Gauge,
  GitCompareArrows,
  Library,
  LayoutDashboard,
  ListChecks,
  Menu,
  Orbit,
  Plus,
  Rows3,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  Timer,
  Trash2,
  Trophy,
  UploadCloud,
  X,
  FlaskConical,
} from "lucide-react";
import {
  DevelopmentChart,
  DistributionChart,
  EpochChart,
  LearningRateChart,
  StudyComparisonChart,
} from "@/components/Charts";
import { ChartBoard } from "@/components/ChartBoard";
import { ArchiveExplorer } from "@/components/ArchiveExplorer";
import { ComparisonWorkspace } from "@/components/ComparisonWorkspace";
import { CorrelationExplorer } from "@/components/CorrelationExplorer";
import { OptunaExplorer } from "@/components/OptunaExplorer";
import { SelectedTrialPopover } from "@/components/SelectedTrialPopover";
import { TrainingCurves } from "@/components/TrainingCurves";
import { TrialsTable } from "@/components/TrialsTable";
import type { TestResultFile } from "@/lib/test-results";
import {
  columnLabel,
  createStudy,
  formatDuration,
  formatMetric,
  getStudyStats,
  numericColumns,
  type SeedFile,
  type Study,
  type StudyConfig,
  type TrialRow,
} from "@/lib/trials";

interface TrialVisualizerProps {
  initialFiles: SeedFile[];
  initialTestResultsFile: TestResultFile | null;
}

type View = "overview" | "compare" | "parameters" | "archive" | "study";
type FileImportMode = "csv-files" | "log-folder";

const WORKSPACE_SELECTION_KEY = "trial-atlas:workspace-selection";
const DIRECTORY_INPUT_ATTRIBUTES = {
  directory: "",
  webkitdirectory: "",
};

function fileImportPath(file: File): string {
  return (file.webkitRelativePath || file.name).replaceAll("\\", "/");
}

function isTrialsCsv(file: File): boolean {
  return fileImportPath(file).split("/").at(-1)?.toLowerCase() === "trials.csv";
}

interface PersistedWorkspaceSession {
  version: 2;
  studies: Study[];
  selectedId: string | null;
  view: View;
}

interface PersistedTrialSelection {
  version: 2;
  curveIndices: number[];
  activeIndex: number | null;
  summaryOpen: boolean;
}

function isStoredStudy(value: unknown): value is Study {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<Study>;
  return (
    typeof candidate.id === "string" &&
    typeof candidate.name === "string" &&
    typeof candidate.fileName === "string" &&
    typeof candidate.dataset === "string" &&
    (candidate.source === "bundled" || candidate.source === "uploaded") &&
    Array.isArray(candidate.columns) &&
    Array.isArray(candidate.rows) &&
    Boolean(candidate.config) &&
    typeof candidate.config?.trial === "string" &&
    typeof candidate.config?.metric === "string"
  );
}

function restoreBundledStudy(current: Study, stored: Study): Study {
  const config = stored.config;
  if (!current.columns.includes(config.trial) || !current.columns.includes(config.metric)) return current;
  const validOptional = (column: string | null) => column == null || current.columns.includes(column);
  return {
    ...current,
    config: {
      ...config,
      learningRate: validOptional(config.learningRate) ? config.learningRate : current.config.learningRate,
      state: validOptional(config.state) ? config.state : current.config.state,
      epoch: validOptional(config.epoch) ? config.epoch : current.config.epoch,
      selectedEpoch: validOptional(config.selectedEpoch) ? config.selectedEpoch : current.config.selectedEpoch,
      duration: validOptional(config.duration) ? config.duration : current.config.duration,
    },
  };
}

const inputClass =
  "focus-ring h-10 w-full appearance-none rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 text-xs text-[#d9e0ed]";

function KpiCard({
  label,
  value,
  context,
  icon,
  accent = "violet",
}: {
  label: string;
  value: string;
  context: string;
  icon: React.ReactNode;
  accent?: "violet" | "cyan" | "lime" | "amber";
}) {
  const styles = {
    violet: "bg-[#9b8cff]/10 text-[#b5aaff] ring-[#9b8cff]/15",
    cyan: "bg-[#4fd5e7]/10 text-[#6fe1ee] ring-[#4fd5e7]/15",
    lime: "bg-[#b8e56e]/10 text-[#c9f18a] ring-[#b8e56e]/15",
    amber: "bg-[#f4c86f]/10 text-[#f6d488] ring-[#f4c86f]/15",
  }[accent];

  return (
    <article className="surface relative overflow-hidden px-5 py-5">
      <div className="absolute -right-10 -top-12 h-28 w-28 rounded-full bg-white/[0.025] blur-xl" />
      <div className="relative flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[#77849c]">{label}</p>
          <p className="mt-3 truncate text-2xl font-semibold tracking-[-0.035em] text-white">{value}</p>
          <p className="mt-1.5 truncate text-[11px] text-[#7f8ca3]">{context}</p>
        </div>
        <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1 ${styles}`}>{icon}</span>
      </div>
    </article>
  );
}

function SelectField({
  label,
  value,
  columns,
  allowNone = false,
  onChange,
}: {
  label: string;
  value: string | null;
  columns: string[];
  allowNone?: boolean;
  onChange: (value: string | null) => void;
}) {
  return (
    <label className="block min-w-0">
      <span className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">{label}</span>
      <select value={value ?? ""} onChange={(event) => onChange(event.target.value || null)} className={inputClass}>
        {allowNone && <option value="">Not used</option>}
        {columns.map((column) => (
          <option key={column} value={column}>{columnLabel(column)}</option>
        ))}
      </select>
    </label>
  );
}

function ConfigurationPanel({ study, onChange }: { study: Study; onChange: (config: StudyConfig) => void }) {
  const numbers = numericColumns(study);
  const set = <Key extends keyof StudyConfig>(key: Key, value: StudyConfig[Key]) => onChange({ ...study.config, [key]: value });

  return (
    <section className="surface fade-up mb-6 overflow-hidden">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[0.07] px-5 py-4 sm:px-6">
        <div>
          <h2 className="text-sm font-semibold text-white">Diagram columns</h2>
          <p className="mt-1.5 text-xs text-[#8290a7]">Automatic matches follow the schema priorities from the Python dashboard.</p>
        </div>
        <span className="rounded-full border border-[#4fd5e7]/20 bg-[#4fd5e7]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#70ddeb]">
          Live preview
        </span>
      </header>
      <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 sm:p-6">
        <SelectField label="Trial axis" value={study.config.trial} columns={numbers} onChange={(value) => value && set("trial", value)} />
        <SelectField label="Primary metric" value={study.config.metric} columns={numbers} onChange={(value) => value && set("metric", value)} />
        <SelectField label="Learning rate" value={study.config.learningRate} columns={numbers} allowNone onChange={(value) => set("learningRate", value)} />
        <SelectField label="State" value={study.config.state} columns={study.columns} allowNone onChange={(value) => set("state", value)} />
        <SelectField label="Last epoch" value={study.config.epoch} columns={numbers} allowNone onChange={(value) => set("epoch", value)} />
        <SelectField label="Selected epoch" value={study.config.selectedEpoch} columns={numbers} allowNone onChange={(value) => set("selectedEpoch", value)} />
        <SelectField label="Duration" value={study.config.duration} columns={numbers} allowNone onChange={(value) => set("duration", value)} />
      </div>
    </section>
  );
}

interface SidebarProps {
  studies: Study[];
  repositoryStudies: Study[];
  selectedId: string | null;
  view: View;
  mobileOpen: boolean;
  onClose: () => void;
  onOverview: () => void;
  onCompare: () => void;
  onParameters: () => void;
  onArchive: () => void;
  onSelect: (study: Study) => void;
  onAddStudy: (study: Study) => void;
  onRemove: (study: Study) => void;
  onFiles: (files: FileList | File[]) => void;
  onFolder: (files: FileList | File[]) => void;
}

function Sidebar({ studies, repositoryStudies, selectedId, view, mobileOpen, onClose, onOverview, onCompare, onParameters, onArchive, onSelect, onAddStudy, onRemove, onFiles, onFolder }: SidebarProps) {
  const [dragging, setDragging] = useState(false);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const loadedIds = new Set(studies.map((study) => study.id));
  const availableStudies = repositoryStudies.filter((study) => !loadedIds.has(study.id));

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    onFiles(Array.from(event.dataTransfer.files));
  };

  return (
    <>
      {mobileOpen && <button type="button" className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm lg:hidden" onClick={onClose} aria-label="Close navigation" />}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-[286px] flex-col border-r border-white/[0.07] bg-[#090f1d]/95 shadow-2xl backdrop-blur-xl transition-transform duration-200 lg:translate-x-0 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-20 items-center justify-between border-b border-white/[0.07] px-5">
          <button type="button" onClick={onOverview} className="focus-ring flex items-center gap-3 rounded-xl text-left">
            <span className="relative flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-[#aa9cff] to-[#6854df] text-white shadow-[0_10px_35px_rgba(117,87,255,.25)]">
              <Orbit size={21} strokeWidth={2.2} />
              <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full border-2 border-[#090f1d] bg-[#b8e56e]" />
            </span>
            <span>
              <span className="block text-sm font-semibold tracking-[-0.01em] text-white">Trial Atlas</span>
              <span className="mt-0.5 block text-[10px] uppercase tracking-[0.13em] text-[#6f7e98]">Visual workspace</span>
            </span>
          </button>
          <button type="button" onClick={onClose} className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg text-[#7f8ba2] hover:bg-white/[0.06] lg:hidden" aria-label="Close navigation">
            <X size={17} />
          </button>
        </div>

        <nav className="min-h-0 flex-1 overflow-y-auto px-3 py-5" aria-label="Studies">
          <p className="px-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#596780]">Workspace</p>
          <button
            type="button"
            onClick={onOverview}
            className={`focus-ring mt-2 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold transition ${
              view === "overview" ? "bg-white/[0.07] text-white" : "text-[#8895aa] hover:bg-white/[0.035] hover:text-white"
            }`}
          >
            <LayoutDashboard size={16} className={view === "overview" ? "text-[#b8e56e]" : "text-[#65748e]"} />
            All studies
            <span className="ml-auto rounded-full bg-white/[0.06] px-2 py-0.5 text-[10px] text-[#8592a8]">{studies.length}</span>
          </button>
          <button
            type="button"
            onClick={onCompare}
            className={`focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold transition ${
              view === "compare" ? "bg-white/[0.07] text-white" : "text-[#8895aa] hover:bg-white/[0.035] hover:text-white"
            }`}
          >
            <GitCompareArrows size={16} className={view === "compare" ? "text-[#4fd5e7]" : "text-[#65748e]"} />
            Compare runs
            <span className="ml-auto rounded-full bg-[#4fd5e7]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#63dce9]">New</span>
          </button>
          <button
            type="button"
            onClick={onParameters}
            className={`focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold transition ${
              view === "parameters" ? "bg-white/[0.07] text-white" : "text-[#8895aa] hover:bg-white/[0.035] hover:text-white"
            }`}
          >
            <SlidersHorizontal size={16} className={view === "parameters" ? "text-[#f4c86f]" : "text-[#65748e]"} />
            Hyperparameters
            <span className="ml-auto rounded-full bg-[#f4c86f]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#f6d488]">Optuna</span>
          </button>
          <button
            type="button"
            onClick={onArchive}
            className={`focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold transition ${
              view === "archive" ? "bg-white/[0.07] text-white" : "text-[#8895aa] hover:bg-white/[0.035] hover:text-white"
            }`}
          >
            <Library size={16} className={view === "archive" ? "text-[#9b8cff]" : "text-[#65748e]"} />
            HPO archive
            <span className="ml-auto rounded-full bg-[#9b8cff]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#b6acff]">Sorted</span>
          </button>
          <Link
            href="/studio/spaces/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <SlidersVertical size={16} className="text-[#65748e]" />
            Config studio
            <span className="ml-auto rounded-full bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#c7ef85]">Edit</span>
          </Link>
          <Link
            href="/epoch-tracking/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <Activity size={16} className="text-[#65748e]" />
            Epoch tracking
            <span className="ml-auto rounded-full bg-[#4fd5e7]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#7fe3f0]">Structure</span>
          </Link>
          <Link
            href="/embeddings/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <Orbit size={16} className="text-[#65748e]" />
            Embeddings
            <span className="ml-auto rounded-full bg-[#4fd5e7]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#7fe3f0]">Live · 3D</span>
          </Link>
          <Link
            href="/phase2/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <FlaskConical size={16} className="text-[#65748e]" />
            Phase 2 results
            <span className="ml-auto rounded-full bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#c7ef85]">2×2 · B6</span>
          </Link>
          <Link
            href="/todo/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <ListChecks size={16} className="text-[#65748e]" />
            Phase 2 todo
            <span className="ml-auto rounded-full bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#c7ef85]">Next</span>
          </Link>
          <Link
            href="/timings/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <Timer size={16} className="text-[#65748e]" />
            Timings
            <span className="ml-auto rounded-full bg-[#4fd5e7]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#7fe3f0]">GPU</span>
          </Link>
          <Link
            href="/configs/"
            className="focus-ring mt-1 flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-xs font-semibold text-[#8895aa] transition hover:bg-white/[0.035] hover:text-white"
          >
            <FileCog size={16} className="text-[#65748e]" />
            Run configs
            <span className="ml-auto rounded-full bg-[#9b8cff]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#b6acff]">Checks</span>
          </Link>

          <div className="mt-7 flex items-center justify-between px-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#596780]">Experiments</p>
            <span className="text-[10px] text-[#4f5d75]">Metric learning</span>
          </div>
          <div className="mt-2 space-y-1">
            {studies.map((study) => {
              const selected = view === "study" && selectedId === study.id;
              return (
                <div key={study.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(study)}
                    className={`focus-ring flex w-full items-center gap-3 rounded-xl px-3 py-3 pr-9 text-left transition ${
                      selected ? "bg-[#9b8cff]/[0.11] text-white ring-1 ring-inset ring-[#9b8cff]/20" : "text-[#94a0b5] hover:bg-white/[0.035] hover:text-white"
                    }`}
                    title={study.fileName}
                  >
                    <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl ${selected ? "bg-[#9b8cff]/15 text-[#b5aaff]" : "bg-white/[0.035] text-[#64738d]"}`}>
                      <BarChart3 size={15} />
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-xs font-semibold">{study.name}</span>
                      <span className="mt-1 block truncate text-[10px] text-[#63718a]">{study.dataset} · {study.rows.length} trials</span>
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onRemove(study);
                    }}
                    className="focus-ring absolute right-2.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-lg text-[#58667f] opacity-0 transition hover:bg-rose-400/10 hover:text-rose-300 group-hover:opacity-100 focus:opacity-100"
                    aria-label={`Remove ${study.name}`}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              );
            })}
          </div>
        </nav>

        <div className="border-t border-white/[0.07] p-3">
          <div className="mb-2">
            <button
              type="button"
              onClick={() => setCatalogOpen((open) => !open)}
              disabled={availableStudies.length === 0}
              className="focus-ring flex w-full items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.025] px-3 py-2.5 text-left text-xs font-semibold text-[#aeb8ca] transition hover:bg-white/[0.05] hover:text-white disabled:cursor-not-allowed disabled:opacity-45"
              aria-expanded={catalogOpen}
            >
              <FolderOpen size={15} className="text-[#9b8cff]" />
              Repository studies
              <span className="ml-auto rounded-full bg-white/[0.06] px-2 py-0.5 text-[9px] text-[#8491a7]">{availableStudies.length}</span>
            </button>
            {catalogOpen && availableStudies.length > 0 && (
              <div className="mt-2 max-h-52 space-y-1 overflow-y-auto rounded-xl border border-white/[0.08] bg-black/15 p-1.5">
                {availableStudies.map((study) => (
                  <button
                    key={study.id}
                    type="button"
                    onClick={() => onAddStudy(study)}
                    className="focus-ring flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition hover:bg-white/[0.05]"
                  >
                    <Plus size={13} className="shrink-0 text-[#b8e56e]" />
                    <span className="min-w-0">
                      <span className="block truncate text-[11px] font-semibold text-[#c7d0df]">{study.name}</span>
                      <span className="mt-0.5 block truncate text-[9px] text-[#66748c]">{study.dataset} · {study.rows.length} trials</span>
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <label className="focus-within:ring-2 focus-within:ring-[#9b8cff] mb-2 flex cursor-pointer items-center gap-3 rounded-xl border border-[#9b8cff]/20 bg-[#9b8cff]/[0.07] px-3 py-2.5 transition hover:border-[#9b8cff]/35 hover:bg-[#9b8cff]/[0.11]">
            <FolderOpen size={15} className="shrink-0 text-[#b5aaff]" />
            <span className="min-w-0">
              <span className="block text-xs font-semibold text-[#d5d0ff]">Add log folder</span>
              <span className="mt-0.5 block text-[9px] text-[#776eaa]">Find every nested trials.csv</span>
            </span>
            <input
              type="file"
              multiple
              className="sr-only"
              {...DIRECTORY_INPUT_ATTRIBUTES}
              onChange={(event) => {
                if (event.target.files) onFolder(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
          <label
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`focus-within:ring-2 focus-within:ring-[#9b8cff] flex cursor-pointer items-center gap-3 rounded-2xl border border-dashed p-3.5 transition ${
              dragging ? "border-[#9b8cff]/70 bg-[#9b8cff]/10" : "border-white/[0.11] bg-white/[0.025] hover:border-white/[0.2] hover:bg-white/[0.04]"
            }`}
          >
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#4fd5e7]/10 text-[#65dce9]">
              <UploadCloud size={17} />
            </span>
            <span className="min-w-0">
              <span className="block text-xs font-semibold text-[#c6cede]">Add CSV trials</span>
              <span className="mt-1 block text-[10px] text-[#66748d]">Drop or choose multiple files</span>
            </span>
            <input
              type="file"
              accept=".csv,text/csv"
              multiple
              className="sr-only"
              onChange={(event) => {
                if (event.target.files) onFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
        </div>
      </aside>
    </>
  );
}

function Overview({ studies, onSelect, onCompare }: { studies: Study[]; onSelect: (study: Study) => void; onCompare: () => void }) {
  const records = studies.map((study) => ({ study, stats: getStudyStats(study) }));
  const totalTrials = studies.reduce((sum, study) => sum + study.rows.length, 0);
  const totalComplete = records.reduce((sum, record) => sum + record.stats.complete, 0);
  const bestRecord = records.reduce<(typeof records)[number] | null>((best, record) => {
    if (record.stats.bestMetric == null) return best;
    return !best || best.stats.bestMetric == null || record.stats.bestMetric > best.stats.bestMetric ? record : best;
  }, null);
  const datasets = new Set(studies.map((study) => study.dataset)).size;

  return (
    <div className="fade-up space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-[#78869e]">
            <Sparkles size={12} className="text-[#b8e56e]" /> Experiment portfolio
          </div>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">All studies, one signal.</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[#8794aa]">Compare the best observed result across every loaded study, then open one to inspect its search trajectory and trials.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
          <div className="flex items-center gap-2 rounded-full border border-white/[0.08] bg-white/[0.035] px-3 py-1.5 text-[11px] text-[#95a1b5]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#b8e56e] shadow-[0_0_10px_rgba(184,229,110,.9)]" />
            {studies.length} studies loaded
          </div>
          <button
            type="button"
            onClick={onCompare}
            className="focus-ring inline-flex h-9 items-center gap-2 rounded-xl bg-[#9b8cff] px-3.5 text-xs font-semibold text-white shadow-[0_10px_30px_rgba(118,91,255,.22)] transition hover:bg-[#aa9eff]"
          >
            <GitCompareArrows size={15} /> Compare runs
          </button>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Total trials" value={totalTrials.toLocaleString()} context={`Across ${studies.length} studies`} icon={<Rows3 size={17} />} accent="violet" />
        <KpiCard label="Completion" value={totalTrials ? `${((totalComplete / totalTrials) * 100).toFixed(1)}%` : "—"} context={`${totalComplete} completed trials`} icon={<CheckCircle2 size={17} />} accent="lime" />
        <KpiCard label="Best metric" value={formatMetric(bestRecord?.stats.bestMetric)} context={bestRecord?.study.name ?? "No metric available"} icon={<Trophy size={17} />} accent="amber" />
        <KpiCard label="Datasets" value={String(datasets)} context="Detected from file names" icon={<Database size={17} />} accent="cyan" />
      </div>

      <ChartBoard
        storageKey="trial-atlas:charts:overview"
        className="grid gap-6"
        items={[
          {
            id: "study-leaderboard",
            title: "Study leaderboard",
            description: "Best observed metric in each CSV. Select a bar to open that study.",
            exportName: "study-leaderboard",
            content: <StudyComparisonChart studies={studies} onSelect={onSelect} />,
          },
        ]}
      />

      <section className="surface overflow-hidden">
        <header className="border-b border-white/[0.07] px-5 py-4 sm:px-6">
          <h2 className="text-[15px] font-semibold text-white">Study registry</h2>
          <p className="mt-1.5 text-xs text-[#8290a7]">Detected schema and best result for every loaded experiment.</p>
        </header>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-xs">
            <thead className="border-b border-white/[0.07] bg-black/10 text-[10px] uppercase tracking-[0.11em] text-[#6f7d95]">
              <tr>
                <th className="px-6 py-3.5 font-semibold">Study</th>
                <th className="px-4 py-3.5 font-semibold">Trials</th>
                <th className="px-4 py-3.5 font-semibold">Best metric</th>
                <th className="px-4 py-3.5 font-semibold">Best trial</th>
                <th className="px-4 py-3.5 font-semibold">Metric column</th>
                <th className="px-6 py-3.5 text-right font-semibold">Open</th>
              </tr>
            </thead>
            <tbody>
              {records
                .sort((a, b) => (b.stats.bestMetric ?? Number.NEGATIVE_INFINITY) - (a.stats.bestMetric ?? Number.NEGATIVE_INFINITY))
                .map(({ study, stats }) => (
                  <tr key={study.id} className="border-b border-white/[0.055] text-[#aeb8ca] last:border-0 hover:bg-white/[0.025]">
                    <td className="px-6 py-4">
                      <button type="button" onClick={() => onSelect(study)} className="focus-ring rounded text-left">
                        <span className="block font-semibold text-white">{study.name}</span>
                        <span className="mt-1 block text-[10px] text-[#6d7a92]">{study.dataset}</span>
                      </button>
                    </td>
                    <td className="px-4 py-4 tabular-nums">{study.rows.length}</td>
                    <td className="px-4 py-4 font-semibold tabular-nums text-[#c9c0ff]">{formatMetric(stats.bestMetric)}</td>
                    <td className="px-4 py-4 tabular-nums">#{stats.bestTrial ?? "—"}</td>
                    <td className="max-w-64 px-4 py-4"><span className="block truncate" title={study.config.metric}>{columnLabel(study.config.metric)}</span></td>
                    <td className="px-6 py-4 text-right">
                      <button type="button" onClick={() => onSelect(study)} className="focus-ring inline-flex items-center gap-1 rounded-lg px-2 py-1.5 font-semibold text-[#a99dff] hover:bg-[#9b8cff]/10">
                        Explore <ChevronRight size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function StudyDashboard({ study, onUpdate }: { study: Study; onUpdate: (study: Study) => void }) {
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const [selectedRow, setSelectedRow] = useState<TrialRow | null>(null);
  const [curveRows, setCurveRows] = useState<TrialRow[]>([]);
  const [trialSummaryOpen, setTrialSummaryOpen] = useState(false);
  const [selectionStorageReady, setSelectionStorageReady] = useState(false);
  const stats = useMemo(() => getStudyStats(study), [study]);
  const selectionStorageKey = `trial-atlas:selected-trials:${study.id}`;

  useEffect(() => {
    setSelectionStorageReady(false);
    let stored: PersistedTrialSelection | null = null;
    try {
      const raw = window.localStorage.getItem(selectionStorageKey);
      const parsed = raw ? (JSON.parse(raw) as Partial<PersistedTrialSelection>) : null;
      if (parsed?.version === 2 && Array.isArray(parsed.curveIndices)) {
        stored = {
          version: 2,
          curveIndices: parsed.curveIndices.filter(
            (index): index is number => Number.isInteger(index) && index >= 0 && index < study.rows.length,
          ),
          activeIndex:
            parsed.activeIndex == null
              ? null
              : Number.isInteger(parsed.activeIndex) && parsed.activeIndex >= 0 && parsed.activeIndex < study.rows.length
                ? parsed.activeIndex
                : null,
          summaryOpen: parsed.summaryOpen === true,
        };
      }
    } catch {
      stored = null;
    }

    if (stored) {
      setCurveRows(stored.curveIndices.map((index) => study.rows[index]));
      setSelectedRow(stored.activeIndex == null ? null : study.rows[stored.activeIndex]);
      setTrialSummaryOpen(stored.activeIndex != null && stored.summaryOpen);
    } else {
      setSelectedRow(null);
      setCurveRows([]);
      setTrialSummaryOpen(false);
    }
    setSelectionStorageReady(true);
  }, [selectionStorageKey]);

  useEffect(() => {
    if (!selectionStorageReady) return;
    const curveIndices = curveRows
      .map((row) => study.rows.indexOf(row))
      .filter((index) => index >= 0);
    const activeIndex = selectedRow ? study.rows.indexOf(selectedRow) : -1;
    const payload: PersistedTrialSelection = {
      version: 2,
      curveIndices,
      activeIndex: activeIndex >= 0 ? activeIndex : null,
      summaryOpen: activeIndex >= 0 && trialSummaryOpen,
    };
    try {
      window.localStorage.setItem(selectionStorageKey, JSON.stringify(payload));
    } catch {
      // Trial selection remains available for the current session if storage is blocked.
    }
  }, [curveRows, selectedRow, selectionStorageKey, selectionStorageReady, study.rows, trialSummaryOpen]);

  const activateTrial = (row: TrialRow | null) => {
    setSelectedRow(row);
    setTrialSummaryOpen(row !== null);
  };

  const toggleCurveTrial = (row: TrialRow) => {
    setCurveRows((current) =>
      current.includes(row) ? current.filter((candidate) => candidate !== row) : [...current, row],
    );
  };

  const selectChartTrial = (row: TrialRow) => {
    setSelectedRow(row);
    setTrialSummaryOpen(true);
    toggleCurveTrial(row);
  };

  const viewTrainingCurves = () => {
    setTrialSummaryOpen(false);
    document.getElementById("training-curves")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <>
      {selectedRow && trialSummaryOpen && (
        <SelectedTrialPopover
          study={study}
          row={selectedRow}
          includedInCurves={curveRows.includes(selectedRow)}
          onClose={() => setTrialSummaryOpen(false)}
          onToggleCurve={() => toggleCurveTrial(selectedRow)}
          onViewCurves={viewTrainingCurves}
        />
      )}
      <div className="fade-up space-y-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-[#9b8cff]/20 bg-[#9b8cff]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#b3a8ff]">{study.dataset}</span>
              <span className="rounded-full border border-white/[0.08] bg-white/[0.035] px-2.5 py-1 text-[10px] font-medium text-[#8996ab]">
                {study.source === "bundled" ? "Repository data" : study.fileName.includes("/") ? "Log folder" : "Uploaded"}
              </span>
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">{study.name}</h1>
            <p className="mt-2 max-w-3xl truncate text-xs text-[#6f7d95]" title={study.fileName}>{study.fileName}</p>
          </div>
          <button
            type="button"
            onClick={() => setConfigurationOpen((open) => !open)}
            className={`focus-ring flex h-10 items-center justify-center gap-2 self-start rounded-xl border px-4 text-xs font-semibold transition xl:self-auto ${
              configurationOpen ? "border-[#9b8cff]/35 bg-[#9b8cff]/12 text-[#c4bcff]" : "border-white/[0.09] bg-white/[0.04] text-[#c3ccdc] hover:bg-white/[0.08] hover:text-white"
            }`}
            aria-expanded={configurationOpen}
          >
            <Settings2 size={15} /> Configure columns
          </button>
        </div>

        {configurationOpen && <ConfigurationPanel study={study} onChange={(config) => onUpdate({ ...study, config })} />}

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard label="Best metric" value={formatMetric(stats.bestMetric)} context={columnLabel(study.config.metric)} icon={<Trophy size={17} />} accent="violet" />
        <KpiCard label="Best trial" value={stats.bestTrial != null ? `#${stats.bestTrial}` : "—"} context={stats.bestRow && study.config.epoch ? `Stopped at epoch ${stats.bestRow[study.config.epoch] ?? "—"}` : "Highest observed value"} icon={<Gauge size={17} />} accent="lime" />
        <KpiCard label="Completed" value={`${stats.complete} / ${study.rows.length}`} context={stats.failed ? `${stats.failed} failed` : "No failed trials"} icon={<CheckCircle2 size={17} />} accent="cyan" />
        <KpiCard label="Median runtime" value={formatDuration(stats.medianDuration)} context={columnLabel(study.config.duration)} icon={<Timer size={17} />} accent="amber" />
        </div>

      <ChartBoard
        storageKey={`trial-atlas:charts:${study.id}:study`}
        items={[
          {
            id: "optimization-trajectory",
            title: "Optimization trajectory",
            description: "Observed metric and cumulative best across the trial sequence. Select points to inspect and compare trials.",
            exportName: `${study.name}-optimization-trajectory`,
            className: "xl:col-span-2",
            content: <DevelopmentChart study={study} selectedRow={selectedRow} selectedRows={curveRows} onSelect={selectChartTrial} />,
          },
          {
            id: "epoch-behavior",
            title: "Epoch behavior",
            description: "Last epoch and model-selected epoch for each trial.",
            exportName: `${study.name}-epochs`,
            content: <EpochChart study={study} selectedRow={selectedRow} selectedRows={curveRows} onSelect={selectChartTrial} />,
          },
          {
            id: "learning-rate-response",
            title: "Learning-rate response",
            description: "Metric response across the log-scaled learning-rate search space.",
            exportName: `${study.name}-learning-rate`,
            content: <LearningRateChart study={study} selectedRow={selectedRow} selectedRows={curveRows} onSelect={selectChartTrial} />,
          },
          {
            id: "metric-distribution",
            title: "Metric distribution",
            description: "Distribution of the selected metric across all plottable trials.",
            exportName: `${study.name}-metric-distribution`,
            className: "xl:col-span-2",
            content: <DistributionChart study={study} />,
          },
        ]}
      />

      <CorrelationExplorer study={study} selectedRow={selectedRow} onSelect={activateTrial} />

      <TrainingCurves study={study} rows={curveRows} onRemove={toggleCurveTrial} />

      <TrialsTable
        study={study}
        selectedRow={selectedRow}
        curveRows={curveRows}
        onSelect={activateTrial}
        onToggleCurve={toggleCurveTrial}
      />
      </div>
    </>
  );
}

function EmptyWorkspace({
  repositoryStudies,
  onAddStudy,
  onFiles,
  onFolder,
}: {
  repositoryStudies: Study[];
  onAddStudy: (study: Study) => void;
  onFiles: (files: FileList) => void;
  onFolder: (files: FileList) => void;
}) {
  return (
    <div className="flex min-h-[66vh] items-center justify-center">
      <div className="surface w-full max-w-2xl px-7 py-10 text-center sm:px-12">
        <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#9b8cff]/10 text-[#b6acff] ring-1 ring-[#9b8cff]/20">
          <FolderOpen size={25} />
        </span>
        <h1 className="mt-5 text-2xl font-semibold tracking-[-0.03em] text-white">Choose your experiments</h1>
        <p className="mt-2 text-sm leading-6 text-[#8794aa]">Select a log folder to find every nested trials.csv study at once, or start with repository data and individual CSVs.</p>
        {repositoryStudies.length > 0 && (
          <div className="mt-6 grid gap-2 text-left sm:grid-cols-2">
            {repositoryStudies.map((study) => (
              <button
                key={study.id}
                type="button"
                onClick={() => onAddStudy(study)}
                className="focus-ring flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.025] px-3.5 py-3 transition hover:border-[#9b8cff]/25 hover:bg-[#9b8cff]/[0.07]"
              >
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#9b8cff]/10 text-[#b7adff]"><Plus size={14} /></span>
                <span className="min-w-0">
                  <span className="block truncate text-xs font-semibold text-[#dce3ee]">{study.name}</span>
                  <span className="mt-1 block truncate text-[10px] text-[#69778f]">{study.dataset} · {study.rows.length} trials</span>
                </span>
              </button>
            ))}
          </div>
        )}
        <div className="mt-6 flex flex-wrap items-center justify-center gap-2">
          <label className="focus-within:ring-2 focus-within:ring-[#9b8cff] inline-flex h-11 cursor-pointer items-center gap-2 rounded-xl bg-[#9b8cff] px-5 text-xs font-semibold text-white shadow-[0_12px_34px_rgba(118,91,255,.25)] transition hover:bg-[#aa9eff]">
            <FolderOpen size={16} /> Choose log folder
            <input
              type="file"
              multiple
              className="sr-only"
              {...DIRECTORY_INPUT_ATTRIBUTES}
              onChange={(event) => {
                if (event.target.files) onFolder(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
          <label className="focus-within:ring-2 focus-within:ring-[#9b8cff] inline-flex h-11 cursor-pointer items-center gap-2 rounded-xl border border-white/[0.1] bg-white/[0.04] px-5 text-xs font-semibold text-[#cbd3e1] transition hover:bg-white/[0.08] hover:text-white">
            <FilePlus2 size={16} /> Choose CSV files
            <input
              type="file"
              accept=".csv,text/csv"
              multiple
              className="sr-only"
              onChange={(event) => {
                if (event.target.files) onFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
        </div>
      </div>
    </div>
  );
}

export function TrialVisualizer({
  initialFiles,
  initialTestResultsFile,
}: TrialVisualizerProps) {
  const repositoryStudies = useMemo(
    () =>
      initialFiles.flatMap((file) => {
        try {
          return [createStudy(file, "bundled")];
        } catch {
          return [];
        }
    }),
    [initialFiles],
  );
  const [studies, setStudies] = useState<Study[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<View>("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [workspaceStorageReady, setWorkspaceStorageReady] = useState(false);
  const storageWarningShown = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const selectedStudy = studies.find((study) => study.id === selectedId) ?? null;

  useEffect(() => {
    setWorkspaceStorageReady(false);
    try {
      const raw = window.localStorage.getItem(WORKSPACE_SELECTION_KEY);
      const stored = raw ? (JSON.parse(raw) as Partial<PersistedWorkspaceSession>) : null;
      if (stored?.version === 2 && Array.isArray(stored.studies)) {
        const repositoryById = new Map(repositoryStudies.map((study) => [study.id, study]));
        const restoredById = new Map<string, Study>();
        stored.studies.filter(isStoredStudy).forEach((storedStudy) => {
          if (storedStudy.source === "bundled") {
            const current = repositoryById.get(storedStudy.id);
            if (current) restoredById.set(current.id, restoreBundledStudy(current, storedStudy));
          } else {
            restoredById.set(storedStudy.id, storedStudy);
          }
        });
        const restored = [...restoredById.values()];
        const restoredSelectedId =
          typeof stored.selectedId === "string" && restored.some((study) => study.id === stored.selectedId)
            ? stored.selectedId
            : null;
        setStudies(restored);
        setSelectedId(restoredSelectedId);
        setView(
          stored.view === "parameters"
            ? "parameters"
            : stored.view === "archive"
              ? "archive"
              : stored.view === "study" && restoredSelectedId
                ? "study"
                : stored.view === "compare"
                  ? "compare"
                  : "overview",
        );
      } else {
        setStudies([]);
        setSelectedId(null);
        setView("overview");
      }
    } catch {
      setStudies([]);
      setSelectedId(null);
      setView("overview");
    }
    // `/?view=parameters|archive|compare` opens that workspace directly (the todo page links here).
    const linked = new URLSearchParams(window.location.search).get("view");
    if (linked === "parameters" || linked === "archive" || linked === "compare") setView(linked);
    setWorkspaceStorageReady(true);
  }, [repositoryStudies]);

  useEffect(() => {
    if (!workspaceStorageReady) return;
    const payload: PersistedWorkspaceSession = { version: 2, studies, selectedId, view };
    try {
      window.localStorage.setItem(WORKSPACE_SELECTION_KEY, JSON.stringify(payload));
    } catch {
      if (!storageWarningShown.current) {
        storageWarningShown.current = true;
        setNotice("The workspace is too large for browser storage, so this session may not survive a reload.");
      }
    }
  }, [selectedId, studies, view, workspaceStorageReady]);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 4_000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const selectStudy = (study: Study) => {
    setSelectedId(study.id);
    setView("study");
    setMobileOpen(false);
  };

  const addRepositoryStudy = (study: Study) => {
    setStudies((current) => current.some((candidate) => candidate.id === study.id) ? current : [...current, study]);
    selectStudy(study);
  };

  const handleFiles = async (fileList: FileList | File[], mode: FileImportMode = "csv-files") => {
    const selectedFiles = Array.from(fileList);
    const csvFiles = selectedFiles
      .filter((file) => mode === "log-folder" ? isTrialsCsv(file) : file.name.toLowerCase().endsWith(".csv"))
      .sort((first, second) => fileImportPath(first).localeCompare(fileImportPath(second)));

    if (csvFiles.length === 0) {
      setNotice(mode === "log-folder" ? "No trials.csv files were found in that folder or its subfolders." : "Choose at least one .csv file.");
      return;
    }

    const results = await Promise.all(csvFiles.map(async (file) => {
      const importPath = mode === "log-folder" ? fileImportPath(file) : file.name;
      try {
        return { study: createStudy({ name: importPath, text: await file.text() }, "uploaded"), failure: null };
      } catch (error) {
        return { study: null, failure: `${importPath}: ${error instanceof Error ? error.message : "Could not parse CSV"}` };
      }
    }));
    const loaded = results.flatMap((result) => result.study ? [result.study] : []);
    const failures = results.flatMap((result) => result.failure ? [result.failure] : []);

    if (loaded.length > 0) {
      setStudies((current) => {
        const byId = new Map(current.map((study) => [study.id, study]));
        loaded.forEach((study) => byId.set(study.id, study));
        return [...byId.values()];
      });
      if (mode === "log-folder") {
        setSelectedId(null);
        setView("overview");
        setMobileOpen(false);
      } else {
        selectStudy(loaded[0]);
      }
      const folderName = mode === "log-folder" ? fileImportPath(csvFiles[0]).split("/")[0] : null;
      setNotice(`${loaded.length} ${loaded.length === 1 ? "study" : "studies"} loaded${folderName ? ` from ${folderName}` : ""}${failures.length ? ` · ${failures.length} skipped` : ""}.`);
    } else {
      setNotice(failures[0] ?? "No CSV files could be loaded.");
    }
  };

  const removeStudy = (study: Study) => {
    const remaining = studies.filter((candidate) => candidate.id !== study.id);
    setStudies(remaining);
    if (selectedId === study.id) {
      setSelectedId(null);
      setView("overview");
    }
  };

  const updateStudy = (updated: Study) => {
    setStudies((current) => current.map((study) => (study.id === updated.id ? updated : study)));
  };

  return (
    <div className="min-h-screen">
      <Sidebar
        studies={studies}
        repositoryStudies={repositoryStudies}
        selectedId={selectedId}
        view={view}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        onOverview={() => {
          setView("overview");
          setMobileOpen(false);
        }}
        onCompare={() => {
          setView("compare");
          setMobileOpen(false);
        }}
        onParameters={() => {
          setView("parameters");
          setMobileOpen(false);
        }}
        onArchive={() => {
          setView("archive");
          setMobileOpen(false);
        }}
        onSelect={selectStudy}
        onAddStudy={addRepositoryStudy}
        onRemove={removeStudy}
        onFiles={handleFiles}
        onFolder={(files) => void handleFiles(files, "log-folder")}
      />

      <div className="min-w-0 lg:pl-[286px]">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-white/[0.07] bg-[#080d18]/80 px-4 backdrop-blur-xl sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <button type="button" onClick={() => setMobileOpen(true)} className="focus-ring flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/[0.08] text-[#a4afc1] lg:hidden" aria-label="Open navigation">
              <Menu size={18} />
            </button>
            <div className="min-w-0">
              <p className="truncate text-xs font-semibold text-[#dce2ee]">
                {view === "overview"
                  ? "Portfolio overview"
                  : view === "compare"
                    ? "Run comparison"
                    : view === "parameters"
                      ? "Hyperparameter behavior"
                      : view === "archive"
                        ? "Sorted HPO archive"
                        : selectedStudy?.name ?? "Trial workspace"}
              </p>
              <p className="mt-0.5 hidden truncate text-[10px] text-[#64718a] sm:block">
                {view === "compare"
                  ? "Shared metrics · trajectories · runtime trade-offs"
                  : view === "parameters"
                    ? "fANOVA importance · sampled responses · categorical effects"
                    : view === "archive"
                      ? "Phase · dataset · method · loss · campaign — filtered and compared"
                      : "Interactive trial diagrams · local browser session"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 rounded-xl border border-white/[0.07] bg-white/[0.025] px-3 py-2 text-[10px] text-[#7d8aa1] sm:flex">
              <span className="h-1.5 w-1.5 rounded-full bg-[#b8e56e]" /> {studies.reduce((sum, study) => sum + study.rows.length, 0)} trials ready
            </div>
            <button
              type="button"
              onClick={() => folderInputRef.current?.click()}
              className="focus-ring flex h-9 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3 text-xs font-semibold text-[#c7cfdd] transition hover:bg-white/[0.08] hover:text-white"
              aria-label="Import a log folder"
              title="Recursively import every trials.csv file"
            >
              <FolderOpen size={15} /> <span className="hidden md:inline">Log folder</span>
            </button>
            <input
              ref={folderInputRef}
              type="file"
              multiple
              className="sr-only"
              {...DIRECTORY_INPUT_ATTRIBUTES}
              onChange={(event) => {
                if (event.target.files) void handleFiles(event.target.files, "log-folder");
                event.target.value = "";
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="focus-ring flex h-9 items-center gap-2 rounded-xl bg-[#9b8cff] px-3.5 text-xs font-semibold text-white shadow-[0_10px_30px_rgba(118,91,255,.22)] transition hover:bg-[#aa9eff]"
            >
              <FilePlus2 size={15} /> <span className="hidden sm:inline">Import CSV</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              multiple
              className="sr-only"
              onChange={(event) => {
                if (event.target.files) void handleFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </div>
        </header>

        <main className="mx-auto w-full max-w-[1540px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          {!workspaceStorageReady ? (
            <div className="flex min-h-[66vh] items-center justify-center">
              <div className="text-center">
                <span className="mx-auto block h-6 w-6 animate-spin rounded-full border-2 border-[#9b8cff]/25 border-t-[#9b8cff]" />
                <p className="mt-3 text-xs text-[#7f8ca2]">Restoring workspace…</p>
              </div>
            </div>
          ) : view === "parameters" ? (
            <OptunaExplorer />
          ) : view === "archive" ? (
            <ArchiveExplorer />
          ) : studies.length === 0 ? (
            <EmptyWorkspace
              repositoryStudies={repositoryStudies}
              onAddStudy={addRepositoryStudy}
              onFiles={handleFiles}
              onFolder={(files) => void handleFiles(files, "log-folder")}
            />
          ) : view === "overview" ? (
            <Overview
              studies={studies}
              onSelect={selectStudy}
              onCompare={() => setView("compare")}
            />
          ) : view === "compare" ? (
            <ComparisonWorkspace
              studies={studies}
              initialTestResultsFile={initialTestResultsFile}
              onOpenStudy={selectStudy}
            />
          ) : selectedStudy ? (
            <StudyDashboard key={selectedStudy.id} study={selectedStudy} onUpdate={updateStudy} />
          ) : null}
        </main>
      </div>

      {notice && (
        <div role="status" className="fixed bottom-5 right-5 z-[70] flex max-w-sm items-center gap-3 rounded-2xl border border-white/[0.1] bg-[#151f34] px-4 py-3 text-xs text-[#d7deea] shadow-2xl">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[#4fd5e7]/10 text-[#64dce9]"><CheckCircle2 size={15} /></span>
          <span className="leading-5">{notice}</span>
          <button type="button" onClick={() => setNotice(null)} className="focus-ring ml-1 rounded text-[#7b899f] hover:text-white" aria-label="Dismiss notification"><X size={14} /></button>
        </div>
      )}
    </div>
  );
}
