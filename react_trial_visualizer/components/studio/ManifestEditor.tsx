"use client";

import { Copy, FolderTree, Plus, Trash2, Wand2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { lintManifest } from "@/lib/studio/lint";
import type { Issue, StudioSchema } from "@/lib/studio/schema";

import { ArgsBuilder } from "./ArgsBuilder";
import { ConfigWorkbench } from "./ConfigWorkbench";
import { useStudio } from "./StudioContext";
import {
  Button,
  Card,
  ChipList,
  Collapsible,
  Field,
  IssueBadge,
  KeyValueEditor,
  NumberInput,
  Select,
  TextInput,
  Toggle,
  parseScalar,
} from "./ui";

type Document = Record<string, unknown>;
type Run = Record<string, unknown>;

const LOSS_ARM_PRESETS: { name: string; entry: string }[] = [
  { name: "nt", entry: "NTXentLoss:no_miner" },
  { name: "ms", entry: "MultiSimilarityLoss:MultiSimilarityMiner" },
  { name: "cir", entry: "CircleLoss:no_miner" },
  { name: "arc", entry: "ArcFaceLoss:no_miner" },
  { name: "prx", entry: "ProxyAnchorLoss:no_miner" },
];

function emptyDocument(): Document {
  return {
    version: 1,
    gpus: ["0"],
    max_concurrent_per_gpu: 1,
    default_gpu_memory_mib: 2048,
    reserve_gpu_memory_mib: 1024,
    poll_interval_seconds: 2,
    launch_delay_seconds: 2,
    save_dir_root: "",
    scheduler_log_dir: "logs/run_scheduler",
    runs: [],
  };
}

function RunCard({
  run,
  index,
  document,
  schema,
  issues,
  onChange,
  onRemove,
  onDuplicate,
}: {
  run: Run;
  index: number;
  document: Document;
  schema: StudioSchema;
  issues: Issue[];
  onChange: (next: Run) => void;
  onRemove: () => void;
  onDuplicate: () => void;
}) {
  const { filesOfKind } = useStudio();
  const name = typeof run.name === "string" ? run.name : `runs[${index}]`;
  const enabled = run.enabled !== false;
  const replaysStudy = Boolean(run.study_dir);

  const experimentOptions = filesOfKind("experiment").map((file) => ({ value: file.path }));
  const hparamOptions = filesOfKind("hparam").map((file) => ({ value: file.path }));

  const set = (key: string, value: unknown) => {
    const next = { ...run };
    if (value === null || value === undefined || value === "") delete next[key];
    else next[key] = value;
    onChange(next);
  };

  const effectiveSaveDir =
    typeof run.save_dir === "string" && run.save_dir
      ? run.save_dir
      : typeof document.save_dir_root === "string" && document.save_dir_root
        ? `${document.save_dir_root}/${name}`
        : null;

  const runIssues = issues.filter((issue) => issue.field === name || issue.field === `runs[${index}]`);

  return (
    <Collapsible
      title={
        <span className={enabled ? "" : "line-through opacity-60"}>
          {name}
          {!enabled && <span className="ml-2 text-[0.7rem] font-normal">disabled</span>}
        </span>
      }
      subtitle={
        replaysStudy
          ? `replays ${String(run.study_dir)}`
          : `${String(run.experiment_config ?? "no experiment config")} · ${String(run.hparam_config ?? "hparam from experiment config")}`
      }
      badge={<IssueBadge issues={runIssues} />}
      defaultOpen={false}
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Name" help="Becomes the run's log name and, with save_dir_root, its output folder.">
          <TextInput
            value={typeof run.name === "string" ? run.name : ""}
            onChange={(value) => set("name", value)}
            mono
          />
        </Field>
        <Field label="Experiment config">
          <Select
            value={(run.experiment_config as string) ?? null}
            onChange={(value) => set("experiment_config", value)}
            options={experimentOptions}
            placeholder="none"
          />
        </Field>
        <Field
          label="Search config"
          help={replaysStudy ? "A study replay reuses the saved search config." : "Overrides the experiment config's own hparam_config."}
        >
          <Select
            value={(run.hparam_config as string) ?? null}
            onChange={(value) => set("hparam_config", value)}
            options={hparamOptions}
            placeholder={replaysStudy ? "not allowed with study_dir" : "from experiment config"}
            disabled={replaysStudy}
          />
        </Field>
        <Field label="Study dir" help="Set this to replay an existing study directory instead of searching.">
          <TextInput
            value={(run.study_dir as string) ?? ""}
            onChange={(value) => set("study_dir", value)}
            mono
          />
        </Field>
        <Field
          label="Save dir"
          help={effectiveSaveDir ? `Effective: ${effectiveSaveDir}` : "Falls back to the experiment config."}
        >
          <TextInput
            value={(run.save_dir as string) ?? ""}
            onChange={(value) => set("save_dir", value)}
            placeholder={
              typeof document.save_dir_root === "string" && document.save_dir_root
                ? `${document.save_dir_root}/${name}`
                : "unset"
            }
            mono
          />
        </Field>
        <Field label="GPU" help="Pin this run to one index or UUID; blank lets the scheduler place it.">
          <TextInput value={(run.gpu as string) ?? ""} onChange={(value) => set("gpu", value)} mono />
        </Field>
        <Field label="GPU memory (MiB)" help="Reserved for admission control, not a hard limit.">
          <NumberInput
            value={(run.gpu_memory_mib as number) ?? null}
            onChange={(value) => set("gpu_memory_mib", value)}
            placeholder={String(document.default_gpu_memory_mib ?? 2048)}
          />
        </Field>
        <Field label="Slots">
          <NumberInput value={(run.slots as number) ?? null} onChange={(value) => set("slots", value)} placeholder="1" />
        </Field>
        <Field label="Priority" help="Higher priority runs are admitted first.">
          <NumberInput
            value={(run.priority as number) ?? null}
            onChange={(value) => set("priority", value)}
            placeholder="0"
          />
        </Field>
      </div>

      <div className="mt-4">
        <Toggle checked={enabled} onChange={(value) => set("enabled", value ? null : false)} label="Enabled" />
      </div>

      <div className="mt-5">
        <p className="studio-label mb-2">Command-line overrides</p>
        <ArgsBuilder
          tokens={Array.isArray(run.args) ? (run.args as string[]) : []}
          onChange={(tokens) => set("args", tokens.length === 0 ? null : tokens)}
          schema={schema}
          forbidden={[
            ...schema.manifest.schedulerOwnedOptions,
            ...(replaysStudy ? schema.manifest.studyReplayOwnedOptions : []),
          ]}
        />
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <div>
          <p className="studio-label mb-2">Environment</p>
          <KeyValueEditor
            entries={(run.env ?? {}) as Record<string, unknown>}
            onChange={(entries) => set("env", Object.keys(entries).length === 0 ? null : entries)}
            keyPlaceholder="VAR"
          />
        </div>
        {replaysStudy && (
          <div>
            <p className="studio-label mb-2">Study replay</p>
            <StudyReplayFields
              value={(run.study_replay ?? {}) as Record<string, unknown>}
              onChange={(next) => set("study_replay", Object.keys(next).length === 0 ? null : next)}
              schema={schema}
            />
          </div>
        )}
      </div>

      <div className="mt-5 flex gap-2">
        <Button onClick={onDuplicate}>
          <Copy className="size-3.5" />
          Duplicate
        </Button>
        <Button variant="danger" onClick={onRemove}>
          <Trash2 className="size-3.5" />
          Remove
        </Button>
      </div>
    </Collapsible>
  );
}

function StudyReplayFields({
  value,
  onChange,
  schema,
}: {
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  schema: StudioSchema;
}) {
  const set = (key: string, next: unknown) => {
    const updated = { ...value };
    if (next === null || next === undefined || next === "" || (Array.isArray(next) && next.length === 0)) {
      delete updated[key];
    } else {
      updated[key] = next;
    }
    onChange(updated);
  };

  return (
    <div className="space-y-3">
      <Field label="Study dir mode">
        <Select
          value={(value.study_dir_mode as string) ?? null}
          onChange={(next) => set("study_dir_mode", next)}
          options={schema.manifest.studyDirModes.map((mode) => ({ value: mode }))}
          placeholder={schema.manifest.defaultStudyDirMode}
        />
      </Field>
      <Field label="Final test top N">
        <NumberInput
          value={(value.final_test_top_n as number) ?? null}
          onChange={(next) => set("final_test_top_n", next)}
        />
      </Field>
      <Field label="Trial numbers">
        <ChipList
          values={Array.isArray(value.final_test_trial_numbers) ? (value.final_test_trial_numbers as unknown[]) : []}
          onChange={(next) => set("final_test_trial_numbers", next)}
          parse={(text) => Math.trunc(Number(text))}
          placeholder="Trial number…"
        />
      </Field>
      <Field label="Comparison seeds">
        <ChipList
          values={Array.isArray(value.comparison_seeds) ? (value.comparison_seeds as unknown[]) : []}
          onChange={(next) => set("comparison_seeds", next)}
          parse={(text) => Math.trunc(Number(text))}
          placeholder="Seed…"
        />
      </Field>
      <Field label="Seed targets">
        <ChipList
          values={Array.isArray(value.comparison_seed_targets) ? (value.comparison_seed_targets as unknown[]) : []}
          onChange={(next) => set("comparison_seed_targets", next)}
          suggestions={schema.manifest.comparisonSeedTargets}
          placeholder="Target…"
        />
      </Field>
      <Field label="Final-test visualization">
        <Select
          value={(value.final_test_visualization as string) ?? null}
          onChange={(next) => set("final_test_visualization", next)}
          options={schema.manifest.finalTestVisualizationModes.map((mode) => ({ value: mode }))}
        />
      </Field>
    </div>
  );
}

function ManifestForm({
  document,
  setDocument,
  schema,
  issues,
}: {
  document: Document;
  setDocument: (next: Document) => void;
  schema: StudioSchema;
  issues: Issue[];
}) {
  const { filesOfKind } = useStudio();
  const [generatorConfig, setGeneratorConfig] = useState<string | null>(null);
  const [generatorHparam, setGeneratorHparam] = useState<string | null>(null);
  const [generatorPrefix, setGeneratorPrefix] = useState("run");

  const runs = Array.isArray(document.runs) ? (document.runs as Run[]) : [];

  const setField = (key: string, value: unknown) => {
    const next = { ...document };
    if (value === null || value === undefined || value === "") delete next[key];
    else next[key] = value;
    setDocument(next);
  };

  const setRuns = (next: Run[]) => setDocument({ ...document, runs: next });

  const issuesFor = (field: string) => issues.filter((issue) => issue.field === field);

  const generateArms = () => {
    if (!generatorConfig) return;
    const generated: Run[] = LOSS_ARM_PRESETS.map((preset) => ({
      name: `${generatorPrefix}_${preset.name}`,
      experiment_config: generatorConfig,
      ...(generatorHparam ? { hparam_config: generatorHparam } : {}),
      args: ["--loss_miner_grid", preset.entry],
    }));
    const existing = new Set(runs.map((run) => run.name));
    setRuns([...runs, ...generated.filter((run) => !existing.has(run.name))]);
  };

  return (
    <div className="space-y-5">
      <Card
        title="Scheduler"
        subtitle="How launch_runs.py places these runs across devices. Each run is one complete process."
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Device" help="cuda places every child on a GPU; cpu keeps training on the host.">
            <Select
              value={(document.device as string) ?? null}
              onChange={(value) => setField("device", value)}
              options={[{ value: "cuda" }, { value: "cpu" }]}
              placeholder="cuda"
            />
          </Field>
          <Field label="GPUs" help="Indexes or UUIDs. Empty means every GPU the launcher discovers." issues={issuesFor("gpus")}>
            <ChipList
              values={Array.isArray(document.gpus) ? (document.gpus as unknown[]) : []}
              onChange={(values) => setField("gpus", values.length === 0 ? null : values.map(String))}
              placeholder="0"
            />
          </Field>
          <Field label="SSL GPUs" help="Only for CPU training that still wants CUDA for the SSL step.">
            <ChipList
              values={Array.isArray(document.ssl_gpus) ? (document.ssl_gpus as unknown[]) : []}
              onChange={(values) => setField("ssl_gpus", values.length === 0 ? null : values.map(String))}
            />
          </Field>
          <Field
            label="Concurrent runs per GPU"
            issues={issuesFor("max_concurrent_per_gpu")}
            help="Throughput saturates well before the slot count does."
          >
            <NumberInput
              value={(document.max_concurrent_per_gpu as number) ?? null}
              onChange={(value) => setField("max_concurrent_per_gpu", value)}
              placeholder="1"
            />
          </Field>
          <Field label="Host-wide run cap" issues={issuesFor("max_total_runs")}>
            <NumberInput
              value={(document.max_total_runs as number) ?? null}
              onChange={(value) => setField("max_total_runs", value)}
              placeholder="unlimited"
            />
          </Field>
          <Field label="Default GPU memory (MiB)" help="Per-run reservation used for admission.">
            <NumberInput
              value={(document.default_gpu_memory_mib as number) ?? null}
              onChange={(value) => setField("default_gpu_memory_mib", value)}
              placeholder="2048"
            />
          </Field>
          <Field label="Reserve GPU memory (MiB)" help="Headroom kept free on every GPU.">
            <NumberInput
              value={(document.reserve_gpu_memory_mib as number) ?? null}
              onChange={(value) => setField("reserve_gpu_memory_mib", value)}
              placeholder="1024"
            />
          </Field>
          <Field label="Poll interval (s)">
            <NumberInput
              value={(document.poll_interval_seconds as number) ?? null}
              onChange={(value) => setField("poll_interval_seconds", value)}
              placeholder="2"
            />
          </Field>
          <Field label="Launch delay (s)" help="Stagger between starts, so caches are not built five times at once.">
            <NumberInput
              value={(document.launch_delay_seconds as number) ?? null}
              onChange={(value) => setField("launch_delay_seconds", value)}
              placeholder="0"
            />
          </Field>
          <Field
            label="Save dir root"
            help="Each run without its own save_dir lands in <root>/<run name>."
            wide
          >
            <TextInput
              value={(document.save_dir_root as string) ?? ""}
              onChange={(value) => setField("save_dir_root", value)}
              mono
              placeholder="ssl/cars/lrml"
            />
          </Field>
          <Field label="Scheduler log dir">
            <TextInput
              value={(document.scheduler_log_dir as string) ?? ""}
              onChange={(value) => setField("scheduler_log_dir", value)}
              mono
              placeholder="logs/run_scheduler"
            />
          </Field>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <Toggle
            checked={document.fail_fast === true}
            onChange={(value) => setField("fail_fast", value ? true : null)}
            label="Fail fast"
            description="Stop scheduling as soon as one run fails."
          />
          <Toggle
            checked={document.allow_parallel_trials === true}
            onChange={(value) => setField("allow_parallel_trials", value ? true : null)}
            label="Allow parallel trials"
            description="Lets a child run several Optuna trials at once. Rarely what you want."
          />
        </div>
      </Card>

      <Card
        title={`Runs · ${runs.length}`}
        subtitle="Each entry becomes one main.py process with its own study."
        actions={
          <Button
            variant="primary"
            onClick={() =>
              setRuns([
                ...runs,
                { name: `run_${runs.length + 1}`, experiment_config: null, hparam_config: null },
              ])
            }
          >
            <Plus className="size-3.5" />
            Add run
          </Button>
        }
      >
        <div className="surface-soft mb-4 p-4">
          <p className="mb-3 flex items-center gap-2 text-xs font-medium text-[#e7ebf4]">
            <Wand2 className="size-3.5 text-[var(--cyan)]" />
            Generate one run per loss
          </p>
          <div className="grid gap-2 sm:grid-cols-4">
            <Select
              value={generatorConfig}
              onChange={setGeneratorConfig}
              options={filesOfKind("experiment").map((file) => ({ value: file.path }))}
              placeholder="Experiment config…"
            />
            <Select
              value={generatorHparam}
              onChange={setGeneratorHparam}
              options={filesOfKind("hparam").map((file) => ({ value: file.path }))}
              placeholder="Search config…"
            />
            <TextInput value={generatorPrefix} onChange={setGeneratorPrefix} placeholder="name prefix" mono />
            <Button variant="primary" onClick={generateArms} disabled={!generatorConfig}>
              <Plus className="size-3.5" />
              Add 5 arms
            </Button>
          </div>
          <p className="mt-2 text-[0.7rem] text-[#8f9bb2]">
            Creates {LOSS_ARM_PRESETS.map((preset) => preset.name).join(", ")} runs, each pinned to one
            loss through <code className="font-mono">--loss_miner_grid</code>.
          </p>
        </div>

        <div className="space-y-2">
          {runs.map((run, index) => (
            <RunCard
              key={index}
              run={run}
              index={index}
              document={document}
              schema={schema}
              issues={issues}
              onChange={(next) => setRuns(runs.map((item, position) => (position === index ? next : item)))}
              onRemove={() => setRuns(runs.filter((_, position) => position !== index))}
              onDuplicate={() => {
                const copy = { ...run, name: `${String(run.name ?? "run")}_copy` };
                setRuns([...runs.slice(0, index + 1), copy, ...runs.slice(index + 1)]);
              }}
            />
          ))}
          {runs.length === 0 && (
            <p className="py-6 text-center text-xs text-[#8f9bb2]">No runs yet.</p>
          )}
        </div>
      </Card>

      <Collapsible
        title="Manifest-wide study replay"
        subtitle="Defaults for every run that replays a study directory. A run's own block wins field by field."
      >
        <StudyReplayFields
          value={(document.study_replay ?? {}) as Record<string, unknown>}
          onChange={(next) => setField("study_replay", Object.keys(next).length === 0 ? null : next)}
          schema={schema}
        />
      </Collapsible>

      <Collapsible title="Advanced" subtitle="Interpreter, entrypoint, and working directory overrides.">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Python" help="Defaults to the interpreter that starts launch_runs.py.">
            <TextInput
              value={(document.python as string) ?? ""}
              onChange={(value) => setField("python", value)}
              mono
              placeholder={String(schema.manifest.defaults.python)}
            />
          </Field>
          <Field label="Entrypoint">
            <TextInput
              value={(document.entrypoint as string) ?? ""}
              onChange={(value) => setField("entrypoint", value)}
              mono
              placeholder="main.py"
            />
          </Field>
          <Field label="Working dir" help="Defaults to wherever launch_runs.py is started from.">
            <TextInput
              value={(document.working_dir as string) ?? ""}
              onChange={(value) => setField("working_dir", value)}
              mono
              placeholder="repository root"
            />
          </Field>
        </div>
        <div className="mt-4">
          <p className="studio-label mb-2">GPU batch-size thresholds</p>
          <ChipList
            values={
              Array.isArray(document.gpu_batch_size_thresholds)
                ? (document.gpu_batch_size_thresholds as unknown[])
                : []
            }
            onChange={(values) =>
              setField("gpu_batch_size_thresholds", values.length === 0 ? null : values)
            }
            parse={(text) => parseScalar(text)}
            placeholder="NTXentLoss:all=0"
          />
        </div>
      </Collapsible>
    </div>
  );
}

export function ManifestEditor() {
  const { schema, knownPaths } = useStudio();

  const lint = useCallback(
    (document: Document) => (schema ? lintManifest(document, schema, knownPaths) : []),
    [schema, knownPaths],
  );

  const summary = useMemo(
    () => (
      <p className="flex items-start gap-2 text-[0.7rem] leading-snug text-[#8f9bb2]">
        <FolderTree className="mt-0.5 size-3.5 shrink-0" />
        Verifying a manifest runs the launcher&apos;s own loader, so every referenced config is checked and
        the exact child command is shown.
      </p>
    ),
    [],
  );

  if (!schema) return null;

  return (
    <ConfigWorkbench
      kind="manifest"
      defaultDirectory="configs/run_manifests"
      emptyDocument={emptyDocument}
      validateKind="manifest"
      validatePayload={(document) => ({ data: document })}
      lint={lint}
      sidebarFooter={summary}
    >
      {({ document, setDocument, issues }) => (
        <ManifestForm document={document} setDocument={setDocument} schema={schema} issues={issues} />
      )}
    </ConfigWorkbench>
  );
}
