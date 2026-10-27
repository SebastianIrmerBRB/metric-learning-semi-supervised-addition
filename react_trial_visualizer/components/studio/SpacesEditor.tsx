"use client";

import { Layers, Link2, Plus, Sparkles } from "lucide-react";
import { useCallback, useMemo } from "react";

import { buildSpaceKeyOptions, relativeConfigPath } from "@/lib/studio/keys";
import { lintHparamConfig } from "@/lib/studio/lint";
import type { SpaceSpec, StudioSchema } from "@/lib/studio/schema";

import { ConfigWorkbench } from "./ConfigWorkbench";
import { SpaceRow } from "./SpaceRow";
import { useStudio } from "./StudioContext";
import {
  Button,
  Card,
  Collapsible,
  Field,
  KeyValueEditor,
  NumberInput,
  Select,
  TextInput,
  Toggle,
} from "./ui";

type Document = Record<string, unknown>;

function emptyDocument(): Document {
  return {
    enabled: true,
    n_trials: 40,
    direction: "maximize",
    metric: "best_valid_mean_average_precision_at_r",
    sampler: "tpe",
    pruner: "none",
    spaces: {
      lr: { type: "float", low: 1e-5, high: 5e-4, log: true },
    },
  };
}

function SpacesForm({
  document,
  setDocument,
  schema,
  path,
  inheritedSpaces,
}: {
  document: Document;
  setDocument: (next: Document) => void;
  schema: StudioSchema;
  path: string | null;
  inheritedSpaces: Record<string, SpaceSpec>;
}) {
  const { filesOfKind } = useStudio();
  const keyOptions = useMemo(() => buildSpaceKeyOptions(schema), [schema]);
  const spaces = (document.spaces ?? {}) as Record<string, SpaceSpec>;
  const perLoss = (document.per_loss ?? {}) as Record<string, { spaces?: Record<string, SpaceSpec> }>;

  const setField = (key: string, value: unknown) => {
    const next = { ...document };
    if (value === null || value === undefined || value === "") delete next[key];
    else next[key] = value;
    setDocument(next);
  };

  const setSpaces = (next: Record<string, SpaceSpec>) => setDocument({ ...document, spaces: next });

  const renameSpace = (from: string, to: string) => {
    const entries = Object.entries(spaces).map(([key, value]) => (key === from ? [to, value] : [key, value]));
    setSpaces(Object.fromEntries(entries) as Record<string, SpaceSpec>);
  };

  const addSpace = (key: string) => {
    if (!key || key in spaces) return;
    setSpaces({ ...spaces, [key]: { type: "float", low: 0.001, high: 1, log: true } });
  };

  /** Seed a new row with the most-used key that is still free, not a placeholder. */
  const nextSuggestedKey = () =>
    schema.usedSpaceKeys.find((entry) => !(entry.key in spaces))?.key ??
    `new_parameter_${Object.keys(spaces).length + 1}`;

  const extendsValue = document.extends;
  const parents = Array.isArray(extendsValue)
    ? (extendsValue as string[])
    : typeof extendsValue === "string"
      ? [extendsValue]
      : [];

  const parentCandidates = filesOfKind("hparam")
    .filter((file) => file.path !== path)
    .map((file) => file.path);

  const unusedSuggestions = schema.usedSpaceKeys
    .filter((entry) => !(entry.key in spaces))
    .slice(0, 8);

  return (
    <div className="space-y-5">
      <Card
        title="Study"
        subtitle="How Optuna runs this search. Every value left blank falls back to the parent config, then to the project default."
      >
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="Study name" help="Trials with the same name share one storage and one dashboard.">
            <TextInput
              value={(document.study_name as string) ?? ""}
              onChange={(value) => setField("study_name", value || null)}
              placeholder="inherited"
              mono
            />
          </Field>
          <Field label="Trials" help="Total trials this study will run, resumed runs included.">
            <NumberInput
              value={(document.n_trials as number) ?? null}
              onChange={(value) => setField("n_trials", value)}
              placeholder="20"
            />
          </Field>
          <Field label="Objective metric">
            <Select
              value={(document.metric as string) ?? null}
              onChange={(value) => setField("metric", value)}
              options={schema.constants.objectiveMetrics.map((value) => ({ value }))}
              placeholder="inherited"
            />
          </Field>
          <Field label="Direction">
            <Select
              value={(document.direction as string) ?? null}
              onChange={(value) => setField("direction", value)}
              options={schema.constants.directions.map((value) => ({ value }))}
              placeholder="inherited"
            />
          </Field>
          <Field label="Sampler">
            <Select
              value={(document.sampler as string) ?? null}
              onChange={(value) => setField("sampler", value)}
              options={schema.constants.samplers.map((value) => ({ value }))}
              placeholder="inherited"
            />
          </Field>
          <Field
            label="TPE startup trials"
            help="Random trials before TPE takes over. Only valid for the TPE sampler."
          >
            <NumberInput
              value={(document.tpe_startup_trials as number) ?? null}
              onChange={(value) => setField("tpe_startup_trials", value)}
              placeholder="inherited"
            />
          </Field>
          <Field label="Pruner">
            <Select
              value={(document.pruner as string) ?? null}
              onChange={(value) => setField("pruner", value)}
              options={schema.constants.pruners.map((value) => ({ value }))}
              placeholder="inherited"
            />
          </Field>
          <Field label="Timeout (seconds)" help="Wall-clock cap for the whole study; blank means none.">
            <NumberInput
              value={(document.timeout as number) ?? null}
              onChange={(value) => setField("timeout", value)}
              placeholder="none"
            />
          </Field>
          <Field label="Parallel trials (n_jobs)" help="Keep this at 1; launch_runs.py parallelizes whole runs.">
            <NumberInput
              value={(document.n_jobs as number) ?? null}
              onChange={(value) => setField("n_jobs", value)}
              placeholder="1"
            />
          </Field>
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Toggle
            checked={document.enabled !== false}
            onChange={(value) => setField("enabled", value)}
            label="Search enabled"
            description="Off runs a single fixed-parameter fit."
          />
          <Toggle
            checked={document.load_if_exists !== false}
            onChange={(value) => setField("load_if_exists", value)}
            label="Resume existing study"
          />
          <Toggle
            checked={document.retry_failed_trials === true}
            onChange={(value) => setField("retry_failed_trials", value)}
            label="Retry failed trials"
          />
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <Field label="Sampler params" help="Passed straight to the Optuna sampler, e.g. multivariate / group.">
            <div className="mt-1">
              <KeyValueEditor
                entries={(document.sampler_params ?? {}) as Record<string, unknown>}
                onChange={(entries) => setField("sampler_params", entries)}
              />
            </div>
          </Field>
          <Field label="Pruner params">
            <div className="mt-1">
              <KeyValueEditor
                entries={(document.pruner_params ?? {}) as Record<string, unknown>}
                onChange={(entries) => setField("pruner_params", entries)}
              />
            </div>
          </Field>
        </div>
      </Card>

      <Card
        title="Inheritance"
        subtitle="Parents merge left to right, and this file outranks all of them. Paths are stored relative to this file."
      >
        <div className="flex flex-wrap items-center gap-2">
          {parents.map((parent) => (
            <span key={parent} className="studio-chip">
              <Link2 className="size-3 opacity-70" />
              {parent}
              <button
                type="button"
                onClick={() => {
                  const next = parents.filter((item) => item !== parent);
                  setField("extends", next.length === 0 ? null : next.length === 1 ? next[0] : next);
                }}
                className="focus-ring rounded-full px-1 text-[#8f9bb2] hover:text-[#f7f8fc]"
              >
                ×
              </button>
            </span>
          ))}
          {parents.length === 0 && <span className="text-xs text-[#8f9bb2]">Standalone config.</span>}
        </div>
        <div className="mt-3 max-w-md">
          <Select
            value=""
            onChange={(value) => {
              if (!value) return;
              const relative = path ? relativeConfigPath(path, value) : value;
              const next = [...parents, relative];
              setField("extends", next.length === 1 ? next[0] : next);
            }}
            options={parentCandidates.map((candidate) => ({ value: candidate }))}
            placeholder="Add a parent config…"
          />
        </div>
      </Card>

      <Card
        title={`Search spaces · ${Object.keys(spaces).length}`}
        subtitle="One card per searched dimension. The key must name a real main.py argument, loss/miner parameter, or SSL config path."
        actions={
          <Button variant="primary" onClick={() => addSpace(nextSuggestedKey())}>
            <Plus className="size-3.5" />
            Add dimension
          </Button>
        }
      >
        {unusedSuggestions.length > 0 && (
          <div className="mb-4">
            <p className="studio-label mb-2 flex items-center gap-1.5">
              <Sparkles className="size-3" />
              Common in this project
            </p>
            <div className="flex flex-wrap gap-1.5">
              {unusedSuggestions.map((entry) => (
                <button
                  key={entry.key}
                  type="button"
                  onClick={() => addSpace(entry.key)}
                  className="focus-ring studio-chip hover:border-[var(--violet)]/50 hover:text-[#f7f8fc]"
                >
                  <Plus className="size-3 opacity-60" />
                  {entry.key}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-3">
          {Object.entries(spaces).map(([key, spec], index) => (
            <SpaceRow
              key={index}
              name={key}
              spec={spec}
              schema={schema}
              keyOptions={keyOptions}
              takenNames={Object.keys(spaces).filter((existing) => existing !== key)}
              onRename={(next) => renameSpace(key, next)}
              onChange={(next) => setSpaces({ ...spaces, [key]: next })}
              onRemove={() => {
                const next = { ...spaces };
                delete next[key];
                setSpaces(next);
              }}
            />
          ))}
          {Object.keys(spaces).length === 0 && (
            <p className="py-6 text-center text-xs text-[#8f9bb2]">
              No local dimensions. This config only reshapes what it inherits.
            </p>
          )}
        </div>
      </Card>

      {Object.keys(inheritedSpaces).length > 0 && (
        <Card
          title="Inherited dimensions"
          subtitle="Searched through the parent chain. Override one to pin a different range for this dataset."
        >
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(inheritedSpaces)
              .filter(([key]) => !(key in spaces))
              .map(([key, spec]) => (
                <button
                  key={key}
                  type="button"
                  title="Copy into this config to override it"
                  onClick={() => setSpaces({ ...spaces, [key]: spec })}
                  className="focus-ring studio-chip hover:border-[var(--cyan)]/50 hover:text-[#f7f8fc]"
                >
                  <Layers className="size-3 opacity-60" />
                  {key}
                </button>
              ))}
          </div>
        </Card>
      )}

      <Card
        title="Per-loss overrides"
        subtitle="Applied once a trial's loss is fixed. Keys that cannot be class-qualified, such as lr, live here."
        actions={
          <div className="w-56">
            <Select
              value=""
              onChange={(loss) => {
                if (!loss || loss in perLoss) return;
                setDocument({ ...document, per_loss: { ...perLoss, [loss]: { spaces: {} } } });
              }}
              options={schema.constants.losses.map((value) => ({ value }))}
              placeholder="Add a loss…"
            />
          </div>
        }
      >
        <div className="space-y-3">
          {Object.entries(perLoss).map(([loss, block]) => {
            const blockSpaces = (block?.spaces ?? {}) as Record<string, SpaceSpec>;
            const setBlockSpaces = (next: Record<string, SpaceSpec>) =>
              setDocument({
                ...document,
                per_loss: { ...perLoss, [loss]: { ...block, spaces: next } },
              });
            return (
              <Collapsible
                key={loss}
                title={loss}
                subtitle={`${Object.keys(blockSpaces).length} override${Object.keys(blockSpaces).length === 1 ? "" : "s"}`}
                defaultOpen={Object.keys(perLoss).length === 1}
              >
                <div className="space-y-3">
                  {Object.entries(blockSpaces).map(([key, spec], index) => (
                    <SpaceRow
                      key={index}
                      name={key}
                      spec={spec}
                      schema={schema}
                      keyOptions={keyOptions}
                      takenNames={Object.keys(blockSpaces).filter((existing) => existing !== key)}
                      onRename={(next) => {
                        const entries = Object.entries(blockSpaces).map(([existing, value]) =>
                          existing === key ? [next, value] : [existing, value],
                        );
                        setBlockSpaces(Object.fromEntries(entries) as Record<string, SpaceSpec>);
                      }}
                      onChange={(next) => setBlockSpaces({ ...blockSpaces, [key]: next })}
                      onRemove={() => {
                        const next = { ...blockSpaces };
                        delete next[key];
                        setBlockSpaces(next);
                      }}
                    />
                  ))}
                  <div className="flex gap-2">
                    <Button
                      onClick={() =>
                        setBlockSpaces({
                          ...blockSpaces,
                          [`new_parameter_${Object.keys(blockSpaces).length + 1}`]: {
                            type: "float",
                            low: 1e-5,
                            high: 1,
                            log: true,
                          },
                        })
                      }
                    >
                      <Plus className="size-3.5" />
                      Add override
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => {
                        const next = { ...perLoss };
                        delete next[loss];
                        setDocument({ ...document, per_loss: next });
                      }}
                    >
                      Remove {loss}
                    </Button>
                  </div>
                </div>
              </Collapsible>
            );
          })}
          {Object.keys(perLoss).length === 0 && (
            <p className="py-4 text-center text-xs text-[#8f9bb2]">
              No per-loss overrides. Every loss shares the search space above.
            </p>
          )}
        </div>
      </Card>
    </div>
  );
}

export function SpacesEditor() {
  const { schema } = useStudio();

  const lint = useCallback(
    (document: Document) => (schema ? lintHparamConfig(document, schema) : []),
    [schema],
  );

  const validatePayload = useCallback((document: Document, path: string | null) => {
    const directory = path ? path.split("/").slice(0, -1).join("/") : "configs/runs";
    return {
      data: document,
      baseDir: directory,
      fileName: path ? path.split("/").pop() : "unsaved.json",
    };
  }, []);

  if (!schema) return null;

  return (
    <ConfigWorkbench
      kind="hparam"
      defaultDirectory="configs/runs"
      emptyDocument={emptyDocument}
      validateKind="hparam"
      validatePayload={validatePayload}
      lint={lint}
    >
      {({ document, setDocument, path, serverResult }) => {
        const resolvedSpaces = (serverResult?.resolved?.spaces ?? {}) as Record<string, SpaceSpec>;
        return (
          <SpacesForm
            document={document}
            setDocument={setDocument}
            schema={schema}
            path={path}
            inheritedSpaces={resolvedSpaces}
          />
        );
      }}
    </ConfigWorkbench>
  );
}
