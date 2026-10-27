"use client";


import { useCallback, useMemo } from "react";

import { lintExperimentConfig } from "@/lib/studio/lint";
import type { CliArgument, Issue, StudioSchema } from "@/lib/studio/schema";

import { CliField } from "./CliField";
import { ConfigWorkbench } from "./ConfigWorkbench";
import { LossMinerGrid } from "./LossMinerGrid";
import { ReferencedConfig } from "./ReferencedConfig";
import { useStudio } from "./StudioContext";
import { Card, Collapsible, Select } from "./ui";

type Document = Record<string, unknown>;

/** The fields these configs actually set, grouped the way a run is reasoned about. */
const SECTIONS: { title: string; subtitle: string; fields: string[] }[] = [
  {
    title: "Dataset and splits",
    subtitle: "What data the run sees, and how it is divided.",
    fields: ["dataset", "dataset_protocol", "val_mode", "cv_k", "cv_mode", "seed", "data_split_seed", "support_seed"],
  },
  {
    title: "Training",
    subtitle: "The optimization budget and the backbone policy.",
    fields: ["mode", "backbone_tuning", "use_cache", "epochs", "patience", "feat_dim", "batch_size", "sampler_m", "lr", "optim", "num_workers", "device"],
  },
  {
    title: "Semi-supervised method",
    subtitle: "Which SSL config runs, and over which label budgets.",
    fields: ["ssl_config", "ssl_label_sampling_modes", "label_budget_grid", "k_shot_grid", "ssl_device", "ssl_gradient_contribution_log_interval"],
  },
  {
    title: "Search and output",
    subtitle: "Where results land and which search config drives them.",
    fields: ["save_dir", "hparam_config", "final_test_after_hpo", "final_test_top_n", "final_test_visualization"],
  },
];

const CURATED = new Set([...SECTIONS.flatMap((section) => section.fields), "loss_miner_grid"]);

function ExperimentForm({
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
  const argumentsByDest = useMemo(
    () => new Map(schema.cliArguments.map((item) => [item.dest, item])),
    [schema],
  );

  const issuesFor = (field: string) => issues.filter((issue) => issue.field === field);

  const setField = (key: string, value: unknown) => {
    const next = { ...document };
    if (value === null || value === undefined || (Array.isArray(value) && value.length === 0 && !(key in document))) {
      delete next[key];
    } else {
      next[key] = value;
    }
    setDocument(next);
  };

  const optionsFor = (dest: string): { value: string; label?: string }[] | undefined => {
    if (dest === "ssl_config") {
      return filesOfKind("ssl-method").map((file) => ({ value: file.path }));
    }
    if (dest === "hparam_config") {
      return filesOfKind("hparam").map((file) => ({ value: file.path }));
    }
    return undefined;
  };

  const renderField = (dest: string) => {
    const argument = argumentsByDest.get(dest);
    if (!argument) return null;
    return (
      <CliField
        key={dest}
        argument={argument}
        value={document[dest]}
        onChange={(value) => setField(dest, value)}
        issues={issuesFor(dest)}
        options={optionsFor(dest)}
      />
    );
  };

  const extraKeys = Object.keys(document).filter((key) => !CURATED.has(key));
  const addableArguments: CliArgument[] = schema.cliArguments
    .filter((argument) => !CURATED.has(argument.dest) && !(argument.dest in document))
    .filter((argument) => argument.dest !== "experiment_config");

  return (
    <div className="space-y-5">
      {SECTIONS.map((section) => (
        <Card key={section.title} title={section.title} subtitle={section.subtitle}>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {section.fields.map(renderField)}
          </div>
          {section.fields.includes("ssl_config") &&
            typeof document.ssl_config === "string" &&
            document.ssl_config && (
              <div className="surface-soft mt-4 p-4">
                <ReferencedConfig
                  path={document.ssl_config}
                  title={`Inside ${document.ssl_config}`}
                />
              </div>
            )}
        </Card>
      ))}

      <Card
        title="Comparison arms"
        subtitle="Every loss/miner pair the run trains and compares under the same budget."
      >
        <div className="grid gap-4">
          <LossMinerGrid
            values={Array.isArray(document.loss_miner_grid) ? (document.loss_miner_grid as string[]) : []}
            onChange={(values) => setField("loss_miner_grid", values)}
            schema={schema}
            issues={issuesFor("loss_miner_grid")}
          />
        </div>
      </Card>

      <Card
        title="Other settings"
        subtitle="Anything else this config pins. Every remaining main.py option can be added here."
        actions={
          <div className="w-64">
            <Select
              value=""
              onChange={(dest) => {
                if (!dest) return;
                const argument = argumentsByDest.get(dest);
                setField(dest, argument?.isList ? [] : argument?.isFlag ? true : argument?.default ?? null);
              }}
              options={addableArguments.map((argument) => ({
                value: argument.dest,
                label: `${argument.dest} — ${argument.help.slice(0, 60)}`,
              }))}
              placeholder="Add a setting…"
            />
          </div>
        }
      >
        {extraKeys.length === 0 ? (
          <p className="py-4 text-center text-xs text-[#8f9bb2]">
            Nothing beyond the grouped fields above.
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {extraKeys.map((key) => {
              const argument = argumentsByDest.get(key);
              if (!argument) {
                return (
                  <div key={key} className="surface-soft p-3">
                    <p className="font-mono text-xs text-[#ff9a9a]">{key}</p>
                    <p className="mt-1 text-[0.7rem] text-[#8f9bb2]">
                      Not a main.py argument. Loading this config fails until it is removed.
                    </p>
                    <button
                      type="button"
                      onClick={() => setField(key, null)}
                      className="focus-ring mt-2 text-[0.7rem] text-[#ff9a9a] underline"
                    >
                      Remove it
                    </button>
                  </div>
                );
              }
              return (
                <div key={key} className="relative">
                  <CliField
                    argument={argument}
                    value={document[key]}
                    onChange={(value) => setField(key, value)}
                    issues={issuesFor(key)}
                    options={optionsFor(key)}
                  />
                  <button
                    type="button"
                    aria-label={`Remove ${key}`}
                    onClick={() => {
                      const next = { ...document };
                      delete next[key];
                      setDocument(next);
                    }}
                    className="focus-ring absolute right-0 top-0 text-[0.65rem] text-[#8f9bb2] hover:text-[#ff9a9a]"
                  >
                    remove
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <Collapsible
        title="Every main.py option"
        subtitle="Reference: the full argument surface this config can override."
      >
        <div className="studio-scroll max-h-96 overflow-y-auto">
          <table className="w-full text-left text-[0.7rem]">
            <tbody>
              {schema.cliArguments.map((argument) => (
                <tr key={argument.dest} className="border-b border-white/[0.05]">
                  <td className="py-1.5 pr-3 align-top font-mono text-[#d9e0ed]">{argument.dest}</td>
                  <td className="py-1.5 pr-3 align-top font-mono text-[#8f9bb2]">
                    {argument.choices ? argument.choices.map(String).join(" | ") : (argument.type ?? "str")}
                  </td>
                  <td className="py-1.5 align-top text-[#8f9bb2]">{argument.help}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Collapsible>
    </div>
  );
}

export function ExperimentEditor() {
  const { schema, knownPaths } = useStudio();

  const lint = useCallback(
    (document: Document) => (schema ? lintExperimentConfig(document, schema, knownPaths) : []),
    [schema, knownPaths],
  );

  if (!schema) return null;

  return (
    <ConfigWorkbench
      kind="experiment"
      defaultDirectory="configs/phase_1"
      emptyDocument={() => ({
        save_dir: "",
        dataset: "Cars196",
        mode: "ssl",
        cv_k: 4,
        epochs: 600,
        patience: 20,
        backbone_tuning: "frozen",
        use_cache: true,
        loss_miner_grid: ["ArcFaceLoss:no_miner"],
      })}
      validateKind="experiment"
      validatePayload={(document, path) => ({ data: document, path })}
      lint={lint}
      sidebarFooter={
        <p className="text-[0.7rem] leading-snug text-[#8f9bb2]">
          Keys here are argparse destination names. Anything <code className="font-mono">main.py</code>{" "}
          accepts on the command line can be pinned in this file.
        </p>
      }
    >
      {({ document, setDocument, issues }) => (
        <ExperimentForm
          document={document}
          setDocument={setDocument}
          schema={schema}
          issues={issues}
        />
      )}
    </ConfigWorkbench>
  );
}
