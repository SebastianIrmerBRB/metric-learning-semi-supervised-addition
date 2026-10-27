"use client";

import { Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { describeSpaceKey, isCategoricalSpec, lintSpace, specChoices } from "@/lib/studio/lint";
import type { KeyOption } from "@/lib/studio/keys";
import type { SpaceSpec, StudioSchema } from "@/lib/studio/schema";

import { ChipList, Combobox, Field, IssueIcon, NumberInput, Select, Toggle, issueTextClass, parseScalar } from "./ui";

type SpaceType = "float" | "int" | "categorical";

function specType(spec: SpaceSpec): SpaceType {
  if (isCategoricalSpec(spec)) return "categorical";
  const type = (spec as { type?: string }).type;
  return type === "int" ? "int" : "float";
}

/** A compact ladder of the values this dimension will actually be sampled from. */
function RangePreview({
  low,
  high,
  log,
  integer,
}: {
  low: number;
  high: number;
  log: boolean;
  integer: boolean;
}) {
  const samples = useMemo(() => {
    if (!Number.isFinite(low) || !Number.isFinite(high) || low > high) return [];
    const steps = 5;
    return Array.from({ length: steps }, (_, index) => {
      const fraction = index / (steps - 1);
      if (log && low > 0) {
        return low * (high / low) ** fraction;
      }
      return low + (high - low) * fraction;
    }).map((value) => (integer ? Math.round(value) : value));
  }, [low, high, log, integer]);

  if (samples.length === 0) return null;

  const decades = log && low > 0 ? Math.log10(high / low) : null;

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between gap-2">
        {samples.map((value, index) => (
          <span
            key={index}
            className={`font-mono text-[0.65rem] ${
              index === 0 || index === samples.length - 1 ? "text-[#d9e0ed]" : "text-[#8f9bb2]"
            }`}
          >
            {formatNumber(value)}
          </span>
        ))}
      </div>
      <div className="mt-1 h-1 rounded-full bg-gradient-to-r from-[var(--violet)]/70 via-[var(--cyan)]/50 to-[var(--lime)]/60" />
      <p className="mt-1.5 text-[0.65rem] text-[#8f9bb2]">
        {log ? "Log-uniform" : "Uniform"}
        {decades !== null && ` · ${decades.toFixed(1)} decades`}
        {integer && " · integers"}
      </p>
    </div>
  );
}

function formatNumber(value: number): string {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(1);
  return Number(value.toPrecision(4)).toString();
}

export function SpaceRow({
  name,
  spec,
  schema,
  keyOptions,
  onRename,
  onChange,
  onRemove,
  takenNames,
}: {
  name: string;
  spec: SpaceSpec;
  schema: StudioSchema;
  keyOptions: KeyOption[];
  onRename: (next: string) => void;
  onChange: (next: SpaceSpec) => void;
  onRemove: () => void;
  takenNames: string[];
}) {
  // The key is committed on blur. Renaming on every keystroke would rebuild the
  // spaces object under the caret and could briefly collide with another key.
  const [draftName, setDraftName] = useState<string | null>(null);
  const shownName = draftName ?? name;
  const collides = draftName !== null && draftName !== name && takenNames.includes(draftName);

  const commitName = () => {
    if (draftName === null) return;
    const next = draftName.trim();
    if (next && next !== name && !takenNames.includes(next)) onRename(next);
    setDraftName(null);
  };

  const info = describeSpaceKey(name, schema);
  const issues = useMemo(() => lintSpace(name, spec, schema), [name, spec, schema]);
  const type = specType(spec);
  const record = (Array.isArray(spec) ? {} : spec) as Record<string, unknown>;

  const changeType = (next: SpaceType) => {
    if (next === "categorical") {
      onChange({ type: "categorical", choices: specChoices(spec) });
      return;
    }
    onChange({
      type: next,
      low: typeof record.low === "number" ? record.low : next === "int" ? 1 : 0.001,
      high: typeof record.high === "number" ? record.high : next === "int" ? 10 : 1,
      log: record.log === true,
    });
  };

  const patch = (changes: Record<string, unknown>) => {
    onChange({ ...record, type, ...changes } as SpaceSpec);
  };

  const worst = issues.find((issue) => issue.level === "error") ?? issues[0];

  return (
    <div className="surface-soft studio-row-enter p-4">
      <div className="flex flex-wrap items-start gap-2">
        <div className="min-w-[16rem] flex-1">
          <Combobox
            value={shownName}
            onChange={setDraftName}
            onCommit={commitName}
            options={keyOptions.map((option) => ({
              value: option.value,
              label: `${option.group} — ${option.label}`,
            }))}
            invalid={!info.known || collides}
            placeholder="parameter key"
          />
          {collides && (
            <p className="mt-1 text-[0.7rem] text-[#ff9a9a]">
              Another dimension already searches {draftName}; the rename will be discarded.
            </p>
          )}
        </div>
        <div className="w-36">
          <Select
            value={type}
            allowEmpty={false}
            onChange={(next) => next && changeType(next as SpaceType)}
            options={[
              { value: "float", label: "float range" },
              { value: "int", label: "int range" },
              { value: "categorical", label: "categorical" },
            ]}
          />
        </div>
        <button
          type="button"
          aria-label={`Remove ${name}`}
          onClick={onRemove}
          className="focus-ring flex size-[2.35rem] items-center justify-center rounded-[0.7rem] border border-white/[0.1] text-[#8f9bb2] hover:border-[#ff9a9a]/40 hover:text-[#ff9a9a]"
        >
          <Trash2 className="size-3.5" />
        </button>
      </div>

      {info.description && (
        <p className="mt-2 text-[0.7rem] leading-snug text-[#8f9bb2]">{info.description}</p>
      )}

      {type === "categorical" ? (
        <div className="mt-3">
          <label className="studio-label mb-1.5">Choices</label>
          <ChipList
            values={specChoices(spec)}
            onChange={(choices) => onChange({ type: "categorical", choices })}
            parse={parseScalar}
            placeholder={
              name === schema.constants.batchSamplerKey ? "e.g. 128:4" : "Type a value and press Enter"
            }
            suggestions={
              name === schema.constants.batchSamplerKey
                ? ["32:4", "64:4", "128:4", "256:4", "128:8", "256:8"]
                : []
            }
          />
        </div>
      ) : (
        <>
          <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1fr_auto]">
            <Field label="Low">
              <NumberInput
                value={typeof record.low === "number" ? record.low : null}
                onChange={(value) => patch({ low: value })}
              />
            </Field>
            <Field label="High">
              <NumberInput
                value={typeof record.high === "number" ? record.high : null}
                onChange={(value) => patch({ high: value })}
              />
            </Field>
            <div className="flex items-end pb-1.5">
              <Toggle
                checked={record.log === true}
                onChange={(value) => patch({ log: value })}
                label="Log scale"
              />
            </div>
          </div>
          {typeof record.low === "number" && typeof record.high === "number" && (
            <RangePreview
              low={record.low}
              high={record.high}
              log={record.log === true}
              integer={type === "int"}
            />
          )}
        </>
      )}

      {worst && (
        <p className={`mt-3 flex gap-1.5 text-[0.7rem] leading-snug ${issueTextClass(worst.level)}`}>
          <IssueIcon level={worst.level} />
          <span>{worst.message}</span>
        </p>
      )}
    </div>
  );
}
