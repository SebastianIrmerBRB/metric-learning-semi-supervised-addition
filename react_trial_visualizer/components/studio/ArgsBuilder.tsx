"use client";

import { Plus, Terminal, X } from "lucide-react";
import { useMemo, useState } from "react";

import type { CliArgument, StudioSchema } from "@/lib/studio/schema";

import { ChipList, Select, parseScalar } from "./ui";

interface ArgumentGroup {
  option: string;
  values: string[];
}

/** Split a flat argv tail into option groups so it can be edited as rows. */
export function parseArgs(tokens: string[]): ArgumentGroup[] {
  const groups: ArgumentGroup[] = [];
  for (const token of tokens) {
    if (token.startsWith("-")) {
      const [option, inlineValue] = token.split("=", 2);
      groups.push({ option, values: inlineValue === undefined ? [] : [inlineValue] });
      continue;
    }
    if (groups.length === 0) {
      groups.push({ option: "", values: [token] });
      continue;
    }
    groups[groups.length - 1].values.push(token);
  }
  return groups;
}

export function serializeArgs(groups: ArgumentGroup[]): string[] {
  return groups.flatMap((group) => (group.option ? [group.option, ...group.values] : group.values));
}

/** Per-run command-line overrides, built from the real argument list. */
export function ArgsBuilder({
  tokens,
  onChange,
  schema,
  forbidden,
}: {
  tokens: string[];
  onChange: (tokens: string[]) => void;
  schema: StudioSchema;
  forbidden: string[];
}) {
  const [pending, setPending] = useState<string | null>(null);
  const groups = useMemo(() => parseArgs(tokens), [tokens]);

  const byOption = useMemo(() => {
    const map = new Map<string, CliArgument>();
    for (const argument of schema.cliArguments) {
      for (const option of argument.options) map.set(option, argument);
    }
    return map;
  }, [schema]);

  const addable = schema.cliArguments
    .filter((argument) => argument.options.length > 0)
    .map((argument) => ({
      value: argument.options[0],
      label: `${argument.options[0]} — ${argument.help.slice(0, 60)}`,
    }))
    .sort((left, right) => left.value.localeCompare(right.value));

  const update = (index: number, next: ArgumentGroup) => {
    onChange(serializeArgs(groups.map((group, position) => (position === index ? next : group))));
  };

  return (
    <div>
      <div className="space-y-2">
        {groups.map((group, index) => {
          const argument = byOption.get(group.option);
          const blocked = forbidden.includes(group.option);
          const unknown = !argument && group.option.startsWith("-");
          return (
            <div
              key={`${group.option}-${index}`}
              className={`surface-soft p-3 ${blocked || unknown ? "border-[#ff9a9a]/40" : ""}`}
            >
              <div className="flex items-center gap-2">
                <Terminal className="size-3.5 shrink-0 text-[#8f9bb2]" />
                <span
                  className={`flex-1 truncate font-mono text-xs ${
                    blocked || unknown ? "text-[#ff9a9a]" : "text-[#e7ebf4]"
                  }`}
                >
                  {group.option || "(positional)"}
                </span>
                <button
                  type="button"
                  aria-label="Remove argument"
                  onClick={() => onChange(serializeArgs(groups.filter((_, position) => position !== index)))}
                  className="focus-ring rounded-lg p-1 text-[#8f9bb2] hover:text-[#ff9a9a]"
                >
                  <X className="size-3.5" />
                </button>
              </div>
              {argument?.help && (
                <p className="mt-1 text-[0.7rem] leading-snug text-[#8f9bb2]">{argument.help}</p>
              )}
              {blocked && (
                <p className="mt-1 text-[0.7rem] text-[#ff9a9a]">
                  The scheduler sets this itself; a run that passes it is rejected.
                </p>
              )}
              {unknown && (
                <p className="mt-1 text-[0.7rem] text-[#ff9a9a]">
                  Not an option <code className="font-mono">main.py</code> accepts.
                </p>
              )}
              <div className="mt-2">
                {argument?.choices ? (
                  <div>
                    <div className="flex flex-wrap gap-1.5">
                      {group.values.map((value, position) => (
                        <span key={`${value}-${position}`} className="studio-chip">
                          {value}
                          <button
                            type="button"
                            aria-label="Remove value"
                            onClick={() =>
                              update(index, {
                                ...group,
                                values: group.values.filter((_, at) => at !== position),
                              })
                            }
                            className="focus-ring rounded-full text-[#8f9bb2] hover:text-[#f7f8fc]"
                          >
                            <X className="size-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                    <div className="mt-2">
                      <Select
                        value=""
                        onChange={(value) =>
                          value && update(index, { ...group, values: [...group.values, value] })
                        }
                        options={argument.choices.map((choice) => ({ value: String(choice) }))}
                        placeholder="Add value…"
                      />
                    </div>
                  </div>
                ) : (
                  <ChipList
                    values={group.values}
                    onChange={(values) => update(index, { ...group, values: values.map(String) })}
                    parse={(text) => String(parseScalar(text))}
                    placeholder="Value…"
                  />
                )}
              </div>
            </div>
          );
        })}
        {groups.length === 0 && (
          <p className="py-2 text-xs text-[#8f9bb2]">No overrides; this run uses its configs as written.</p>
        )}
      </div>

      <div className="mt-2 flex gap-2">
        <div className="min-w-0 flex-1">
          <Select value={pending} onChange={setPending} options={addable} placeholder="Add an option…" />
        </div>
        <button
          type="button"
          disabled={!pending}
          onClick={() => {
            if (!pending) return;
            onChange(serializeArgs([...groups, { option: pending, values: [] }]));
            setPending(null);
          }}
          className="focus-ring flex h-[2.35rem] shrink-0 items-center gap-1.5 rounded-[0.7rem] border border-white/[0.12] px-3 text-xs text-[#d9e0ed] hover:bg-white/[0.05] disabled:opacity-40"
        >
          <Plus className="size-3.5" />
          Add
        </button>
      </div>
    </div>
  );
}
