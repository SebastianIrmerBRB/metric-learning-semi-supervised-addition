"use client";

import { X } from "lucide-react";

import type { CliArgument, Issue } from "@/lib/studio/schema";

import { ChipList, Field, NumberInput, Select, TextInput, Toggle, parseScalar } from "./ui";

function parseForArgument(argument: CliArgument): (text: string) => unknown {
  if (argument.type === "int") return (text) => Math.trunc(Number(text));
  if (argument.type === "float") return (text) => Number(text);
  if (argument.type === "str" || argument.type === "path") return (text) => text;
  return parseScalar;
}

/** Render one main.py argument with the widget its argparse action implies. */
export function CliField({
  argument,
  value,
  onChange,
  issues,
  options,
  help,
}: {
  argument: CliArgument;
  value: unknown;
  onChange: (value: unknown) => void;
  issues?: Issue[];
  /** Overrides the choice list, used for the file-path arguments. */
  options?: { value: string; label?: string }[];
  help?: string;
}) {
  const label = argument.dest.replace(/_/g, " ");
  const description = help ?? argument.help;
  const defaultHint =
    argument.default === null || argument.default === undefined
      ? undefined
      : `default ${JSON.stringify(argument.default)}`;

  if (argument.isFlag) {
    return (
      <Field label={label} issues={issues}>
        <div className="pt-1">
          <Toggle
            checked={value === true}
            onChange={(next) => onChange(next)}
            label={description || argument.dest}
          />
        </div>
      </Field>
    );
  }

  const choiceOptions =
    options ?? (argument.choices ? argument.choices.map((choice) => ({ value: String(choice) })) : null);

  if (argument.isList) {
    const values = Array.isArray(value) ? value : [];
    if (choiceOptions) {
      return (
        <Field label={label} help={description} issues={issues}>
          <div className="flex flex-wrap gap-1.5">
            {values.map((item, index) => (
              <span key={`${String(item)}-${index}`} className="studio-chip">
                {String(item)}
                <button
                  type="button"
                  aria-label="Remove"
                  onClick={() => onChange(values.filter((_, position) => position !== index))}
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
              onChange={(next) => next && onChange([...values, next])}
              options={choiceOptions}
              placeholder="Add…"
            />
          </div>
        </Field>
      );
    }
    return (
      <Field label={label} help={description} issues={issues}>
        <ChipList values={values} onChange={onChange} parse={parseForArgument(argument)} />
      </Field>
    );
  }

  if (choiceOptions) {
    return (
      <Field label={label} help={description} hint={defaultHint} issues={issues}>
        <Select
          value={value === null || value === undefined ? null : String(value)}
          onChange={(next) => onChange(next)}
          options={choiceOptions}
          placeholder="unset"
        />
      </Field>
    );
  }

  if (argument.type === "int" || argument.type === "float") {
    return (
      <Field label={label} help={description} hint={defaultHint} issues={issues}>
        <NumberInput
          value={typeof value === "number" ? value : null}
          onChange={(next) => onChange(next)}
          placeholder={defaultHint ? String(argument.default) : "unset"}
        />
      </Field>
    );
  }

  if (typeof value === "boolean") {
    return (
      <Field label={label} help={description} issues={issues}>
        <div className="pt-1">
          <Toggle checked={value} onChange={(next) => onChange(next)} label={argument.dest} />
        </div>
      </Field>
    );
  }

  return (
    <Field label={label} help={description} hint={defaultHint} issues={issues}>
      <TextInput
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(next) => onChange(next === "" ? null : next)}
        placeholder={argument.default === null ? "null" : "unset"}
        mono={argument.type === "path"}
      />
    </Field>
  );
}
