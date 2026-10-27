"use client";

import { Plus, X } from "lucide-react";
import { useState } from "react";

import type { Issue, StudioSchema } from "@/lib/studio/schema";

import { Field, Select } from "./ui";

/** Build `Loss:Miner` pairs from the two real vocabularies instead of free text. */
export function LossMinerGrid({
  values,
  onChange,
  schema,
  issues,
}: {
  values: string[];
  onChange: (values: string[]) => void;
  schema: StudioSchema;
  issues?: Issue[];
}) {
  const [loss, setLoss] = useState<string | null>(null);
  const [miner, setMiner] = useState<string | null>("no_miner");

  const classification = loss ? schema.constants.classificationLosses.includes(loss) : false;
  const pending = loss ? `${loss}:${classification ? "no_miner" : miner ?? "no_miner"}` : null;

  return (
    <Field
      label="loss / miner grid"
      help="Each entry becomes its own comparison arm. A classification loss always pairs with no_miner."
      issues={issues}
      wide
    >
      <div className="flex flex-wrap gap-1.5">
        {values.map((entry, index) => {
          const [entryLoss, entryMiner] = entry.split(":", 2);
          const valid =
            schema.constants.losses.includes(entryLoss) && schema.constants.miners.includes(entryMiner);
          return (
            <span
              key={`${entry}-${index}`}
              className={`studio-chip ${valid ? "" : "border-[#ff9a9a]/50 text-[#ff9a9a]"}`}
            >
              {entry}
              <button
                type="button"
                aria-label={`Remove ${entry}`}
                onClick={() => onChange(values.filter((_, position) => position !== index))}
                className="focus-ring rounded-full text-[#8f9bb2] hover:text-[#f7f8fc]"
              >
                <X className="size-3" />
              </button>
            </span>
          );
        })}
        {values.length === 0 && <span className="text-xs text-[#8f9bb2]">No arms configured.</span>}
      </div>

      <div className="mt-2 grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
        <Select
          value={loss}
          onChange={setLoss}
          options={schema.constants.losses.map((value) => ({ value }))}
          placeholder="Loss…"
        />
        <Select
          value={classification ? "no_miner" : miner}
          onChange={setMiner}
          disabled={classification}
          allowEmpty={false}
          options={schema.constants.miners.map((value) => ({ value }))}
        />
        <button
          type="button"
          disabled={!pending || values.includes(pending)}
          onClick={() => pending && onChange([...values, pending])}
          className="focus-ring flex h-[2.35rem] items-center gap-1.5 rounded-[0.7rem] border border-white/[0.12] px-3 text-xs text-[#d9e0ed] hover:bg-white/[0.05] disabled:opacity-40"
        >
          <Plus className="size-3.5" />
          Add arm
        </button>
      </div>
    </Field>
  );
}
