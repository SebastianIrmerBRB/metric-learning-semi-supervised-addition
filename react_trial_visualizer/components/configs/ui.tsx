"use client";

import type { ReactNode } from "react";
import { CircleCheck, CircleDashed, CircleX, Link2Off, Shuffle, Timer, TriangleAlert } from "lucide-react";

import { METHOD_COLOR, methodLabel, type IssueType } from "@/lib/study-configs";

export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
  render,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (value: T) => void;
  render?: (value: T) => string;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</span>
      <select
        className="focus-ring min-w-[7rem] rounded-lg border border-white/[0.08] bg-[#0f1522] px-2.5 py-1.5 text-xs text-white"
        value={value}
        onChange={(event) => {
          const next = options.find((option) => option === event.target.value);
          if (next !== undefined) onChange(next);
        }}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {render ? render(option) : option}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Toggle<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (value: T) => void;
  label?: string;
}) {
  const control = (
    <div className="inline-flex flex-wrap rounded-lg border border-white/[0.08] bg-[#0f1522] p-0.5 text-[11px] font-semibold" role="group">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.hint}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={`focus-ring rounded-md px-2.5 py-1 text-[11px] font-semibold transition ${
            value === option.value ? "bg-white/[0.09] text-white" : "text-[#8895aa] hover:text-white"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
  if (!label) return control;
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</span>
      {control}
    </div>
  );
}

export function Checkbox({ checked, onChange, children }: { checked: boolean; onChange: (checked: boolean) => void; children: ReactNode }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-[11px] font-semibold text-[#c8d2e4]">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} className="accent-[#9b8cff]" />
      {children}
    </label>
  );
}

export function MethodName({ method, className = "" }: { method: string; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: METHOD_COLOR[method] ?? "#c8d2e4" }} aria-hidden />
      {methodLabel(method)}
    </span>
  );
}

export const ISSUE_STYLE: Record<IssueType, { icon: typeof CircleCheck; text: string; bg: string; border: string }> = {
  replay: { icon: Link2Off, text: "text-[#f4b4b4]", bg: "bg-[#d03b3b]/10", border: "border-[#d03b3b]/35" },
  series: { icon: Shuffle, text: "text-[#f4b4b4]", bg: "bg-[#d03b3b]/10", border: "border-[#d03b3b]/35" },
  deviation: { icon: TriangleAlert, text: "text-[#f6d488]", bg: "bg-[#f4c86f]/10", border: "border-[#f4c86f]/30" },
  cap: { icon: Timer, text: "text-[#f6d488]", bg: "bg-[#f4c86f]/10", border: "border-[#f4c86f]/30" },
  incomplete: { icon: CircleDashed, text: "text-[#7fe3f0]", bg: "bg-[#4fd5e7]/10", border: "border-[#4fd5e7]/25" },
  missing: { icon: CircleX, text: "text-[#f4b4b4]", bg: "bg-[#d03b3b]/10", border: "border-[#d03b3b]/35" },
};

export type Status = "ok" | "warn" | "partial" | "missing" | "none" | "extra";

const STATUS_STYLE: Record<Status, { icon: typeof CircleCheck | null; text: string; box: string }> = {
  ok: { icon: CircleCheck, text: "text-[#c7ef85]", box: "border-[#b8e56e]/25 bg-[#b8e56e]/[0.07]" },
  warn: { icon: TriangleAlert, text: "text-[#f6d488]", box: "border-[#f4c86f]/30 bg-[#f4c86f]/[0.08]" },
  partial: { icon: CircleDashed, text: "text-[#7fe3f0]", box: "border-[#4fd5e7]/25 bg-[#4fd5e7]/[0.07]" },
  missing: { icon: CircleX, text: "text-[#f4b4b4]", box: "border-[#d03b3b]/40 bg-[#d03b3b]/[0.1]" },
  none: { icon: null, text: "text-[#56627a]", box: "border-dashed border-white/[0.08]" },
  extra: { icon: null, text: "text-[#b6acff]", box: "border-[#9b8cff]/30 bg-[#9b8cff]/[0.07]" },
};

/** A status always carries an icon and a word, never colour alone. */
export function StatusChip({ status, children, title }: { status: Status; children: ReactNode; title?: string }) {
  const style = STATUS_STYLE[status];
  const Icon = style.icon;
  return (
    <span
      title={title}
      className={`inline-flex min-h-[1.5rem] items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10.5px] font-semibold tabular-nums ${style.text} ${style.box}`}
    >
      {Icon && <Icon size={11} className="shrink-0" aria-hidden />}
      {children}
    </span>
  );
}

export function StatusLegend({ items }: { items: { status: Status; label: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#8895aa]">
      {items.map((item) => (
        <StatusChip key={item.label} status={item.status}>
          {item.label}
        </StatusChip>
      ))}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-xl border border-dashed border-white/[0.1] px-4 py-6 text-center text-xs text-[#7f8ba2]">{children}</p>;
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="max-w-4xl text-xs leading-relaxed text-[#8895aa]">{children}</p>;
}
