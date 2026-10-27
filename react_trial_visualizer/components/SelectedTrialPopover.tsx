"use client";

import { Activity, BarChart3, Clock3, Gauge, Minus, MoveDown, Plus, X } from "lucide-react";
import {
  columnLabel,
  formatDuration,
  formatMetric,
  formatNumber,
  numericValue,
  type Study,
  type TrialRow,
} from "@/lib/trials";

interface SelectedTrialPopoverProps {
  study: Study;
  row: TrialRow;
  includedInCurves: boolean;
  onClose: () => void;
  onToggleCurve: () => void;
  onViewCurves: () => void;
}

function Detail({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-white/[0.07] bg-black/10 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-[#6f7d96]">
        {icon} {label}
      </div>
      <p className="mt-1.5 truncate text-sm font-semibold tabular-nums text-[#e7ebf4]">{value}</p>
    </div>
  );
}

export function SelectedTrialPopover({ study, row, includedInCurves, onClose, onToggleCurve, onViewCurves }: SelectedTrialPopoverProps) {
  const state = study.config.state ? row[study.config.state] ?? "Unknown" : "Complete";
  const duration = study.config.duration ? numericValue(row[study.config.duration]) : null;

  return (
    <aside
      className="fade-up fixed left-4 top-20 z-40 w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-[20px] border border-[#9b8cff]/20 bg-[#111a2e]/95 shadow-[0_24px_80px_rgba(0,0,0,.42)] backdrop-blur-xl lg:left-[306px]"
      aria-label={`Selected trial ${row[study.config.trial] ?? ""} summary`}
    >
      <div className="flex items-start justify-between border-b border-white/[0.07] px-4 py-3.5">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#9b8cff]/12 text-[#b7adff] ring-1 ring-[#9b8cff]/20">
            <Gauge size={17} />
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-white">Trial #{row[study.config.trial] ?? "—"}</h2>
              <span className="rounded-full border border-[#b8e56e]/20 bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.09em] text-[#c9f18a]">
                {state}
              </span>
            </div>
            <p className="mt-1 truncate text-[10px] text-[#71809a]">{study.name} · {study.dataset}</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="focus-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[#75839a] transition hover:bg-white/[0.06] hover:text-white"
          aria-label="Close selected trial summary"
        >
          <X size={15} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 p-3.5">
        <Detail
          label="mAP@R"
          value={formatMetric(numericValue(row[study.config.metric]))}
          icon={<BarChart3 size={11} />}
        />
        <Detail
          label={columnLabel(study.config.epoch)}
          value={study.config.epoch ? formatNumber(numericValue(row[study.config.epoch]), 7) : "—"}
          icon={<Activity size={11} />}
        />
        <Detail
          label="Selected epoch"
          value={study.config.selectedEpoch ? formatNumber(numericValue(row[study.config.selectedEpoch]), 7) : "—"}
          icon={<MoveDown size={11} />}
        />
        <Detail
          label="Runtime"
          value={formatDuration(duration)}
          icon={<Clock3 size={11} />}
        />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.07] px-4 py-3">
        <p className="max-w-48 truncate text-[10px] text-[#6d7a91]" title={study.config.learningRate ? row[study.config.learningRate] ?? undefined : undefined}>
          LR {study.config.learningRate ? formatNumber(numericValue(row[study.config.learningRate])) : "not detected"}
        </p>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onToggleCurve}
            className={`focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[10px] font-semibold transition ${includedInCurves ? "bg-[#4fd5e7]/10 text-[#78e1ec] hover:bg-[#4fd5e7]/18" : "bg-[#9b8cff]/12 text-[#c2baff] hover:bg-[#9b8cff]/20"}`}
          >
            {includedInCurves ? <Minus size={12} /> : <Plus size={12} />}
            {includedInCurves ? "Remove" : "Add to curves"}
          </button>
          <button
            type="button"
            onClick={onViewCurves}
            className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg bg-white/[0.045] px-2.5 text-[10px] font-semibold text-[#b4bed0] transition hover:bg-white/[0.08] hover:text-white"
          >
            View <MoveDown size={12} />
          </button>
        </div>
      </div>
    </aside>
  );
}
