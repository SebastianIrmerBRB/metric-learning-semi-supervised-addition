"use client";

import { useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";

import {
  ISSUE_LABEL,
  PHASES,
  PHASE_LABEL,
  datasetLabel,
  sortDatasets,
  type ConfigPhase,
  type Issue,
  type IssueType,
} from "@/lib/study-configs";
import { Empty, ISSUE_STYLE, Note, Select } from "@/components/configs/ui";

const TYPE_ORDER: IssueType[] = ["replay", "series", "deviation", "cap", "incomplete", "missing"];
const TYPE_HINT: Record<IssueType, string> = {
  replay: "A seeded or Phase-2 replay did not use the linked Phase-1 study's best trial, or its runs mix trials.",
  series: "The runs of one folder were started with different settings (other than seed and label budget).",
  deviation: "Within one phase and dataset, a protocol setting differs from what most studies used.",
  cap: "Folds trained until the last epoch, so the epoch budget stopped them, not early stopping.",
  incomplete: "Fewer trials or runs finished than were planned.",
  missing: "A folder without any config, or a Phase-2 folder a manifest plans but F: does not have.",
};

export type IssueTarget = "compare" | "trace" | "coverage";

export function ChecksView({ issues, onOpen }: { issues: Issue[]; onOpen: (issue: Issue, target: IssueTarget) => void }) {
  const [types, setTypes] = useState<Set<IssueType>>(() => new Set(TYPE_ORDER.filter((type) => type !== "incomplete")));
  const [phase, setPhase] = useState<ConfigPhase | "*">("*");
  const [dataset, setDataset] = useState<string>("*");
  const [query, setQuery] = useState("");

  const counts = useMemo(() => {
    const result = Object.fromEntries(TYPE_ORDER.map((type) => [type, 0])) as Record<IssueType, number>;
    for (const issue of issues) {
      if ((phase === "*" || issue.phase === phase) && (dataset === "*" || issue.dataset === dataset)) result[issue.type] += 1;
    }
    return result;
  }, [issues, phase, dataset]);

  const datasets = useMemo(() => sortDatasets(issues.map((issue) => issue.dataset)), [issues]);
  const needle = query.trim().toLowerCase();
  const visible = issues.filter(
    (issue) =>
      types.has(issue.type) &&
      (phase === "*" || issue.phase === phase) &&
      (dataset === "*" || issue.dataset === dataset) &&
      (!needle || `${issue.title} ${issue.detail} ${issue.entries.join(" ")}`.toLowerCase().includes(needle)),
  );

  const grouped = useMemo(() => {
    const groups: { phase: ConfigPhase; items: Issue[] }[] = [];
    for (const current of PHASES) {
      const items = visible
        .filter((issue) => issue.phase === current)
        .sort(
          (left, right) =>
            TYPE_ORDER.indexOf(left.type) - TYPE_ORDER.indexOf(right.type) ||
            datasets.indexOf(left.dataset) - datasets.indexOf(right.dataset) ||
            left.title.localeCompare(right.title),
        );
      if (items.length) groups.push({ phase: current, items });
    }
    return groups;
  }, [visible, datasets]);

  return (
    <div className="mt-6 space-y-5">
      <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {TYPE_ORDER.map((type) => {
          const style = ISSUE_STYLE[type];
          const Icon = style.icon;
          const active = types.has(type);
          return (
            <button
              key={type}
              type="button"
              aria-pressed={active}
              title={TYPE_HINT[type]}
              onClick={() =>
                setTypes((current) => {
                  const next = new Set(current);
                  if (next.has(type)) next.delete(type);
                  else next.add(type);
                  return next;
                })
              }
              className={`focus-ring rounded-xl border px-3 py-2.5 text-left transition ${
                active ? `${style.border} ${style.bg}` : "border-white/[0.06] bg-white/[0.015] opacity-60 hover:opacity-100"
              }`}
            >
              <div className={`flex items-center gap-1.5 text-[11px] font-semibold ${style.text}`}>
                <Icon size={13} aria-hidden />
                {ISSUE_LABEL[type]}
              </div>
              <div className="mt-1 text-xl font-semibold tabular-nums text-white">{counts[type]}</div>
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap items-end gap-3">
        <Select<ConfigPhase | "*">
          label="Phase"
          value={phase}
          options={["*", ...PHASES]}
          onChange={setPhase}
          render={(value) => (value === "*" ? "All phases" : PHASE_LABEL[value])}
        />
        <Select label="Dataset" value={dataset} options={["*", ...datasets]} onChange={setDataset} render={(value) => (value === "*" ? "All datasets" : datasetLabel(value))} />
        <label className="flex min-w-[16rem] flex-1 flex-col gap-1 text-xs">
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Search</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="patience, LRML, Cars196/SLADE, …"
            className="focus-ring rounded-lg border border-white/[0.08] bg-[#0f1522] px-2.5 py-1.5 text-xs text-white placeholder:text-[#56627a]"
          />
        </label>
        <span className="pb-1.5 text-[11px] tabular-nums text-[#7f8ba2]">{visible.length} shown</span>
      </div>

      <Note>
        Checks compare what the runs recorded, not what the configs in the repository say today. Click a card above to show or
        hide that kind; each finding opens the table that shows it.
      </Note>

      {grouped.length === 0 && <Empty>Nothing to report for this selection.</Empty>}

      {grouped.map((group) => (
        <section key={group.phase}>
          <h2 className="mb-2 text-sm font-semibold text-white">
            {PHASE_LABEL[group.phase]} <span className="font-normal text-[#7f8ba2]">· {group.items.length}</span>
          </h2>
          <ul className="divide-y divide-white/[0.05] overflow-hidden rounded-2xl border border-white/[0.07] bg-white/[0.015]">
            {group.items.map((issue) => {
              const style = ISSUE_STYLE[issue.type];
              const Icon = style.icon;
              const plan = issue.id.startsWith("missing-plan|");
              return (
                <li key={issue.id} className="flex flex-wrap items-start gap-3 px-4 py-3">
                  <span className={`mt-0.5 flex w-[9.5rem] shrink-0 items-center gap-1.5 text-[10.5px] font-semibold ${style.text}`}>
                    <Icon size={13} aria-hidden />
                    {ISSUE_LABEL[issue.type]}
                  </span>
                  <div className="min-w-[16rem] flex-1">
                    <p className="text-[12.5px] font-semibold leading-snug text-[#e4e9f2]">
                      <span className="mr-1.5 rounded bg-white/[0.06] px-1.5 py-px text-[10px] font-semibold text-[#aeb9cc]">
                        {datasetLabel(issue.dataset)}
                      </span>
                      {issue.title}
                    </p>
                    <p className="mt-1 text-[11.5px] leading-relaxed text-[#98a4ba]">{issue.detail}</p>
                  </div>
                  <div className="flex shrink-0 gap-1.5 text-[11px]">
                    {plan ? (
                      <OpenButton onClick={() => onOpen(issue, "coverage")}>Coverage</OpenButton>
                    ) : (
                      <>
                        {(issue.type === "deviation" || issue.entries.length > 0) && (
                          <OpenButton onClick={() => onOpen(issue, "compare")}>Compare</OpenButton>
                        )}
                        {issue.entries.length > 0 && issue.type !== "deviation" && (
                          <OpenButton onClick={() => onOpen(issue, "trace")}>Trace</OpenButton>
                        )}
                      </>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}

function OpenButton({ onClick, children }: { onClick: () => void; children: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="focus-ring inline-flex items-center gap-0.5 rounded-lg border border-[#9b8cff]/30 bg-[#9b8cff]/10 px-2 py-1 text-[11px] font-semibold text-[#d8d2ff] hover:bg-[#9b8cff]/20"
    >
      {children}
      <ChevronRight size={12} aria-hidden />
    </button>
  );
}
