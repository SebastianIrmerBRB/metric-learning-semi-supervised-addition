"use client";

import { useMemo, useState } from "react";

import {
  datasetLabel,
  formatNumber,
  formatSpace,
  formatValue,
  lossLabel,
  methodLabel,
  sortDatasets,
  sortLosses,
  sortMethods,
  spaceKey,
  valueKey,
  type ConfigEntry,
  type SettingValue,
  type SpaceParam,
} from "@/lib/study-configs";
import { ALL, type Filters } from "@/components/configs/filters";
import { Checkbox, Empty, Note, Select, Toggle } from "@/components/configs/ui";

type HpoPhase = "phase0" | "phase1";

interface SpecGroup {
  spec: SpaceParam;
  entries: ConfigEntry[];
}

interface Hover {
  x: number;
  y: number;
  lines: string[];
}

const BAR_WIDTH = 150;

/** Shared horizontal scale of one parameter row, logarithmic when every range is. */
function rowScale(specs: SpaceParam[], values: number[]) {
  const numeric = specs.filter((spec) => spec.low != null && spec.high != null);
  const lows = [...numeric.map((spec) => spec.low as number), ...values];
  const highs = [...numeric.map((spec) => spec.high as number), ...values];
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const log = numeric.length > 0 && numeric.every((spec) => spec.log) && min > 0;
  const toUnit = (value: number) => {
    if (max === min) return 0.5;
    return log ? (Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min)) : (value - min) / (max - min);
  };
  return { min, max, log, x: (value: number) => 4 + toUnit(value) * (BAR_WIDTH - 8) };
}

function groupSpecs(entries: ConfigEntry[], param: string): SpecGroup[] {
  const groups = new Map<string, SpecGroup>();
  for (const entry of entries) {
    const spec = entry.space?.[param];
    if (!spec) continue;
    const key = spaceKey(spec);
    const group = groups.get(key);
    if (group) group.entries.push(entry);
    else groups.set(key, { spec, entries: [entry] });
  }
  return [...groups.values()];
}

function bestValue(entry: ConfigEntry, param: string): SettingValue | undefined {
  return entry.hpo?.best?.params[param];
}

/**
 * Phase-1 search spaces were derived from Phase 0, so the page can lay the
 * Phase-0 range underneath each Phase-1 bar and show how far it narrowed.
 */
export function SearchSpaceView({ entries, filters, setFilters }: { entries: ConfigEntry[]; filters: Filters; setFilters: (next: Partial<Filters>) => void }) {
  const [phase, setPhase] = useState<HpoPhase>(filters.phase === "phase0" ? "phase0" : "phase1");
  const [overlay, setOverlay] = useState(true);
  const [hover, setHover] = useState<Hover | null>(null);

  const hpo = useMemo(() => entries.filter((entry) => entry.phase === phase && entry.space && Object.keys(entry.space).length), [entries, phase]);
  const methods = sortMethods(hpo.map((entry) => entry.method));
  const method = methods.includes(filters.method) ? filters.method : methods.includes("LP") ? "LP" : methods[0] ?? "";
  const inMethod = hpo.filter((entry) => entry.method === method);
  const losses = sortLosses(inMethod.map((entry) => entry.loss));
  const loss = losses.includes(filters.loss) ? filters.loss : ALL;
  const selected = inMethod.filter((entry) => loss === ALL || entry.loss === loss);
  const datasets = sortDatasets(selected.map((entry) => entry.dataset));
  const params = useMemo(() => {
    const all = [...new Set(selected.flatMap((entry) => Object.keys(entry.space ?? {})))];
    const rank = (param: string) => (param.startsWith("ssl_config.") ? 2 : param.startsWith("loss.") || param.startsWith("miner.") ? 1 : 0);
    return all.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  }, [selected]);

  const phase0 = useMemo(
    () => entries.filter((entry) => entry.phase === "phase0" && entry.method === method && (loss === ALL || entry.loss === loss) && entry.space),
    [entries, method, loss],
  );
  const showOverlay = phase === "phase1" && overlay;

  return (
    <div className="mt-6 space-y-4" onMouseLeave={() => setHover(null)}>
      <div className="flex flex-wrap items-end gap-3">
        <Toggle<HpoPhase>
          label="Phase"
          value={phase}
          onChange={setPhase}
          options={[
            { value: "phase0", label: "Phase 0 · random" },
            { value: "phase1", label: "Phase 1 · TPE" },
          ]}
        />
        <Select label="Method" value={method} options={methods} onChange={(value) => setFilters({ method: value })} render={methodLabel} />
        <Select label="Loss" value={loss} options={[ALL, ...losses]} onChange={(value) => setFilters({ loss: value })} render={(value) => (value === ALL ? "All losses" : lossLabel(value))} />
        {phase === "phase1" && (
          <div className="pb-1.5">
            <Checkbox checked={overlay} onChange={setOverlay}>
              Phase-0 range underneath
            </Checkbox>
          </div>
        )}
      </div>
      <Note>
        One row per searched parameter, one column per dataset, on a scale shared across the row (logarithmic when every range is).
        The violet bar is the searched range, the white tick the best trial&apos;s value
        {showOverlay ? ", the grey band behind it the Phase-0 range of the same study" : ""}. Ranges that differ from the row&apos;s most
        common one are shaded; with all losses selected, a column that searched different ranges per loss shows one bar per range.
      </Note>
      {datasets.length === 0 ? (
        <Empty>No studies with a search space for this selection.</Empty>
      ) : (
        <div className="surface overflow-x-auto studio-scroll">
          <table className="w-max min-w-full border-separate border-spacing-0 text-left">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 min-w-[17rem] bg-[#111a2d] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                  Parameter
                </th>
                {datasets.map((dataset) => (
                  <th key={dataset} className="min-w-[11rem] border-l border-white/[0.06] bg-[#111a2d] px-3 py-2 text-xs font-semibold text-white">
                    {datasetLabel(dataset)}
                    <div className="text-[10px] font-normal text-[#6f7c92]">
                      {selected.filter((entry) => entry.dataset === dataset).length} stud
                      {selected.filter((entry) => entry.dataset === dataset).length === 1 ? "y" : "ies"}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {params.map((param) => {
                const cells = datasets.map((dataset) => groupSpecs(selected.filter((entry) => entry.dataset === dataset), param));
                const overlays = datasets.map((dataset) => groupSpecs(phase0.filter((entry) => entry.dataset === dataset), param));
                const allSpecs = [...cells.flat(), ...(showOverlay ? overlays.flat() : [])].map((group) => group.spec);
                const bests = cells
                  .flat()
                  .flatMap((group) => group.entries.map((entry) => bestValue(entry, param)))
                  .filter((value): value is number => typeof value === "number");
                const numeric = allSpecs.some((spec) => !spec.choices);
                const scale = numeric ? rowScale(allSpecs.filter((spec) => !spec.choices), bests) : null;
                // A column differs when its ranges (which loss searched which) differ from most columns'.
                const signatures = cells.map((groups) =>
                  groups
                    .map((group) => `${spaceKey(group.spec)}:${group.entries.map((entry) => entry.loss).sort().join(",")}`)
                    .sort()
                    .join("|"),
                );
                const counts = new Map<string, number>();
                for (const signature of signatures) if (signature) counts.set(signature, (counts.get(signature) ?? 0) + 1);
                const modeSignature = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
                return (
                  <tr key={param}>
                    <th scope="row" className="sticky left-0 z-10 border-t border-white/[0.05] bg-[#0e1626] px-3 py-2 align-top font-normal">
                      <div className="font-mono text-[11px] text-[#e4e9f2]">{param}</div>
                      {scale && (
                        <div className="mt-0.5 text-[10px] text-[#6f7c92]">
                          axis {formatNumber(scale.min)} – {formatNumber(scale.max)}
                          {scale.log ? " · log" : ""}
                        </div>
                      )}
                    </th>
                    {cells.map((groups, index) => {
                      const differs = counts.size > 1 && signatures[index] !== "" && signatures[index] !== modeSignature;
                      return (
                        <td key={datasets[index]} className={`border-l border-t border-white/[0.05] px-3 py-2 align-top ${differs ? "bg-[#f4c86f]/[0.15] shadow-[inset_2px_0_0_#f4c86f]" : ""}`}>
                          {groups.length === 0 ? (
                            <span className="text-[11px] text-[#56627a]">not searched</span>
                          ) : (
                            <div className="space-y-1.5">
                              {groups.map((group) => (
                                <SpecCell
                                  key={spaceKey(group.spec)}
                                  group={group}
                                  param={param}
                                  scale={scale}
                                  overlay={showOverlay ? overlays[index] : []}
                                  labelLosses={groups.length > 1 || loss === ALL}
                                  onHover={setHover}
                                />
                              ))}
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {hover && (
        <div
          role="tooltip"
          className="pointer-events-none fixed z-50 max-w-xs rounded-lg border border-white/[0.12] bg-[#0b1222]/95 px-3 py-2 text-[11px] leading-relaxed text-[#e4e9f2] shadow-xl"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
        >
          {hover.lines.map((line, index) => (
            <div key={index} className={index === 0 ? "font-semibold text-white" : "text-[#aeb9cc]"}>
              {line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SpecCell({
  group,
  param,
  scale,
  overlay,
  labelLosses,
  onHover,
}: {
  group: SpecGroup;
  param: string;
  scale: ReturnType<typeof rowScale> | null;
  overlay: SpecGroup[];
  labelLosses: boolean;
  onHover: (hover: Hover | null) => void;
}) {
  const { spec } = group;
  const lossNames = group.entries.map((entry) => lossLabel(entry.loss)).join(", ");
  const bests = group.entries
    .map((entry) => ({ entry, value: bestValue(entry, param) }))
    .filter((item) => item.value !== undefined);
  const lines = [
    `${param} · ${formatSpace(spec)}`,
    `${group.entries.length} stud${group.entries.length === 1 ? "y" : "ies"}: ${lossNames}`,
    ...bests.map((item) => `best ${lossLabel(item.entry.loss)}: ${formatValue(item.value ?? null)} (trial #${item.entry.hpo?.best?.number ?? "?"})`),
    ...overlay.map((item) => `Phase 0: ${formatSpace(item.spec)}`),
  ];
  const onMove = (event: React.MouseEvent) => onHover({ x: event.clientX, y: event.clientY, lines });

  if (spec.choices) {
    const phase0Choices = new Set(overlay.flatMap((item) => (item.spec.choices ?? []).map((choice) => valueKey(choice))));
    const own = new Set(spec.choices.map((choice) => valueKey(choice)));
    const dropped = overlay.length ? [...phase0Choices].filter((choice) => !own.has(choice)) : [];
    const bestKeys = new Set(bests.map((item) => valueKey(item.value ?? null)));
    return (
      <div onMouseMove={onMove} onMouseLeave={() => onHover(null)}>
        <div className="flex flex-wrap gap-1">
          {spec.choices.map((choice) => (
            <span
              key={valueKey(choice)}
              className={`rounded px-1 py-px font-mono text-[10.5px] ${
                bestKeys.has(valueKey(choice)) ? "bg-white/[0.14] text-white ring-1 ring-white/40" : "bg-[#9b8cff]/15 text-[#d8d2ff]"
              }`}
            >
              {formatValue(choice)}
            </span>
          ))}
          {dropped.map((choice) => (
            <span key={choice} className="rounded px-1 py-px font-mono text-[10.5px] text-[#56627a] line-through" title="Searched in Phase 0 only">
              {formatValue(JSON.parse(choice) as SettingValue)}
            </span>
          ))}
        </div>
        {labelLosses && <div className="mt-0.5 text-[10px] text-[#6f7c92]">{lossNames}</div>}
      </div>
    );
  }

  if (!scale || spec.low == null || spec.high == null) {
    return <span className="font-mono text-[11px] text-[#c8d2e4]">{formatSpace(spec)}</span>;
  }
  const x0 = scale.x(spec.low);
  const x1 = scale.x(spec.high);
  return (
    <div onMouseMove={onMove} onMouseLeave={() => onHover(null)}>
      <svg width={BAR_WIDTH} height={14} viewBox={`0 0 ${BAR_WIDTH} 14`} className="block overflow-visible" aria-label={`${param} ${formatSpace(spec)}`}>
        <line x1={4} x2={BAR_WIDTH - 4} y1={7} y2={7} stroke="rgba(148,163,184,0.18)" strokeWidth={1} />
        {overlay
          .filter((item) => item.spec.low != null && item.spec.high != null)
          .map((item) => (
            <rect
              key={spaceKey(item.spec)}
              x={scale.x(item.spec.low as number)}
              y={2}
              width={Math.max(2, scale.x(item.spec.high as number) - scale.x(item.spec.low as number))}
              height={10}
              rx={3}
              fill="rgba(148,163,184,0.22)"
            />
          ))}
        <rect x={x0} y={4} width={Math.max(3, x1 - x0)} height={6} rx={3} fill="#9b8cff" />
        {bests
          .filter((item): item is { entry: ConfigEntry; value: number } => typeof item.value === "number")
          .map((item) => (
            <line key={item.entry.id} x1={scale.x(item.value)} x2={scale.x(item.value)} y1={1} y2={13} stroke="#ffffff" strokeWidth={2} strokeLinecap="round" />
          ))}
      </svg>
      <div className="mt-0.5 font-mono text-[10px] text-[#aeb9cc]">{formatSpace(spec)}</div>
      {labelLosses && <div className="text-[10px] text-[#6f7c92]">{lossNames}</div>}
    </div>
  );
}
