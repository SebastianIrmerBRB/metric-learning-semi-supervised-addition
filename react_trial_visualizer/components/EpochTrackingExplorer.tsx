"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ArrowLeft, ChevronDown, Film, GitGraph, Info } from "lucide-react";
import {
  METRIC_GROUP_BLURBS,
  METRIC_GROUP_LABELS,
  deltaDirection,
  epochExtent,
  formatDelta,
  formatValue,
  scale,
  seriesExtent,
  ticks,
  type EpochTrackingRun,
  type EpochTrackingSnapshot,
  type SeriesTuple,
} from "@/lib/epoch-tracking";

const COLOR = {
  grid: "#27334c",
  axis: "#77859f",
  line: "#9b8cff",
  band: "rgba(155, 140, 255, 0.16)",
  baseline: "#f4c86f",
};

const CHART = { width: 520, height: 210, top: 16, right: 18, bottom: 34, left: 54 };
const INNER_WIDTH = CHART.width - CHART.left - CHART.right;
const INNER_HEIGHT = CHART.height - CHART.top - CHART.bottom;

function projectionLabel(layout: EpochTrackingRun["layout"]): string | null {
  if (layout === "tsne") return "t-SNE";
  if (layout === "pacmap") return "PaCMAP";
  if (layout === "pca") return "PCA";
  return null;
}

function MetricChart({
  points,
  label,
  baselineEpoch,
}: {
  points: SeriesTuple[];
  label: string;
  baselineEpoch: number;
}) {
  if (points.length === 0) return null;
  const xExtent = epochExtent([points]);
  const hasBand = points.some(([, mean, low, high]) => low !== mean || high !== mean);
  const yExtent = seriesExtent([points], hasBand);

  const x = (epoch: number) => scale(epoch, xExtent, CHART.left, CHART.left + INNER_WIDTH);
  const y = (value: number) => scale(value, yExtent, CHART.top + INNER_HEIGHT, CHART.top);

  const linePath = points
    .map(([epoch, mean], index) => `${index === 0 ? "M" : "L"}${x(epoch)},${y(mean)}`)
    .join(" ");
  const bandPath = hasBand
    ? [
        ...points.map(([epoch, , low], index) => `${index === 0 ? "M" : "L"}${x(epoch)},${y(low)}`),
        ...[...points].reverse().map(([epoch, , , high]) => `L${x(epoch)},${y(high)}`),
        "Z",
      ].join(" ")
    : null;

  const showsBaseline = xExtent.min <= baselineEpoch;

  return (
    <figure className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-3">
      <figcaption className="mb-1 px-1 text-xs font-semibold text-[#c8d2e4]">{label}</figcaption>
      <svg viewBox={`0 0 ${CHART.width} ${CHART.height}`} className="w-full" role="img" aria-label={label}>
        {ticks(yExtent, 4).map((value) => (
          <g key={`y-${value}`}>
            <line
              x1={CHART.left}
              x2={CHART.left + INNER_WIDTH}
              y1={y(value)}
              y2={y(value)}
              stroke={COLOR.grid}
              strokeWidth={1}
            />
            <text x={CHART.left - 8} y={y(value) + 3} textAnchor="end" fontSize={9} fill={COLOR.axis}>
              {formatValue(value, Math.abs(value) >= 10 ? 1 : 3)}
            </text>
          </g>
        ))}
        {points.map(([epoch]) => (
          <text
            key={`x-${epoch}`}
            x={x(epoch)}
            y={CHART.top + INNER_HEIGHT + 16}
            textAnchor="middle"
            fontSize={9}
            fill={COLOR.axis}
          >
            {epoch}
          </text>
        ))}
        <text
          x={CHART.left + INNER_WIDTH / 2}
          y={CHART.height - 4}
          textAnchor="middle"
          fontSize={9}
          fill={COLOR.axis}
        >
          Epoch
        </text>
        {showsBaseline ? (
          <line
            x1={x(baselineEpoch)}
            x2={x(baselineEpoch)}
            y1={CHART.top}
            y2={CHART.top + INNER_HEIGHT}
            stroke={COLOR.baseline}
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.65}
          />
        ) : null}
        {bandPath ? <path d={bandPath} fill={COLOR.band} /> : null}
        <path d={linePath} fill="none" stroke={COLOR.line} strokeWidth={2} />
        {points.map(([epoch, mean]) => (
          <circle
            key={`p-${epoch}`}
            cx={x(epoch)}
            cy={y(mean)}
            r={epoch === baselineEpoch ? 4 : 2.8}
            fill={epoch === baselineEpoch ? "none" : COLOR.line}
            stroke={epoch === baselineEpoch ? COLOR.baseline : "none"}
            strokeWidth={1.6}
          />
        ))}
      </svg>
    </figure>
  );
}

function SummaryTable({
  run,
  metrics,
  higherIsBetter,
  labels,
}: {
  run: EpochTrackingRun;
  metrics: string[];
  higherIsBetter: Record<string, boolean>;
  labels: Record<string, string>;
}) {
  const rows = run.summary.filter((row) => metrics.includes(row.metric));
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-[#7f8ba2]">
            <th className="px-3 py-2 text-left font-semibold">Metric</th>
            <th className="px-3 py-2 text-right font-semibold">Baseline</th>
            <th className="px-3 py-2 text-right font-semibold">Final</th>
            <th className="px-3 py-2 text-right font-semibold">Delta</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const direction = deltaDirection(row.metric, row.delta, higherIsBetter);
            const tone =
              direction === "better"
                ? "text-[#b8e56e]"
                : direction === "worse"
                  ? "text-[#f08a7a]"
                  : "text-[#98a4ba]";
            return (
              <tr key={row.metric} className="border-t border-white/[0.06]">
                <td className="px-3 py-2 text-[#c8d2e4]">{labels[row.metric] ?? row.label}</td>
                <td className="px-3 py-2 text-right tabular-nums text-[#98a4ba]">
                  {formatValue(row.baseline)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-white">{formatValue(row.final)}</td>
                <td className={`px-3 py-2 text-right font-semibold tabular-nums ${tone}`}>
                  {formatDelta(row.delta)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function EmptyState() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>
      <h1 className="mt-6 text-2xl font-semibold text-white">No epoch-tracking snapshot yet</h1>
      <p className="mt-3 text-sm leading-relaxed text-[#98a4ba]">
        This view reads <code className="rounded bg-white/[0.06] px-1.5 py-0.5">public/epoch-tracking.json</code>,
        which is written by the exporter. Train a run with tracking enabled, then export it:
      </p>
      <pre className="mt-4 overflow-x-auto rounded-xl border border-white/[0.07] bg-black/40 p-4 text-[11px] leading-relaxed text-[#c8d2e4]">
{`python main.py ... \\
  --evaluate_test \\
  --track_epoch_artifacts graph embeddings test \\
  --track_epoch_interval 3

python evaluate/export_react.py --root logs/<your run>`}
      </pre>
      <p className="mt-4 text-xs text-[#7f8ba2]">See docs/epoch_tracking.md for the full option list.</p>
    </main>
  );
}

export function EpochTrackingExplorer({ snapshot }: { snapshot: EpochTrackingSnapshot | null }) {
  const [selectedId, setSelectedId] = useState<string | null>(snapshot?.runs[0]?.id ?? null);

  const run = useMemo(
    () => snapshot?.runs.find((candidate) => candidate.id === selectedId) ?? snapshot?.runs[0] ?? null,
    [snapshot, selectedId],
  );

  if (!snapshot || !run) return <EmptyState />;

  const groups = Object.entries(snapshot.metricGroups).filter(([, metrics]) => metrics.length > 0);

  return (
    <main className="mx-auto max-w-[1180px] px-5 pb-20 pt-8">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>

      <header className="mt-5">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold text-white">Epoch tracking</h1>
          <span className="rounded-full bg-[#4fd5e7]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#7fe3f0]">
            {snapshot.runCount} run{snapshot.runCount === 1 ? "" : "s"} · {snapshot.foldCount} fold
            {snapshot.foldCount === 1 ? "" : "s"}
          </span>
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#98a4ba]">
          Structure across the epochs of a fold, starting at the epoch{" "}
          <span className="font-semibold text-[#f4c86f]">-1</span> baseline — before any optimizer step.
          Without that baseline a curve cannot separate structure the SSL phase built from structure the
          pretrained backbone already had.
        </p>
        <p className="mt-2 flex items-center gap-2 text-[11px] text-[#7f8ba2]">
          <Info size={12} />
          Measured off the selection path: nothing here was read back into training, early stopping, or Optuna.
        </p>
      </header>

      <div className="mt-6 max-w-3xl">
        <label
          htmlFor="epoch-tracking-run"
          className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]"
        >
          Tracking run
        </label>
        <div className="relative mt-1.5">
          <select
            id="epoch-tracking-run"
            value={run.id}
            onChange={(event) => setSelectedId(event.target.value)}
            className="focus-ring h-11 w-full appearance-none rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 pr-10 text-xs font-semibold text-[#d9e0ed]"
          >
            {snapshot.runs.map((candidate) => (
              <option key={candidate.id} value={candidate.id}>
                {candidate.label} — {candidate.id}
              </option>
            ))}
          </select>
          <ChevronDown
            aria-hidden="true"
            size={15}
            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#7f8ba2]"
          />
        </div>
        <p className="mt-1.5 text-[11px] text-[#65748e]">
          {snapshot.runs.length === 1
            ? "Only one run is present in this snapshot. Export a broader root to add the others."
            : `${snapshot.runs.length} runs are available in this snapshot.`}
        </p>
      </div>

      <section className="mt-6 rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
        <h2 className="text-sm font-semibold text-white">{run.label}</h2>
        <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 text-xs sm:grid-cols-4">
          {[
            ["Dataset", run.dataset],
            ["Method", run.method],
            ["Loss", run.loss],
            ["Seed", run.seed == null ? null : String(run.seed)],
            ["Projection", projectionLabel(run.layout)],
            ["Folds", run.folds.length === 1 && run.folds[0] === -1 ? "single run" : String(run.folds.length)],
            ["Tracked epochs", run.epochs.join(", ")],
          ]
            .filter(([, value]) => value != null && value !== "")
            .map(([label, value]) => (
              <div key={String(label)}>
                <dt className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[#65748e]">{label}</dt>
                <dd className="mt-0.5 text-[#c8d2e4]">{value}</dd>
              </div>
            ))}
        </dl>
      </section>

      {groups.map(([group, metrics]) => (
        <section key={group} className="mt-9">
          <h2 className="text-base font-semibold text-white">
            {METRIC_GROUP_LABELS[group] ?? group}
          </h2>
          {METRIC_GROUP_BLURBS[group] ? (
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-[#8895aa]">
              {METRIC_GROUP_BLURBS[group]}
            </p>
          ) : null}
          <div className="mt-3 rounded-2xl border border-white/[0.07] bg-white/[0.02]">
            <SummaryTable
              run={run}
              metrics={metrics}
              higherIsBetter={snapshot.higherIsBetter}
              labels={snapshot.metricLabels}
            />
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {metrics
              .filter((metric) => (run.series[metric]?.length ?? 0) > 0)
              .map((metric) => (
                <MetricChart
                  key={metric}
                  points={run.series[metric]}
                  label={snapshot.metricLabels[metric] ?? metric}
                  baselineEpoch={snapshot.baselineEpoch}
                />
              ))}
          </div>
        </section>
      ))}

      {snapshot.overviewFigures.length > 0 ? (
        <section className="mt-10">
          <h2 className="flex items-center gap-2 text-base font-semibold text-white">
            <GitGraph size={16} className="text-[#9b8cff]" />
            Across runs
          </h2>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            {snapshot.overviewFigures.map((figure) => (
              <figure key={figure.name} className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/${figure.file}`} alt={figure.caption} className="w-full rounded-lg" loading="lazy" />
                <figcaption className="mt-2 px-1 text-[11px] leading-relaxed text-[#8895aa]">
                  {figure.caption}
                </figcaption>
              </figure>
            ))}
          </div>
        </section>
      ) : null}

      {run.filmstrips.length > 0 ? (
        <section className="mt-10">
          <h2 className="flex items-center gap-2 text-base font-semibold text-white">
            <Film size={16} className="text-[#4fd5e7]" />
            Filmstrips
          </h2>
          <p className="mt-1 max-w-3xl text-xs leading-relaxed text-[#8895aa]">
            The per-epoch pictures in order — epoch -1 at the left, the trained model at the right. Each
            fold is shown separately; scroll a filmstrip sideways.
          </p>
          <div className="mt-4 space-y-7">
            {run.filmstrips.map((strip) => {
              const foldLabel = strip.fold === -1 ? "Single run" : `Fold ${strip.fold}`;
              return (
                <section key={strip.fold}>
                  <h3 className="mb-2 text-xs font-semibold text-[#c8d2e4]">{foldLabel}</h3>
                  <div className="space-y-4">
                    {(["graph", "embeddings"] as const)
                      .filter((kind) => strip[kind])
                      .map((kind) => (
                        <figure
                          key={kind}
                          className="overflow-x-auto rounded-2xl border border-white/[0.07] bg-white/[0.02] p-3"
                        >
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={`/${strip[kind]}`}
                            alt={`${kind} across epochs for ${foldLabel.toLowerCase()}`}
                            className="h-[300px] max-w-none rounded-lg"
                            loading="lazy"
                          />
                          <figcaption className="mt-2 px-1 text-[11px] capitalize text-[#8895aa]">
                            {kind}
                            {projectionLabel(run.layout) ? ` · ${projectionLabel(run.layout)}` : ""}
                          </figcaption>
                        </figure>
                      ))}
                  </div>
                </section>
              );
            })}
          </div>
        </section>
      ) : null}

      <footer className="mt-12 border-t border-white/[0.07] pt-4 text-[11px] text-[#65748e]">
        Snapshot generated {new Date(snapshot.generatedAt).toLocaleString()} from{" "}
        <code className="rounded bg-white/[0.06] px-1.5 py-0.5">{snapshot.root}</code> · rebuild with{" "}
        <code className="rounded bg-white/[0.06] px-1.5 py-0.5">npm run sync-epoch-tracking</code>
      </footer>
    </main>
  );
}
