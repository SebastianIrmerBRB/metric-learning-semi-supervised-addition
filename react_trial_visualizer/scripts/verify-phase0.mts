/**
 * Checks the Phase 0 range rules in lib/phase0.ts.
 *
 *   node --experimental-strip-types scripts/verify-phase0.mts
 *   npm run verify-phase0
 *
 * Three synthetic sweeps pin the decision rules to responses whose right answer
 * is known by construction (an optimum against the low edge, an optimum in the
 * interior, a response that never degrades). The rest run against the exported
 * archive, so the numbers behind the Range finder tab can be re-checked after
 * any change to the export or the rules.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_RANGE_SETTINGS,
  assessStudy,
  availableGradientSeries,
  collectSamples,
  estimateStartupTrials,
  findRange,
  fitNoiseModel,
  foldBreakdown,
  fromAxis,
  gradientFoldRows,
  mixedFoldCounts,
  niceBound,
  summariseEpochs,
  toAxis,
  toSpaces,
  type GradientIndex,
  type RangeSettings,
  type StudyAssessment,
} from "../lib/phase0.ts";
import type { ArchiveStudy } from "../lib/sorted-studies.ts";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const snapshotPath = path.join(projectRoot, "public", "sorted-studies.json");
const gradientPath = path.join(projectRoot, "public", "sorted-gradients.json");

let failures = 0;
const check = (name: string, passed: boolean, detail = "") => {
  if (!passed) failures += 1;
  console.log(`${passed ? "  ok  " : "  FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
};
const heading = (text: string) => console.log(`\n${text}`);

const median = (values: number[]) => {
  const sorted = [...values].sort((left, right) => left - right);
  if (sorted.length === 0) return Number.NaN;
  const middle = (sorted.length - 1) / 2;
  const low = Math.floor(middle);
  const high = Math.ceil(middle);
  return low === high ? sorted[low] : (sorted[low] + sorted[high]) / 2;
};

/** A sweep of 40 log-spaced trials whose response is whatever `shape` says. */
function syntheticStudy(
  name: string,
  shape: (x: number) => number,
  low: number,
  high: number,
): ArchiveStudy {
  const trials = Array.from({ length: 40 }, (_, index) => {
    const axis = Math.log10(low) + ((Math.log10(high) - Math.log10(low)) * index) / 39;
    const x = 10 ** axis;
    const value = shape(x);
    return {
      number: index,
      state: "COMPLETE",
      value,
      params: { lr: x },
      metrics: { selected_epoch: 5, last_epoch: 20 },
      folds: [-0.004, 0.004, -0.002, 0.002].map((offset, fold) => ({
        fold,
        value: value + offset,
        precisionAt1: value + offset,
        lastEpoch: 20,
        selectedEpoch: 5,
      })),
    };
  });
  return {
    id: name,
    path: name,
    direction: "maximize",
    metric: "best_valid_mean_average_precision_at_r",
    parameters: [
      {
        name: "lr",
        kind: "numeric",
        distribution: "float",
        log: true,
        low,
        high,
        searched: true,
      },
    ],
    trials,
  } as unknown as ArchiveStudy;
}

// ------------------------------------------------------------------ transforms
heading("axis transforms");
check("log10 round trip", Math.abs(fromAxis(toAxis(1e-4, true), true) - 1e-4) < 1e-18);
check("linear passthrough", fromAxis(toAxis(7.5, false), false) === 7.5);

// ----------------------------------------------------------------- noise model
heading("paired noise model");
{
  // Folds differ by a large shared offset; trials differ by a tiny amount. The
  // raw fold spread sees the offset, the decomposition must not.
  const foldOffsets = [0.1, -0.1, 0.05, -0.05];
  const rows = Array.from({ length: 6 }, (_, trial) =>
    foldOffsets.map((offset) => 0.5 + offset + trial * 0.001),
  );
  const model = fitNoiseModel(rows);
  check(
    "the shared fold effect inflates the raw spread",
    (model.foldSigma ?? 0) > 0.05,
    `fold sigma ${model.foldSigma?.toFixed(4)}`,
  );
  check(
    "the decomposition removes it",
    (model.residualSigma ?? 1) < 1e-9,
    `residual sigma ${model.residualSigma?.toExponential(2)}`,
  );
  check(
    "the comparison error is sigma * sqrt(2/F)",
    Math.abs(
      (model.comparisonError ?? 0) - (model.residualSigma ?? 0) * Math.sqrt(2 / 4),
    ) < 1e-12,
  );
}

// -------------------------------------------------------------- decision rules
heading("optimum pinned against the low edge");
{
  const study = syntheticStudy("edge-low", (x) => 0.65 - 0.05 * (Math.log10(x) + 5), 1e-5, 1e-1);
  const assessment = assessStudy(study, DEFAULT_RANGE_SETTINGS);
  // Exact bounds: rounding is applied after the margin and is asserted on its own.
  const exact: RangeSettings = { ...DEFAULT_RANGE_SETTINGS, roundBounds: false };
  const finding = findRange("lr", [assessment], exact);
  if (finding.kind !== "numeric") throw new Error("expected a numeric finding");
  check("verdict extends downward", finding.verdict === "extend-low", finding.verdict);
  check("the best trial is flagged at the low edge", finding.bestAtEdge === "low");
  check(
    "the recommendation drops below what was searched",
    finding.recommended != null && finding.recommended[0] < 1e-5 && finding.beyondSearched,
    finding.recommended?.[0].toExponential(2),
  );
  check(
    "the margin is exactly half a decade",
    finding.recommended != null &&
      finding.good != null &&
      Math.abs(Math.log10(finding.recommended[0] / finding.good[0]) + 0.5) < 1e-9,
  );
}

heading("optimum in the interior, degradation on both sides");
{
  const peak = Math.log10(1e-3);
  const study = syntheticStudy("interior", (x) => 0.65 - 0.35 * (Math.log10(x) - peak) ** 2, 1e-6, 1);
  const assessment = assessStudy(study, DEFAULT_RANGE_SETTINGS);
  const finding = findRange("lr", [assessment], DEFAULT_RANGE_SETTINGS);
  if (finding.kind !== "numeric") throw new Error("expected a numeric finding");
  check("verdict does not extend", finding.verdict === "narrow" || finding.verdict === "keep", finding.verdict);
  check("no edge is flagged", finding.bestAtEdge === null);
  check(
    "the good region brackets the true peak",
    finding.good != null && finding.good[0] < 1e-3 && finding.good[1] > 1e-3,
    finding.good?.map((value) => value.toExponential(2)).join(" .. "),
  );
}

heading("response that never degrades");
{
  const study = syntheticStudy("flat", () => 0.6, 1e-5, 1e-1);
  const assessment = assessStudy(study, DEFAULT_RANGE_SETTINGS);
  const finding = findRange("lr", [assessment], DEFAULT_RANGE_SETTINGS);
  if (finding.kind !== "numeric") throw new Error("expected a numeric finding");
  check("verdict extends both ways", finding.verdict === "extend-both", finding.verdict);
  check("both edges read as flat", finding.flatLow && finding.flatHigh);
}


// --------------------------------------------------------------- rounding
heading("rounded bounds");
for (const [value, direction, expected] of [
  [0.0000357, "down", 3e-5],
  [0.0000357, "up", 4e-5],
  [224, "down", 200],
  [224, "up", 250],
  [0.885, "up", 1],
] as const) {
  const got = niceBound(value, direction);
  check(
    `${value} ${direction} -> ${got}`,
    Math.abs(got - expected) < 1e-12 * Math.max(1, expected),
    `expected ${expected}`,
  );
}
check(
  "rounding down never increases and rounding up never decreases",
  [1e-5, 0.37, 91, 4.2].every(
    (value) => niceBound(value, "down") <= value && niceBound(value, "up") >= value,
  ),
);

// ---------------------------------------------------------- startup trials
heading("startup trials");
{
  // 10 hits in 100 trials: n >= ln(0.1)/ln(0.9) = 21.85 -> 22.
  const synthetic = syntheticStudy("startup", (x) => (x < 1e-4 ? 0.65 : 0.2), 1e-6, 1);
  const assessment = assessStudy(synthetic, DEFAULT_RANGE_SETTINGS);
  const summary = estimateStartupTrials([assessment]);
  const estimate = summary.perStudy[0];
  check("a hit rate is measured", estimate.p > 0 && estimate.p < 1, `p = ${estimate.p.toFixed(3)}`);
  check(
    "the analytic n is ceil(ln(1-c)/ln(1-p))",
    estimate.analytic === Math.ceil(Math.log(1 - summary.confidence) / Math.log(1 - estimate.p)),
  );
  check(
    "the subset curve crosses the confidence exactly at the reported n",
    estimate.empirical != null &&
      estimate.curve[estimate.empirical - 1].probability >= summary.confidence &&
      (estimate.empirical === 1 ||
        estimate.curve[estimate.empirical - 2].probability < summary.confidence),
  );
  check(
    "the curve rises monotonically",
    estimate.curve.every(
      (point, index) => index === 0 || point.probability >= estimate.curve[index - 1].probability - 1e-12,
    ),
  );
  check("the recommendation respects the density floor", (estimate.recommended ?? 0) >= summary.floor);
}

// ------------------------------------------------------------- the real export
if (!existsSync(snapshotPath)) {
  console.log(
    `\nSkipping the archive checks: ${path.relative(projectRoot, snapshotPath)} is missing. Run npm run sync-sorted first.`,
  );
} else {
  const snapshot = JSON.parse(readFileSync(snapshotPath, "utf8"));
  heading(`archive (${snapshot.studies.length} studies)`);

  const started = performance.now();
  const assessments: StudyAssessment[] = snapshot.studies.map((study: ArchiveStudy) =>
    assessStudy(study, DEFAULT_RANGE_SETTINGS),
  );
  const elapsed = performance.now() - started;
  check("assessing the whole archive stays interactive", elapsed < 1500, `${elapsed.toFixed(0)} ms`);

  const rawSigmas: number[] = [];
  const residualSigmas: number[] = [];
  const resolutions: number[] = [];
  for (const assessment of assessments) {
    if (assessment.noise.foldSigma != null) rawSigmas.push(assessment.noise.foldSigma);
    if (assessment.noise.residualSigma != null) residualSigmas.push(assessment.noise.residualSigma);
    if (assessment.resolution != null) resolutions.push(assessment.resolution);
  }
  console.log(
    `  raw fold sigma ${median(rawSigmas).toFixed(4)} -> residual ${median(residualSigmas).toFixed(4)} ` +
      `(${(median(rawSigmas) / median(residualSigmas)).toFixed(1)}x smaller); median effect ${median(resolutions).toFixed(1)}x epsilon`,
  );
  check(
    "removing the fold effect materially lowers the noise floor",
    median(residualSigmas) < median(rawSigmas) / 2,
  );
  check(
    "most studies resolve their own effect",
    resolutions.filter((value) => value <= 1).length < resolutions.length / 2,
    `${resolutions.filter((value) => value <= 1).length} of ${resolutions.length} cannot`,
  );

  const fixture =
    assessments.find((assessment) => assessment.study.path === "phase_0/supervised/cars196/arcface") ??
    assessments[0];
  const searchedNames = fixture.study.parameters
    .filter((parameter) => parameter.searched)
    .map((parameter) => parameter.name);
  const findings = searchedNames.map((name) => findRange(name, [fixture], DEFAULT_RANGE_SETTINGS));
  check("every searched parameter yields a finding", findings.length === searchedNames.length);
  check(
    "collectSamples returns one row per trial that set the parameter",
    collectSamples("lr", [fixture], true).length ===
      fixture.trials.filter((trial) => trial.trial.params.lr != null).length,
  );

  heading("the controls move the answer");
  const wider: RangeSettings = { ...DEFAULT_RANGE_SETTINGS, noiseMultiplier: 8 };
  const tighter: RangeSettings = { ...DEFAULT_RANGE_SETTINGS, noiseMultiplier: 0.5 };
  const wide = findRange("lr", [assessStudy(fixture.study, wider)], wider);
  const tight = findRange("lr", [assessStudy(fixture.study, tighter)], tighter);
  check(
    "a wider tolerance yields a wider good region",
    wide.kind === "numeric" &&
      tight.kind === "numeric" &&
      wide.good != null &&
      tight.good != null &&
      wide.good[1] - wide.good[0] > tight.good[1] - tight.good[0],
  );
  const noMargin = findRange("lr", [fixture], {
    ...DEFAULT_RANGE_SETTINGS,
    roundBounds: false,
    marginLow: 0,
    marginHigh: 0,
  });
  check(
    "with no margin the recommendation is exactly the good region",
    noMargin.kind === "numeric" &&
      noMargin.good != null &&
      noMargin.recommended != null &&
      Math.abs(noMargin.recommended[0] - noMargin.good[0]) < 1e-12 &&
      Math.abs(noMargin.recommended[1] - noMargin.good[1]) < 1e-12,
  );
  // Rounding must only ever widen: a rounded range that dropped a trial the
  // exact one kept would quietly contradict the tolerance band.
  const searched = fixture.study.parameters.filter((parameter) => parameter.searched);
  const rounded = searched.every((parameter) => {
    const loose = findRange(parameter.name, [fixture], DEFAULT_RANGE_SETTINGS);
    const tight = findRange(parameter.name, [fixture], {
      ...DEFAULT_RANGE_SETTINGS,
      roundBounds: false,
    });
    if (loose.kind !== "numeric" || tight.kind !== "numeric") return true;
    if (!loose.recommended || !tight.recommended) return true;
    return (
      loose.recommended[0] <= tight.recommended[0] + 1e-12 &&
      loose.recommended[1] >= tight.recommended[1] - 1e-12
    );
  });
  check("a rounded range always contains the exact one", rounded);

  const deadOff = assessStudy(fixture.study, { ...DEFAULT_RANGE_SETTINGS, deadMode: "off" });
  check(
    "turning the failure rule off revives the collapsed trials",
    deadOff.counts.dead === 0,
    `${fixture.counts.dead} -> ${deadOff.counts.dead}`,
  );

  heading("validation only");
  check(
    "every objective is a validation metric",
    (snapshot.objectiveMetrics ?? []).every((metric: string) => metric.startsWith("best_valid_")),
    (snapshot.objectiveMetrics ?? []).join(", "),
  );
  check(
    "the snapshot carries no test metric at all",
    !JSON.stringify(snapshot).includes('"test_'),
  );

  heading("folds and epochs");
  const foldRows = foldBreakdown(assessments);
  const counts = mixedFoldCounts(assessments);
  console.log(`  fold counts present: ${counts.join(", ")}`);
  check("per-fold records were exported", foldRows.length > 0 && foldRows[0].trials > 0);
  check("each fold row names the studies behind it", foldRows.every((row) => row.studies > 0));
  // Offsets are measured against each trial's own mean, so they cancel over a
  // population that ran the same number of folds — but only over such a one.
  const uniform = assessments.filter(
    (assessment) => (assessment.trials[0]?.trial.folds?.length ?? 0) === counts[0],
  );
  const uniformSum = foldBreakdown(uniform)
    .map((row) => row.effect ?? 0)
    .reduce((total, value) => total + value, 0);
  check(
    "within one fold count the offsets cancel",
    Math.abs(uniformSum) < 0.005,
    `sum ${uniformSum.toFixed(5)}`,
  );

  const budget = summariseEpochs(assessments);
  console.log(
    `  epoch budget ${budget.budget ?? budget.budgets.join("/")}: ${budget.atBudget}/${budget.folds} folds ran out ` +
      `(${((budget.share ?? 0) * 100).toFixed(2)}%) across ${budget.studiesAtBudget}/${budget.studies} studies`,
  );
  check("epoch usage was measured", budget.folds > 0 && budget.atBudget <= budget.folds);

  heading("startup trials over the archive");
  const startup = estimateStartupTrials(assessments);
  console.log(
    `  median hit rate ${((startup.p ?? 0) * 100).toFixed(1)}% -> n = ${startup.median}, hardest study ${startup.worst}`,
  );
  check("a recommendation exists", startup.median != null);
  check("it respects the density floor", (startup.median ?? 0) >= startup.floor);

  heading("gradient diagnostics");
  if (!existsSync(gradientPath)) {
    console.log("  skipped: public/sorted-gradients.json is missing");
  } else {
    const index: GradientIndex = JSON.parse(readFileSync(gradientPath, "utf8")).studies;
    const catalog = availableGradientSeries(assessments, index);
    console.log(`  ${catalog.trials} trials with diagnostics, ${catalog.series.length} series`);
    check("gradient series were found", catalog.series.includes("ratio"));
    const rows = gradientFoldRows(assessments, index, "ratio", "median");
    check(
      "the aggregate row covers exactly the per-fold rows",
      rows.aggregate != null &&
        rows.aggregate.samples === rows.rows.reduce((total, row) => total + row.samples, 0),
    );
    if (catalog.hasTarget) {
      const withTarget = assessments.filter((assessment) =>
        Object.values(index[assessment.study.id] ?? {}).some((trial) =>
          Object.values(trial).some((fold) => fold.targetDeviation),
        ),
      );
      const targeted = gradientFoldRows(withTarget, index, "ratio", "median");
      const deviation = targeted.aggregate?.targetDeviation;
      console.log(
        `  ${withTarget.length} studies run a controller target; achieved ratio sits ` +
          `${deviation?.toFixed(2)} decades off it (${(10 ** (deviation ?? 0)).toPrecision(3)}x)`,
      );
      check("target deviation is reported", deviation != null);
    }
  }

  heading("emitted Phase 1 config");
  const spaces = toSpaces(findings);
  const expected = findings.filter(
    (finding) =>
      (finding.kind === "numeric" && finding.recommended) ||
      (finding.kind === "categorical" && finding.surviving.length > 0),
  ).length;
  check("one entry per finding that produced a range", Object.keys(spaces).length === expected);
  check(
    "every numeric space has low < high",
    Object.values(spaces).every((space) => {
      const entry = space as { type: string; low?: number; high?: number };
      return entry.type === "categorical" || (entry.low as number) < (entry.high as number);
    }),
  );
}

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
