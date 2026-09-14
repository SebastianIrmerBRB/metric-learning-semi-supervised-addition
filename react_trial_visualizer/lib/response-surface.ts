/**
 * Quadratic response-surface model over two search parameters, plus the
 * marching-squares helpers used to draw it.
 *
 * Optuna reports importance per parameter, which cannot show how two knobs
 * trade off against each other. Fitting a second-order polynomial to the
 * completed trials recovers that: the mean surface shows where the model
 * believes the optimum sits, and the standard error shows where it is guessing
 * because no trial was ever sampled nearby.
 */

export interface SurfaceSample {
  /** Search-space position, normalized to [0, 1] over the plotted domain. */
  x: number;
  y: number;
  value: number;
}

export type SurfaceKind = "quadratic" | "linear";

export interface StationaryPoint {
  x: number;
  y: number;
  kind: "maximum" | "minimum" | "saddle";
}

export interface SurfaceModel {
  kind: SurfaceKind;
  /** Fitted coefficients, i.e. the model's degrees of freedom. */
  terms: number;
  observations: number;
  residualDegreesOfFreedom: number;
  rSquared: number;
  /** Residual standard deviation of the fit, in objective units. */
  sigma: number;
  predict: (x: number, y: number) => number;
  /** Standard error of the predicted mean at a point, in objective units. */
  standardError: (x: number, y: number) => number;
  stationaryPoint: StationaryPoint | null;
}

/** Basis centered on the box so the normal equations stay well conditioned. */
function basis(kind: SurfaceKind, x: number, y: number): number[] {
  const u = x - 0.5;
  const v = y - 0.5;
  return kind === "quadratic" ? [1, u, v, u * u, u * v, v * v] : [1, u, v];
}

/** Gauss-Jordan inverse; returns null when the design is rank deficient. */
function invert(matrix: number[][]): number[][] | null {
  const size = matrix.length;
  const scale = Math.max(...matrix.flat().map(Math.abs)) || 1;
  const work = matrix.map((row, index) => [
    ...row,
    ...Array.from({ length: size }, (_, column) => (column === index ? 1 : 0)),
  ]);

  for (let column = 0; column < size; column += 1) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivotRow][column])) {
        pivotRow = row;
      }
    }
    if (Math.abs(work[pivotRow][column]) < 1e-10 * scale) return null;
    [work[column], work[pivotRow]] = [work[pivotRow], work[column]];

    const pivot = work[column][column];
    for (let index = 0; index < 2 * size; index += 1) work[column][index] /= pivot;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = work[row][column];
      if (factor === 0) continue;
      for (let index = 0; index < 2 * size; index += 1) {
        work[row][index] -= factor * work[column][index];
      }
    }
  }
  return work.map((row) => row.slice(size));
}

/** Where the fitted quadratic flattens out, in normalized coordinates. */
function stationaryPointOf(coefficients: number[]): StationaryPoint | null {
  const [, bx, by, bxx, bxy, byy] = coefficients;
  const determinant = 4 * bxx * byy - bxy * bxy;
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;

  // Solve the 2x2 system grad(f) = 0 for the centered coordinates.
  const u = (-2 * byy * bx + bxy * by) / determinant;
  const v = (-2 * bxx * by + bxy * bx) / determinant;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;

  return {
    x: u + 0.5,
    y: v + 0.5,
    kind:
      determinant < 0
        ? "saddle"
        : bxx < 0
          ? "maximum"
          : "minimum",
  };
}

function fitKind(kind: SurfaceKind, samples: SurfaceSample[]): SurfaceModel | null {
  const terms = kind === "quadratic" ? 6 : 3;
  // Two spare observations is the floor for a residual variance worth quoting.
  if (samples.length < terms + 2) return null;

  const design = samples.map((sample) => basis(kind, sample.x, sample.y));
  const targets = samples.map((sample) => sample.value);
  const normal = Array.from({ length: terms }, (_, row) =>
    Array.from({ length: terms }, (_, column) =>
      design.reduce((total, entry) => total + entry[row] * entry[column], 0),
    ),
  );
  const projection = Array.from({ length: terms }, (_, row) =>
    design.reduce((total, entry, index) => total + entry[row] * targets[index], 0),
  );
  const inverse = invert(normal);
  if (!inverse) return null;

  const coefficients = inverse.map((row) =>
    row.reduce((total, value, index) => total + value * projection[index], 0),
  );
  const predict = (x: number, y: number) =>
    basis(kind, x, y).reduce(
      (total, value, index) => total + value * coefficients[index],
      0,
    );

  const mean = targets.reduce((total, value) => total + value, 0) / targets.length;
  let residualSum = 0;
  let totalSum = 0;
  for (const [index, sample] of samples.entries()) {
    residualSum += (targets[index] - predict(sample.x, sample.y)) ** 2;
    totalSum += (targets[index] - mean) ** 2;
  }
  const residualDegreesOfFreedom = samples.length - terms;
  const sigma = Math.sqrt(Math.max(0, residualSum) / residualDegreesOfFreedom);
  if (!Number.isFinite(sigma)) return null;

  return {
    kind,
    terms,
    observations: samples.length,
    residualDegreesOfFreedom,
    rSquared: totalSum > 0 ? 1 - residualSum / totalSum : 0,
    sigma,
    predict,
    standardError: (x: number, y: number) => {
      const entry = basis(kind, x, y);
      let variance = 0;
      for (let row = 0; row < terms; row += 1) {
        for (let column = 0; column < terms; column += 1) {
          variance += entry[row] * inverse[row][column] * entry[column];
        }
      }
      return sigma * Math.sqrt(Math.max(0, variance));
    },
    stationaryPoint: kind === "quadratic" ? stationaryPointOf(coefficients) : null,
  };
}

/**
 * Fits the richest surface the trial count supports: a full quadratic when
 * there are enough trials and they are not collinear, otherwise a plane.
 */
export function fitResponseSurface(samples: SurfaceSample[]): SurfaceModel | null {
  return fitKind("quadratic", samples) ?? fitKind("linear", samples);
}

export interface SurfaceGrid {
  /** Sampled values, row 0 at the top of the plot area. */
  values: number[][];
  minimum: number;
  maximum: number;
}

/** Samples a field on a regular grid; row 0 is the top edge (highest y). */
export function sampleGrid(
  field: (x: number, y: number) => number,
  columns: number,
  rows: number,
): SurfaceGrid {
  const values: number[][] = [];
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;

  for (let row = 0; row <= rows; row += 1) {
    const y = 1 - row / rows;
    const line: number[] = [];
    for (let column = 0; column <= columns; column += 1) {
      const value = field(column / columns, y);
      line.push(value);
      if (value < minimum) minimum = value;
      if (value > maximum) maximum = value;
    }
    values.push(line);
  }
  return { values, minimum, maximum };
}

/** Evenly spaced interior thresholds; `bands` fills need `bands - 1` of them. */
export function contourLevels(
  minimum: number,
  maximum: number,
  bands: number,
): number[] {
  if (!(maximum > minimum)) return [];
  return Array.from(
    { length: bands - 1 },
    (_, index) => minimum + ((maximum - minimum) * (index + 1)) / bands,
  );
}

export interface PlotBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface ContourLayer {
  level: number;
  /** Filled region where the field is at or above `level`. */
  fill: string;
  /** The iso-line bounding that region. */
  line: string;
}

interface Vertex {
  x: number;
  y: number;
  crossing: boolean;
}

function round(value: number): string {
  return (Math.round(value * 100) / 100).toString();
}

/**
 * Region above one threshold, as a path in pixel space.
 *
 * Each cell is clipped against the half-space `value >= level`
 * (Sutherland-Hodgman on the four corners, which is marching squares with
 * linear edge interpolation). Cells that lie entirely inside are merged along
 * each row first, which keeps the emitted path short for the wide flat parts
 * of a surface. Neighbouring cell polygons share their cut edges exactly, so
 * the layers stack without seams.
 */
function contourLayer(grid: number[][], level: number, box: PlotBox): ContourLayer {
  const rows = grid.length - 1;
  const columns = grid[0].length - 1;
  const stepX = box.width / columns;
  const stepY = box.height / rows;
  const pixelX = (column: number) => box.left + column * stepX;
  const pixelY = (row: number) => box.top + row * stepY;

  let fill = "";
  let line = "";

  for (let row = 0; row < rows; row += 1) {
    let runStart = -1;
    const flushRun = (end: number) => {
      if (runStart < 0) return;
      fill += `M${round(pixelX(runStart))},${round(pixelY(row))}H${round(
        pixelX(end),
      )}V${round(pixelY(row + 1))}H${round(pixelX(runStart))}Z`;
      runStart = -1;
    };

    for (let column = 0; column < columns; column += 1) {
      const corners = [
        { value: grid[row][column], x: pixelX(column), y: pixelY(row) },
        { value: grid[row][column + 1], x: pixelX(column + 1), y: pixelY(row) },
        {
          value: grid[row + 1][column + 1],
          x: pixelX(column + 1),
          y: pixelY(row + 1),
        },
        { value: grid[row + 1][column], x: pixelX(column), y: pixelY(row + 1) },
      ];
      const inside = corners.map((corner) => corner.value >= level);

      if (inside.every(Boolean)) {
        if (runStart < 0) runStart = column;
        continue;
      }
      flushRun(column);
      if (!inside.some(Boolean)) continue;

      const polygon: Vertex[] = [];
      for (let index = 0; index < 4; index += 1) {
        const current = corners[index];
        const next = corners[(index + 1) % 4];
        if (inside[index]) {
          polygon.push({ x: current.x, y: current.y, crossing: false });
        }
        if (inside[index] === inside[(index + 1) % 4]) continue;
        const span = next.value - current.value;
        const weight = span === 0 ? 0.5 : (level - current.value) / span;
        polygon.push({
          x: current.x + (next.x - current.x) * weight,
          y: current.y + (next.y - current.y) * weight,
          crossing: true,
        });
      }
      if (polygon.length < 3) continue;

      fill += polygon
        .map(
          (vertex, index) =>
            `${index === 0 ? "M" : "L"}${round(vertex.x)},${round(vertex.y)}`,
        )
        .join("");
      fill += "Z";

      // Consecutive crossings are exactly the cut edges: the iso-line itself.
      for (let index = 0; index < polygon.length; index += 1) {
        const current = polygon[index];
        const next = polygon[(index + 1) % polygon.length];
        if (!current.crossing || !next.crossing) continue;
        line += `M${round(current.x)},${round(current.y)}L${round(next.x)},${round(
          next.y,
        )}`;
      }
    }
    flushRun(columns);
  }

  return { level, fill, line };
}

/** One stacked layer per threshold, lowest first, for painter-order drawing. */
export function contourLayers(
  grid: number[][],
  levels: number[],
  box: PlotBox,
): ContourLayer[] {
  return levels.map((level) => contourLayer(grid, level, box));
}
