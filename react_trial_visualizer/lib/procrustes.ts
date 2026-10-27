/**
 * Epoch-to-epoch alignment of tracked layouts, the browser twin of
 * evaluate/export_embedding_views.py (procrustes_align and tracking_view's
 * align_all).
 *
 * Each tracked epoch fits its own projection, free to rotate or mirror the whole
 * picture. Aligning every epoch to the one before it on the shared points
 * removes that motion; a common scale then keeps sizes comparable across epochs.
 * Live views are assembled in the page as epochs arrive, so this runs here
 * instead of in the exporter.
 */

/** Eigen-decomposition of a small symmetric matrix (row-major, d x d) by Jacobi rotations. */
function symmetricEigen(matrix: number[], d: number): { values: number[]; vectors: number[] } {
  const a = [...matrix];
  const v = Array.from({ length: d * d }, (_, index): number => (index % (d + 1) === 0 ? 1 : 0));
  for (let sweep = 0; sweep < 60; sweep += 1) {
    let off = 0;
    for (let p = 0; p < d; p += 1) for (let q = p + 1; q < d; q += 1) off += a[p * d + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < d; p += 1) {
      for (let q = p + 1; q < d; q += 1) {
        const apq = a[p * d + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q * d + q] - a[p * d + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < d; k += 1) {
          const akp = a[k * d + p];
          const akq = a[k * d + q];
          a[k * d + p] = c * akp - s * akq;
          a[k * d + q] = s * akp + c * akq;
        }
        for (let k = 0; k < d; k += 1) {
          const apk = a[p * d + k];
          const aqk = a[q * d + k];
          a[p * d + k] = c * apk - s * aqk;
          a[q * d + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < d; k += 1) {
          const vkp = v[k * d + p];
          const vkq = v[k * d + q];
          v[k * d + p] = c * vkp - s * vkq;
          v[k * d + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: Array.from({ length: d }, (_, index) => a[index * d + index]), vectors: v };
}

function centered(layout: ArrayLike<number>, d: number): Float64Array {
  const n = layout.length / d;
  const mean = new Array<number>(d).fill(0);
  for (let row = 0; row < n; row += 1) for (let axis = 0; axis < d; axis += 1) mean[axis] += layout[row * d + axis];
  for (let axis = 0; axis < d; axis += 1) mean[axis] /= n || 1;
  const result = new Float64Array(layout.length);
  for (let row = 0; row < n; row += 1) for (let axis = 0; axis < d; axis += 1) result[row * d + axis] = layout[row * d + axis] - mean[axis];
  return result;
}

function rmsScale(values: Float64Array, n: number): number {
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) sum += values[index] ** 2;
  return Math.sqrt(sum / (n || 1)) || 1;
}

/**
 * Rotate/reflect and scale `layout` onto `reference` (both n x d, row-major),
 * like numpy's `u @ vt` from the SVD of the cross-covariance. The rotation is
 * the orthogonal polar factor M (MᵀM)^(-1/2), which equals U Vᵀ whenever M has
 * full rank; a degenerate layout is left unrotated.
 */
export function procrustesAlign(layout: ArrayLike<number>, reference: ArrayLike<number>, d: number): Float64Array {
  const n = layout.length / d;
  const source = centered(layout, d);
  const target = centered(reference, d);
  const sourceScale = rmsScale(source, n);
  const targetScale = rmsScale(target, n);
  const m = new Array<number>(d * d).fill(0);
  for (let row = 0; row < n; row += 1) {
    for (let i = 0; i < d; i += 1) {
      const si = source[row * d + i] / sourceScale;
      for (let j = 0; j < d; j += 1) m[i * d + j] += si * (target[row * d + j] / targetScale);
    }
  }
  const mtm = new Array<number>(d * d).fill(0);
  for (let i = 0; i < d; i += 1) for (let j = 0; j < d; j += 1) for (let k = 0; k < d; k += 1) mtm[i * d + j] += m[k * d + i] * m[k * d + j];
  const { values, vectors } = symmetricEigen(mtm, d);
  const largest = Math.max(...values);
  let rotation: number[];
  if (!(largest > 0) || values.some((value) => value <= largest * 1e-12)) {
    rotation = Array.from({ length: d * d }, (_, index) => (index % (d + 1) === 0 ? 1 : 0));
  } else {
    // (MᵀM)^(-1/2) = Q Λ^(-1/2) Qᵀ, then R = M (MᵀM)^(-1/2).
    const inverseRoot = new Array<number>(d * d).fill(0);
    for (let i = 0; i < d; i += 1) {
      for (let j = 0; j < d; j += 1) {
        let sum = 0;
        for (let k = 0; k < d; k += 1) sum += (vectors[i * d + k] * vectors[j * d + k]) / Math.sqrt(values[k]);
        inverseRoot[i * d + j] = sum;
      }
    }
    rotation = new Array<number>(d * d).fill(0);
    for (let i = 0; i < d; i += 1) for (let j = 0; j < d; j += 1) for (let k = 0; k < d; k += 1) rotation[i * d + j] += m[i * d + k] * inverseRoot[k * d + j];
  }
  const aligned = new Float64Array(layout.length);
  for (let row = 0; row < n; row += 1) {
    for (let j = 0; j < d; j += 1) {
      let sum = 0;
      for (let k = 0; k < d; k += 1) sum += (source[row * d + k] / sourceScale) * rotation[k * d + j];
      aligned[row * d + j] = sum * targetScale;
    }
  }
  return aligned;
}

/**
 * Align each layout to the one before it, center it, and scale all of them by
 * one common extent into [-1, 1], as the exporter's align_all does.
 */
export function alignSeries(layouts: ArrayLike<number>[], d: number): Float32Array[] {
  const aligned: Float64Array[] = [];
  let previous: Float64Array | null = null;
  for (const layout of layouts) {
    const current = previous == null ? Float64Array.from(layout) : procrustesAlign(layout, previous, d);
    const recentered = centered(current, d);
    aligned.push(recentered);
    previous = recentered;
  }
  let extent = 0;
  for (const layout of aligned) for (let index = 0; index < layout.length; index += 1) extent = Math.max(extent, Math.abs(layout[index]));
  extent ||= 1;
  return aligned.map((layout) => Float32Array.from(layout, (value) => value / extent));
}
