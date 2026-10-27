"""t-SNE on a CUDA device, following scikit-learn's ``TSNE`` step for step.

sklearn's default (``method="barnes_hut"``) spends almost all of its time in the
Barnes-Hut force approximation, on the CPU. This module keeps every other part
of that algorithm as sklearn runs it and replaces only that part with the exact
forces, computed on the GPU:

* input affinities from the ``3 * perplexity + 1`` nearest neighbours (squared
  Euclidean), a per-point binary search for the perplexity (100 steps,
  tolerance 1e-5, natural log), then ``P + P.T`` normalized to sum to one;
* ``init="pca"`` through sklearn's own PCA, rescaled so PC1 has std 1e-4;
* Student-t kernel with ``max(n_components - 1, 1)`` degrees of freedom;
* 250 iterations with early exaggeration 12 and momentum 0.5, then momentum
  0.8 up to ``max_iter``; ``learning_rate="auto"``; per-coordinate gains
  (+0.2 / *0.8, floor 0.01); the KL-progress and gradient-norm checks every 50
  iterations.

The attractive force over the sparse neighbour affinities is what sklearn
computes too. The repulsive force is summed over every pair instead of
approximated with a tree (``angle=0.5``): by a Triton kernel that makes one pass
per row and allocates nothing quadratic, or, without Triton, in row blocks sized
so the pairwise buffers stay under ``block_memory_bytes``. The result is therefore a slightly
more accurate optimization of the same objective from the same start, not
sklearn's coordinates to the last digit.
"""

from __future__ import annotations

import math

import numpy as np
import torch

#: Default ceiling for the pairwise buffers of one force block.
DEFAULT_BLOCK_MEMORY_BYTES = 256 * 1024**2

_EXPLORATION_MAX_ITER = 250
_N_ITER_CHECK = 50
_PERPLEXITY_STEPS = 100
_PERPLEXITY_TOLERANCE = 1e-5
_EPSILON_DBL = 1e-8
_MACHINE_EPSILON = float(np.finfo(np.double).eps)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _rows_per_block(n_samples, bytes_per_cell, block_memory_bytes):
    return max(1, min(n_samples, int(block_memory_bytes // max(1, n_samples * bytes_per_cell))))


def _nearest_neighbors(features, n_neighbors, block_memory_bytes):
    """Exact squared-Euclidean neighbours of every row, excluding the row itself."""

    n_samples, n_features = features.shape
    device = features.device
    norms = (features * features).sum(dim=1)
    distances, indices = [], []
    for start, stop in _block_bounds(n_samples, _rows_per_block(n_samples, 4 * 3, block_memory_bytes)):
        block = norms[start:stop, None] + norms[None, :] - 2.0 * (features[start:stop] @ features.T)
        block[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = math.inf
        positions = torch.topk(block, n_neighbors, dim=1, largest=False, sorted=True).indices
        # The Gram expansion above loses precision for close pairs, so the
        # neighbours' distances are recomputed from their differences.
        difference = features[start:stop, None, :] - features[positions]
        distances.append((difference * difference).sum(dim=2))
        indices.append(positions)
    return torch.cat(distances), torch.cat(indices)


def _block_bounds(n_samples, rows):
    return [(start, min(n_samples, start + rows)) for start in range(0, n_samples, rows)]


def _conditional_probabilities(squared_distances, perplexity):
    """sklearn's ``_binary_search_perplexity``, all rows at once, in float64."""

    distances = squared_distances.to(torch.float32).to(torch.float64)
    n_samples = distances.shape[0]
    device = distances.device
    desired_entropy = math.log(perplexity)
    beta = torch.ones(n_samples, dtype=torch.float64, device=device)
    beta_min = torch.full_like(beta, -math.inf)
    beta_max = torch.full_like(beta, math.inf)
    probabilities = torch.zeros_like(distances)
    active = torch.ones(n_samples, dtype=torch.bool, device=device)
    for _ in range(_PERPLEXITY_STEPS):
        current = torch.exp(-distances * beta[:, None])
        total = current.sum(dim=1)
        total = torch.where(total == 0.0, torch.full_like(total, _EPSILON_DBL), total)
        current = current / total[:, None]
        entropy = torch.log(total) + beta * (distances * current).sum(dim=1)
        # Rows that already met the tolerance keep the probabilities they met it with.
        probabilities = torch.where(active[:, None], current, probabilities)
        difference = entropy - desired_entropy
        converged = difference.abs() <= _PERPLEXITY_TOLERANCE
        step = active & ~converged
        increase = step & (difference > 0.0)
        decrease = step & ~(difference > 0.0)
        beta_min = torch.where(increase, beta, beta_min)
        beta_max = torch.where(decrease, beta, beta_max)
        new_beta = beta.clone()
        new_beta = torch.where(
            increase,
            torch.where(torch.isinf(beta_max), beta * 2.0, (beta + beta_max) / 2.0),
            new_beta,
        )
        new_beta = torch.where(
            decrease,
            torch.where(torch.isinf(beta_min), beta / 2.0, (beta + beta_min) / 2.0),
            new_beta,
        )
        beta = new_beta
        active = step
        if not bool(active.any()):
            break
    return probabilities


def _joint_probabilities(neighbor_distances, neighbor_indices, perplexity):
    """Symmetric, normalized neighbour affinities, padded per row (see :func:`_pad_rows`)."""

    n_samples, n_neighbors = neighbor_indices.shape
    conditional = _conditional_probabilities(neighbor_distances, perplexity).to(torch.float32)
    device = neighbor_indices.device
    rows = torch.arange(n_samples, device=device).repeat_interleave(n_neighbors)
    cols = neighbor_indices.reshape(-1)
    values = conditional.reshape(-1).to(torch.float64)
    # P + P.T: an edge found from both ends is one entry holding both halves.
    keys = torch.cat([rows * n_samples + cols, cols * n_samples + rows])
    unique_keys, positions = torch.unique(keys, sorted=True, return_inverse=True)
    summed = torch.zeros(len(unique_keys), dtype=torch.float64, device=device)
    summed.index_add_(0, positions, torch.cat([values, values]))
    summed = summed / max(float(summed.sum()), _MACHINE_EPSILON)
    return _pad_rows(unique_keys // n_samples, unique_keys % n_samples, summed.to(torch.float32), n_samples)


def _pad_rows(rows, cols, values, n_samples):
    """Lay the affinities out as ``(n_samples, max_degree)`` rows padded with zeros.

    Summing each row along a dimension is deterministic on CUDA, where
    ``index_add_`` over repeated indices is not, and t-SNE turns a difference in
    the last bit into a different layout. ``rows`` must be sorted.
    """

    device = rows.device
    starts = torch.searchsorted(rows, torch.arange(n_samples + 1, device=device))
    width = int((starts[1:] - starts[:-1]).max())
    slots = torch.arange(len(rows), device=device) - starts[rows]
    neighbors = torch.zeros((n_samples, width), dtype=torch.long, device=device)
    affinities = torch.zeros((n_samples, width), dtype=values.dtype, device=device)
    neighbors[rows, slots] = cols
    affinities[rows, slots] = values
    return neighbors, affinities


def _kernel(squared_distances, degrees_of_freedom):
    kernel = degrees_of_freedom / (degrees_of_freedom + squared_distances)
    if degrees_of_freedom != 1:
        kernel = kernel ** ((degrees_of_freedom + 1.0) / 2.0)
    return kernel


def _repulsion_blocks(embedding, degrees_of_freedom, block_rows):
    """``(sum of pair kernels, unnormalized repulsive force)`` with row-block buffers."""

    n_samples, n_components = embedding.shape
    negative = torch.zeros_like(embedding)
    normalizer = torch.zeros((), dtype=torch.float64, device=embedding.device)
    for start, stop in _block_bounds(n_samples, block_rows):
        block = embedding[start:stop]
        squared = torch.zeros((stop - start, n_samples), dtype=embedding.dtype, device=embedding.device)
        for axis in range(n_components):
            squared.add_((block[:, axis, None] - embedding[None, :, axis]).square_())
        pair_kernel = _kernel(squared, degrees_of_freedom)
        # Each row's kernel with itself is exactly 1 and is not a pair.
        normalizer += pair_kernel.sum(dtype=torch.float64) - (stop - start)
        squared_kernel = pair_kernel.square_()
        negative[start:stop] = block * squared_kernel.sum(dim=1, keepdim=True) - squared_kernel @ embedding
    return normalizer, negative


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton ships with CUDA builds of torch
    triton = None


if triton is not None:

    @triton.jit
    def _repulsion_row_kernel(
        positions,
        kernel_sums,
        squared_sums,
        weighted_sums,
        n_samples,
        degrees_of_freedom,
        exponent,
        DIMS: tl.constexpr,
        UNIT_DOF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One program per row i: sum the pair kernel w_ij, w_ij^2 and w_ij^2 y_j
        # over every j in chunks, so no n x n buffer is ever allocated.
        row = tl.program_id(0)
        y0 = tl.load(positions + row * DIMS)
        y1 = tl.load(positions + row * DIMS + 1)
        y2 = 0.0
        if DIMS == 3:
            y2 = tl.load(positions + row * DIMS + 2)
        kernel_sum = 0.0
        squared_sum = 0.0
        weighted0 = 0.0
        weighted1 = 0.0
        weighted2 = 0.0
        for start in range(0, n_samples, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = offsets < n_samples
            z0 = tl.load(positions + offsets * DIMS, mask=mask, other=0.0)
            z1 = tl.load(positions + offsets * DIMS + 1, mask=mask, other=0.0)
            d0 = y0 - z0
            d1 = y1 - z1
            distance = d0 * d0 + d1 * d1
            if DIMS == 3:
                z2 = tl.load(positions + offsets * DIMS + 2, mask=mask, other=0.0)
                d2 = y2 - z2
                distance += d2 * d2
            kernel = degrees_of_freedom / (degrees_of_freedom + distance)
            if not UNIT_DOF:
                kernel = tl.exp(exponent * tl.log(kernel))
            kernel = tl.where(mask, kernel, 0.0)
            squared = kernel * kernel
            kernel_sum += tl.sum(kernel)
            squared_sum += tl.sum(squared)
            weighted0 += tl.sum(squared * z0)
            weighted1 += tl.sum(squared * z1)
            if DIMS == 3:
                weighted2 += tl.sum(squared * z2)
        tl.store(kernel_sums + row, kernel_sum)
        tl.store(squared_sums + row, squared_sum)
        tl.store(weighted_sums + row * DIMS, weighted0)
        tl.store(weighted_sums + row * DIMS + 1, weighted1)
        if DIMS == 3:
            tl.store(weighted_sums + row * DIMS + 2, weighted2)


def _repulsion_fused(embedding, degrees_of_freedom):
    """:func:`_repulsion_blocks` as one Triton pass per row, with O(n) memory."""

    n_samples, n_components = embedding.shape
    positions = embedding.contiguous()
    kernel_sums = torch.empty(n_samples, dtype=embedding.dtype, device=embedding.device)
    squared_sums = torch.empty_like(kernel_sums)
    weighted_sums = torch.empty_like(positions)
    _repulsion_row_kernel[(n_samples,)](
        positions,
        kernel_sums,
        squared_sums,
        weighted_sums,
        n_samples,
        float(degrees_of_freedom),
        (degrees_of_freedom + 1.0) / 2.0,
        DIMS=n_components,
        UNIT_DOF=degrees_of_freedom == 1,
        BLOCK=1024,
    )
    # Each row's kernel with itself is exactly 1 and is not a pair.
    normalizer = kernel_sums.sum(dtype=torch.float64) - n_samples
    negative = positions * squared_sums[:, None] - weighted_sums
    return normalizer, negative


def _use_fused_repulsion(embedding):
    return triton is not None and embedding.is_cuda and embedding.shape[1] in (2, 3)


def _gradient(embedding, neighbors, p_values, degrees_of_freedom, block_rows, compute_error, fused=None):
    """KL gradient with exact repulsion; also the KL divergence when asked."""

    # Attraction over the neighbour graph, exactly as sklearn sums it. Padding
    # slots carry zero affinity and so contribute nothing.
    difference = embedding[:, None, :] - embedding[neighbors]
    kernel = _kernel((difference * difference).sum(dim=2), degrees_of_freedom)
    positive = ((p_values * kernel)[:, :, None] * difference).sum(dim=1)

    # Repulsion over every pair.
    if _use_fused_repulsion(embedding) if fused is None else fused:
        normalizer, negative = _repulsion_fused(embedding, degrees_of_freedom)
    else:
        normalizer, negative = _repulsion_blocks(embedding, degrees_of_freedom, block_rows)

    total = normalizer.to(embedding.dtype)
    gradient = (positive - negative / total) * (2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom)
    error = None
    if compute_error:
        q_values = kernel / total
        error = (
            p_values
            * torch.log(p_values.clamp(min=_FLOAT32_TINY) / q_values.clamp(min=_FLOAT32_TINY))
        ).sum()
    return gradient, error


def _gradient_descent(
    embedding,
    objective,
    *,
    start_iteration,
    max_iter,
    momentum,
    learning_rate,
    n_iter_without_progress,
    min_grad_norm,
    update,
    gains,
):
    best_error = float(np.finfo(float).max)
    error = best_error
    best_iteration = iteration = start_iteration
    for iteration in range(start_iteration, max_iter):
        check_convergence = (iteration + 1) % _N_ITER_CHECK == 0
        gradient, error_tensor = objective(check_convergence or iteration == max_iter - 1)
        increase = (update * gradient) < 0.0
        gains = torch.where(increase, gains + 0.2, gains * 0.8).clamp_(min=0.01)
        gradient = gradient * gains
        update = momentum * update - learning_rate * gradient
        embedding += update
        if error_tensor is not None:
            error = float(error_tensor)
        if check_convergence:
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            if error < best_error:
                best_error = error
                best_iteration = iteration
            elif iteration - best_iteration > n_iter_without_progress:
                break
            if gradient_norm <= min_grad_norm:
                break
    return embedding, error, iteration


def gpu_tsne(
    features,
    *,
    n_components=2,
    perplexity=30.0,
    early_exaggeration=12.0,
    max_iter=1000,
    n_iter_without_progress=300,
    min_grad_norm=1e-7,
    random_state=0,
    device="cuda",
    block_memory_bytes=DEFAULT_BLOCK_MEMORY_BYTES,
):
    """Embed ``features`` (n_samples, n_features) with t-SNE on ``device``.

    Returns float32 NumPy coordinates of shape (n_samples, n_components), like
    ``TSNE(init="pca", learning_rate="auto", method="barnes_hut").fit_transform``.
    """

    from sklearn.decomposition import PCA

    features = np.ascontiguousarray(features)
    if features.dtype not in (np.float32, np.float64):
        features = features.astype(np.float32)
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE needs at least two samples")
    if not 0.0 < float(perplexity) < n_samples:
        raise ValueError("perplexity must be positive and smaller than the number of samples")
    device = torch.device(device)
    n_neighbors = min(n_samples - 1, int(3.0 * float(perplexity) + 1))
    degrees_of_freedom = max(int(n_components) - 1, 1)
    learning_rate = max(n_samples / float(early_exaggeration) / 4.0, 50.0)

    with torch.no_grad():
        source = torch.as_tensor(features, device=device)
        neighbor_distances, neighbor_indices = _nearest_neighbors(source, n_neighbors, block_memory_bytes)
        del source
        neighbors, p_values = _joint_probabilities(neighbor_distances, neighbor_indices, float(perplexity))
        del neighbor_distances, neighbor_indices

        pca = PCA(n_components=n_components, random_state=random_state)
        pca.set_output(transform="default")
        initial = pca.fit_transform(features).astype(np.float32, copy=False)
        initial = initial / np.std(initial[:, 0]) * 1e-4
        embedding = torch.as_tensor(initial, dtype=torch.float32, device=device).contiguous()

        # Pairwise buffers per block: squared distances, kernel, and the axis difference.
        block_rows = _rows_per_block(n_samples, 4 * 3, block_memory_bytes)
        exaggerated = p_values * float(early_exaggeration)

        def objective(values):
            return lambda compute_error: _gradient(
                embedding, neighbors, values, degrees_of_freedom, block_rows, compute_error
            )

        update = torch.zeros_like(embedding)
        gains = torch.ones_like(embedding)
        embedding, _, iteration = _gradient_descent(
            embedding,
            objective(exaggerated),
            start_iteration=0,
            max_iter=_EXPLORATION_MAX_ITER,
            momentum=0.5,
            learning_rate=learning_rate,
            n_iter_without_progress=_EXPLORATION_MAX_ITER,
            min_grad_norm=min_grad_norm,
            update=update,
            gains=gains,
        )
        if iteration < _EXPLORATION_MAX_ITER or max_iter - _EXPLORATION_MAX_ITER > 0:
            # sklearn restarts the update and gains with the second schedule.
            embedding, _, iteration = _gradient_descent(
                embedding,
                objective(p_values),
                start_iteration=iteration + 1,
                max_iter=int(max_iter),
                momentum=0.8,
                learning_rate=learning_rate,
                n_iter_without_progress=n_iter_without_progress,
                min_grad_norm=min_grad_norm,
                update=torch.zeros_like(embedding),
                gains=torch.ones_like(embedding),
            )
        return embedding.cpu().numpy()
