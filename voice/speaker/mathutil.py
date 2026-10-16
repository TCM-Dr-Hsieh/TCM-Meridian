"""Numeric helpers for the speaker tracker, in plain numpy (the app's own environment has no scikit-learn)."""
from __future__ import annotations

import numpy as np


def unit_rows(values) -> np.ndarray:
    """Scale every row (or a single vector) to length 1."""
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def _assign(points: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest centre for every point, and the squared distance to it."""
    distance = ((points ** 2).sum(1)[:, None] - 2.0 * points @ centres.T + (centres ** 2).sum(1)[None, :])
    labels = distance.argmin(1)
    return labels, np.maximum(distance[np.arange(len(points)), labels], 0.0)


def _seed(points: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ starting centres (weighted)."""
    n = len(points)
    chosen = [int(rng.choice(n, p=weights / weights.sum()))]
    nearest = ((points - points[chosen[0]]) ** 2).sum(1)
    for _ in range(1, k):
        mass = weights * nearest
        total = mass.sum()
        pick = int(rng.choice(n, p=mass / total)) if total > 0 else int(rng.integers(n))
        chosen.append(pick)
        nearest = np.minimum(nearest, ((points - points[pick]) ** 2).sum(1))
    return points[chosen].copy()


def _lloyd(points: np.ndarray, weights: np.ndarray, centres: np.ndarray, max_iter: int):
    labels = np.zeros(len(points), dtype=int)
    for _ in range(max_iter):
        labels, _ = _assign(points, centres)
        updated = centres.copy()
        for j in range(len(centres)):
            member = labels == j
            if member.any():                                    # an emptied cluster keeps its old centre
                updated[j] = np.average(points[member], axis=0, weights=weights[member])
        moved = float(((updated - centres) ** 2).sum())
        centres = updated
        if moved < 1e-10:
            break
    labels, distance = _assign(points, centres)
    return centres, labels, float((weights * distance).sum())


def kmeans(points, weights, k: int, *, init=None, n_init: int = 10, seed: int = 0, max_iter: int = 100):
    """Weighted k-means. Returns (centres, labels, inertia).

    Without `init`: the best of `n_init` runs from k-means++ starts (deterministic for a given `seed`). With `init`: one run
    from those centres, which keeps the identity of each centre (the i-th result is the i-th start), so a refit never swaps
    the groups' numbers.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(points) < k:
        raise ValueError(f'{len(points)} 個點不足以分成 {k} 群。')
    if init is not None:
        return _lloyd(points, weights, np.array(init, dtype=np.float64), max_iter)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, n_init)):
        run = _lloyd(points, weights, _seed(points, weights, k, rng), max_iter)
        if best is None or run[2] < best[2]:
            best = run
    return best


def tied_two_component(values, *, iters: int = 300) -> tuple[float, float, float]:
    """Two Gaussians with one shared variance fitted to 1-D data by EM: (lower mean, upper mean, variance)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    low, high = (float(v) for v in np.percentile(x, [25, 75]))
    for _ in range(20):                                         # 1-D 2-means as the starting point
        upper = np.abs(x - high) < np.abs(x - low)
        if upper.all() or not upper.any():
            break
        low, high = float(x[~upper].mean()), float(x[upper].mean())
    mean = np.array([low, high])
    var = max(float(x.var()), 1e-6)
    share = np.array([0.5, 0.5])
    last = -np.inf
    for _ in range(iters):
        log = -0.5 * (x[:, None] - mean[None, :]) ** 2 / var - 0.5 * np.log(var) + np.log(share)[None, :]
        top = log.max(1, keepdims=True)
        total = np.log(np.exp(log - top).sum(1, keepdims=True)) + top
        resp = np.exp(log - total)
        mass = resp.sum(0)
        if (mass < 1e-9).any():
            break
        mean = (resp * x[:, None]).sum(0) / mass
        var = max(float((resp * (x[:, None] - mean[None, :]) ** 2).sum() / len(x)), 1e-6)
        share = mass / len(x)
        now = float(total.sum())
        if abs(now - last) < 1e-9:
            break
        last = now
    order = np.argsort(mean)
    return float(mean[order[0]]), float(mean[order[1]]), var
