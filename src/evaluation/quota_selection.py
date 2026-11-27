"""Several tiles per cluster: how many each cluster gets, which ones it gives, and how the resulting set is scored.

Protocol: ``docs/architecture/quota-selection-experiment.md``. A selection of ``total`` tiles never contains two tiles
that share area. All scores use the pixel descriptors or tile positions, never an embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import selection as sel
from .tile_descriptors import RARE_FEATURES

ALLOCATIONS = ("equal", "proportional")
METHODS = ("m1", "m1p", "m2", "m3")
RANDOM_METHOD = "m2"
TRIM = 0.05  # m1p ignores this share of a cluster: the members least like its centre
LOW_TISSUE_WHITE = 0.7
RARE_SHARES = (0.05, 0.01)


def quotas(sizes: Sequence[int], total: int, rule: str) -> np.ndarray:
    """Places per cluster; ``sizes`` ordered from the largest cluster to the smallest, sums to ``total``."""
    sizes = np.asarray(sizes, dtype=float)
    k = len(sizes)
    if k == 0 or k > total:
        raise ValueError(f"cannot give {total} places to {k} clusters with at least one each")
    if rule == "equal":
        out = np.full(k, total // k)
        out[: total % k] += 1
        return out
    if rule != "proportional":
        raise ValueError(f"unknown allocation {rule!r}")
    exact = sizes / sizes.sum() * total
    out = np.maximum(np.floor(exact).astype(int), 1)
    while out.sum() < total:  # largest remainders first, the larger cluster on equal remainders
        out[int(np.argmax(exact - out))] += 1
    while out.sum() > total:  # the minimum of one pushed the sum over: take from the most over-served cluster that can spare one
        spare = np.flatnonzero(out > 1)
        out[spare[int(np.argmax((out - exact)[spare]))]] -= 1
    return out


@dataclass
class Picks:
    """A selection: tile indices, the number of the cluster each came from (0 = largest) and its turn within the cluster."""

    tiles: np.ndarray
    cluster: np.ndarray
    turn: np.ndarray
    fallbacks: int

    def __len__(self) -> int:
        return len(self.tiles)


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def cluster_order(labels: np.ndarray) -> np.ndarray:
    """Cluster ids from the largest to the smallest."""
    clusters, sizes = np.unique(labels, return_counts=True)
    return clusters[np.argsort(-sizes, kind="stable")]


def select(
    vectors: np.ndarray,
    labels: np.ndarray,
    centres: np.ndarray,
    total: int,
    *,
    allocation: str,
    method: str,
    conflicts: np.ndarray,
    rng: np.random.Generator | None = None,
    trim: float = TRIM,
) -> Picks:
    """``total`` tiles taken from the clusters in turns (largest cluster first), see the module docstring.

    ``vectors`` and ``centres`` are in the clustering space; similarities are cosine. ``conflicts`` is the ``[n, n]``
    overlap matrix (diagonal True). A cluster that runs out of free members hands its places to the largest clusters
    that still have one (counted in ``fallbacks``). Fewer than ``total`` tiles come back only if the slide runs out.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == RANDOM_METHOD and rng is None:
        raise ValueError("random selection within clusters needs a generator")
    unit = _unit(vectors)
    centre_unit = _unit(centres)
    order = cluster_order(labels)
    members = [np.flatnonzero(labels == c) for c in order]
    wanted = quotas([len(m) for m in members], total, allocation)
    similarity = [unit[m] @ centre_unit[c] for m, c in zip(members, order)]
    allowed = []
    for m, s in zip(members, similarity):
        keep = np.ones(len(m), dtype=bool)
        if method == "m1p":
            drop = int(np.floor(trim * len(m)))
            keep[np.argsort(s, kind="stable")[:drop]] = False
        allowed.append(keep)
    nearest = [np.full(len(m), np.inf) for m in members]  # cosine distance to the closest pick of the same cluster
    blocked = np.zeros(len(vectors), dtype=bool)
    taken: list[list[int]] = [[] for _ in members]
    chosen: list[tuple[int, int, int]] = []

    def pick(number: int) -> bool:
        free = np.flatnonzero(allowed[number] & ~blocked[members[number]])
        if len(free) == 0:
            return False
        if method == RANDOM_METHOD:
            local = int(rng.choice(free))
        elif method in ("m1", "m1p") and taken[number]:
            local = int(free[np.argmax(nearest[number][free])])
        else:  # nearest to the centre: every pick of m3, the first one of m1 and m1p
            local = int(free[np.argmax(similarity[number][free])])
        tile = int(members[number][local])
        chosen.append((tile, number, len(taken[number])))
        taken[number].append(tile)
        blocked[conflicts[tile]] = True
        blocked[tile] = True
        if method in ("m1", "m1p"):
            nearest[number] = np.minimum(nearest[number], 1.0 - unit[members[number]] @ unit[tile])
        return True

    for turn in range(int(wanted.max())):
        for number in range(len(members)):
            if turn < wanted[number] and len(chosen) < total:
                pick(number)
    fallbacks = 0
    while len(chosen) < total:
        progressed = False
        for number in range(len(members)):
            if len(chosen) < total and pick(number):
                fallbacks += 1
                progressed = True
        if not progressed:
            break
    chosen.sort(key=lambda item: (item[1], item[2]))
    tiles, cluster, turn = (np.array([c[i] for c in chosen], dtype=int) for i in range(3))
    return Picks(tiles, cluster, turn, fallbacks)


def one_cluster(vectors: np.ndarray, total: int, conflicts: np.ndarray) -> Picks:
    """No clustering: the tile nearest the mean of the slide, then the most different ones (M1 on everything)."""
    centre = _unit(vectors).mean(axis=0, keepdims=True)
    return select(vectors, np.zeros(len(vectors), dtype=int), centre, total, allocation="equal", method="m1", conflicts=conflicts)


class ValueScorer:
    """Scores sets of tiles of one slide (or of the tiles ``tiles`` of it, for the tissue-fragment test)."""

    def __init__(self, manifest: pd.DataFrame, table: pd.DataFrame, descriptors: np.ndarray, groups: np.ndarray,
                 *, tiles: np.ndarray | None = None):
        tiles = np.arange(len(manifest)) if tiles is None else np.asarray(tiles, dtype=int)
        self.tiles = tiles
        self.flags = {share: sel.rare_flags(table, RARE_FEATURES, share) for share in RARE_SHARES}
        self.D = descriptors
        self.groups = groups
        self.group_counts = np.bincount(groups[tiles], minlength=int(groups.max()) + 1).astype(float)
        self.white = table["white"].to_numpy()
        side_mm = float(manifest["width"].iloc[0]) * float(manifest["mpp"].iloc[0]) / 1000.0
        self.nuclei = table["nuclear_density"].to_numpy() * side_mm**2
        self.mm_per_px = float(manifest["mpp"].iloc[0]) / 1000.0
        self.positions = manifest[["x", "y"]].to_numpy(dtype=float)
        self.size = int(manifest["width"].iloc[0])
        _, self.cell = np.unique((self.positions // 4096).astype(int), axis=0, return_inverse=True)
        pairs = np.random.default_rng(0).integers(len(tiles), size=(200000, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        distance = np.sqrt(((descriptors[tiles[pairs[:, 0]]] - descriptors[tiles[pairs[:, 1]]]) ** 2).sum(axis=1))
        self.duplicate_below = float(np.quantile(distance, 0.01))

    def score(self, selected: np.ndarray) -> dict[str, float]:
        selected = np.asarray(selected, dtype=int)
        mean_distance, p95 = sel.coverage_on(self.D[self.tiles], self.D[selected])
        chosen = self.D[selected]
        between = np.sqrt(((chosen[:, None, :] - chosen[None, :, :]) ** 2).sum(axis=2))[np.triu_indices(len(selected), 1)]
        points = self.positions[selected]
        apart = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(apart, np.inf)
        shared = sel.overlap_matrix(points, self.size)
        out = {"n_selected": float(len(selected)), "coverage_mean": mean_distance, "coverage_p95": p95,
               "jsd_groups": sel.jensen_shannon(np.bincount(self.groups[selected], minlength=len(self.group_counts)).astype(float),
                                                self.group_counts),
               "diversity_mean": float(between.mean()), "near_duplicates": float((between < self.duplicate_below).sum()),
               "low_tissue": float((self.white[selected] > LOW_TISSUE_WHITE).sum()), "nuclei": float(self.nuclei[selected].sum()),
               "grid_cells": float(len(np.unique(self.cell[selected]))),
               "nearest_neighbour_mm": float(apart.min(axis=1).mean() * self.mm_per_px),
               "overlapping_pairs": float((shared.sum() - len(selected)) / 2)}
        for share, flags in self.flags.items():
            found = flags[selected]
            suffix = f"{round(100 * share)}"
            out[f"rare{suffix}_categories"] = float(found.any(axis=0).sum())
            out[f"rare{suffix}_tiles"] = float(found.any(axis=1).sum())
        return out

