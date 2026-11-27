"""How many tiles per round? The rules of ``docs/architecture/k-selection-experiment.md``, applied to selection scores.

Coverage improves with every added tile, so different K are compared through ``G``, the share of the way from random
selection (0) to KMeans on the descriptors (1) at the same K. Everything here is a pure function of the score table the
experiment writes (one row per selection), so the rules can be tested and re-applied without clustering again.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd

from . import selection as sel
from .selection_cli import COST_ORDER

MAIN = "l2"
NOISE = "noise-control"
RANDOM = "random"
CEILING = "descriptor-kmeans"
SCORE_COLUMNS = ("coverage_mean", "coverage_p95", "rare_categories", "jsd_groups", "grid_cells", "diversity_mean",
                 "nearest_neighbour_mm", "overlapping_pairs")


def comparable(scores: pd.DataFrame) -> pd.DataFrame:
    """The rows the analysis compares: selections without overlap, plus the main variant with overlap allowed as ``free``."""
    free = scores[(scores["variant"] == MAIN) & (scores["constraint"] == "free")].assign(variant="free")
    return pd.concat([scores[scores["constraint"] == "no-overlap"], free], ignore_index=True)


def add_gap(scores: pd.DataFrame) -> pd.DataFrame:
    """Adds ``gap`` (G) and ``beats_random`` (share of random draws with worse coverage, ties counted half)."""
    base = scores[(scores["variant"] == "baseline") & (scores["constraint"] == "no-overlap")]
    random = {k: np.sort(g["coverage_mean"].to_numpy()) for k, g in base[base["method"] == RANDOM].groupby("k")}
    ceiling = base[base["method"] == CEILING].groupby("k")["coverage_mean"].mean()
    missing = sorted(set(scores["k"]) - (set(random) & set(ceiling.index)))
    if missing:
        raise ValueError(f"no random or descriptor-kmeans selections for K = {missing}")
    level = scores["k"].map({k: v.mean() for k, v in random.items()}).to_numpy()
    top = scores["k"].map(ceiling).to_numpy()
    if (level - top <= 0).any():
        raise ValueError("descriptor-kmeans does not cover better than random: G is undefined")
    beats = np.empty(len(scores))
    for i, (k, value) in enumerate(zip(scores["k"], scores["coverage_mean"])):
        draws = random[k]
        worse = len(draws) - np.searchsorted(draws, value, side="right")
        equal = np.searchsorted(draws, value, side="right") - np.searchsorted(draws, value, side="left")
        beats[i] = (worse + 0.5 * equal) / len(draws)
    return scores.assign(gap=(level - scores["coverage_mean"].to_numpy()) / (level - top), beats_random=beats)


def min_jsd(group_counts: Sequence[float], k: int) -> float:
    """The smallest Jensen-Shannon divergence any ``k`` tiles can reach against the groups' shares of the slide."""
    counts = np.asarray(group_counts, dtype=float)
    best = np.inf
    for picks in itertools.combinations_with_replacement(range(len(counts)), k):
        histogram = np.bincount(picks, minlength=len(counts)).astype(float)
        if (histogram <= counts).all():
            best = min(best, sel.jensen_shannon(histogram, counts))
    return float(best)


def summarize_k(scores: pd.DataFrame, group_counts: Sequence[float]) -> pd.DataFrame:
    """Mean and sd over seeds (or draws) per (variant, method, k) of :func:`comparable` rows with ``gap``."""
    keys = ["variant", "method", "k"]
    columns = ["gap", "beats_random", *SCORE_COLUMNS]
    grouped = scores.groupby(keys, sort=False)
    table = grouped[columns].mean().join(grouped[columns].std(ddof=0).add_suffix("_sd")).reset_index()
    table = table.merge(grouped.size().rename("n").reset_index(), on=keys)
    table = table.merge(grouped["beats_random"].min().rename("beats_random_min").reset_index(), on=keys)
    floor = {k: min_jsd(group_counts, int(k)) for k in sorted(table["k"].unique())}
    table["jsd_floor"] = table["k"].map(floor)
    table["jsd_excess"] = table["jsd_groups"] - table["jsd_floor"]
    return table


def tie_threshold(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt(0.5 * (sd_a**2 + sd_b**2)))


def choose_k(summary: pd.DataFrame) -> pd.DataFrame:
    """Rule 1 per (variant, model): best mean G; K within the pooled sd of it tie; the smallest tied K is recommended."""
    rows = []
    part = summary[summary["variant"] != "baseline"]
    for (variant, method), group in part.groupby(["variant", "method"], sort=False):
        group = group.sort_values("k")
        best = group.loc[group["gap"].idxmax()]
        tied = [int(r.k) for r in group.itertuples()
                if r.k == best["k"] or best["gap"] - r.gap < tie_threshold(r.gap_sd, best["gap_sd"])]
        recommended = min(tied)
        chosen = group[group["k"] == recommended].iloc[0]
        rows.append({"variant": variant, "method": method, "best_k": int(best["k"]), "best_gap": float(best["gap"]),
                     "tied_k": " ".join(map(str, tied)), "recommended_k": recommended, "recommended_gap": float(chosen["gap"]),
                     "all_tied": len(tied) == len(group), "gap_range": float(group["gap"].max() - group["gap"].min())})
    return pd.DataFrame(rows)


def order_models(scores: pd.DataFrame, variant: str = MAIN) -> pd.DataFrame:
    """Rule 2: G averaged over K for every seed, then over seeds; gaps below the pooled sd are ties the cheaper model wins."""
    part = scores[(scores["variant"] == variant) & (scores["method"] != NOISE)]
    per_seed = part.groupby(["method", "seed"])["gap"].mean()
    table = per_seed.groupby("method").agg(gap="mean", gap_sd=lambda s: float(s.std(ddof=0))).reset_index()
    cost = {name: position for position, name in enumerate(COST_ORDER)}
    remaining = sorted(table.to_dict("records"), key=lambda r: (-r["gap"], cost.get(r["method"], len(cost))))
    ordered = []
    while remaining:
        best = remaining[0]
        tied = [r for r in remaining if r is best or best["gap"] - r["gap"] < tie_threshold(r["gap_sd"], best["gap_sd"])]
        winner = min(tied, key=lambda r: (cost.get(r["method"], len(cost)), -r["gap"]))
        ordered.append({**winner, "tied_with": ", ".join(r["method"] for r in tied if r is not winner)})
        remaining = [r for r in remaining if r is not winner]
    out = pd.DataFrame(ordered)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def median_item(values: pd.Series) -> int:
    """The index label of the lower median of ``values`` (ties broken by the smaller label): rule 5."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return int(ordered[(len(ordered) - 1) // 2][0])


def representative_seeds(scores: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """Per (method, k): the seed (random: the draw) of median coverage; ``k = 0`` is the one seed used across all K."""
    rows = []
    for method, group in scores[scores["variant"].isin([MAIN, "baseline"]) & (scores["constraint"] == "no-overlap")].groupby("method", sort=False):
        for k in ks:
            part = group[group["k"] == k].set_index("seed")["coverage_mean"]
            rows.append({"method": method, "k": int(k), "seed": median_item(part)})
        across = group[group["k"].isin(ks)].groupby("seed")["coverage_mean"].agg(["mean", "size"])
        across = across[across["size"] == len(ks)]["mean"]
        if len(across):
            rows.append({"method": method, "k": 0, "seed": median_item(across)})
    return pd.DataFrame(rows)


def marginal_gains(summary: pd.DataFrame) -> pd.DataFrame:
    """What one more tile buys: the drop of mean coverage and the change of G from K to the next K."""
    rows = []
    part = summary[summary["variant"].isin([MAIN, "baseline"])]
    for method, group in part.groupby("method", sort=False):
        group = group.sort_values("k")
        for a, b in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            rows.append({"method": method, "k_from": int(a.k), "k_to": int(b.k), "coverage_drop": a.coverage_mean - b.coverage_mean,
                         "gap_change": b.gap - a.gap})
    return pd.DataFrame(rows)


def nesting(selected: dict[tuple, np.ndarray], positions: np.ndarray, size: int) -> pd.DataFrame:
    """Share of a K selection found again (the same tile, or an overlapping one) in the K+1 selection of the same seed."""
    series: dict[tuple, dict[int, np.ndarray]] = {}
    for (variant, method, k, seed, constraint), indices in selected.items():
        if variant in (MAIN, "baseline") and constraint == "no-overlap":
            series.setdefault((method, seed), {})[k] = indices
    values: dict[tuple, list[dict]] = {}
    for (method, _), by_k in series.items():
        ks = sorted(by_k)
        for a, b in zip(ks[:-1], ks[1:]):
            values.setdefault((method, a, b), []).append(sel.agreement(by_k[a], by_k[b], positions, size))
    return pd.DataFrame([{"method": m, "k_from": a, "k_to": b, "identical": float(np.mean([v["identical_share"] for v in rows])),
                          "identical_or_overlapping": float(np.mean([v["matched_share"] for v in rows]))}
                         for (m, a, b), rows in values.items()])


def selection_stability(selected: dict[tuple, np.ndarray], positions: np.ndarray, size: int) -> pd.DataFrame:
    """Per (variant, method, k): agreement of the chosen tiles between all pairs of seeds (or subsamples, or draws up to 30)."""
    groups: dict[tuple, dict[int, np.ndarray]] = {}
    for (variant, method, k, seed, constraint), indices in selected.items():
        if constraint == "no-overlap" and (variant != "baseline" or seed < 30):
            groups.setdefault((variant, method, k), {})[seed] = indices
    rows = []
    for (variant, method, k), by_seed in groups.items():
        pairs = list(itertools.combinations(sorted(by_seed), 2))
        if not pairs:
            continue
        shared = [sel.agreement(by_seed[a], by_seed[b], positions, size) for a, b in pairs]
        rows.append({"variant": variant, "method": method, "k": k, "identical": float(np.mean([s["identical_share"] for s in shared])),
                     "identical_or_overlapping": float(np.mean([s["matched_share"] for s in shared]))})
    return pd.DataFrame(rows)
