"""The rules of ``docs/architecture/quota-selection-experiment.md`` as pure functions of the score tables, plus the
numbers that explain a result: which factor matters, which kinds of tissue a set covers, which descriptors."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import selection as sel
from .selection_cli import COST_ORDER
from .tile_descriptors import FEATURES

CONFIG = ["model", "k", "allocation", "method"]
NOISE = "noise-control"
RANDOM = "random"
RECOMMENDED_MODEL = "lunit-dino-s16"  # with one tile per cluster: the recommendation of the tile-selection experiment
REF_RECOMMENDED = "dotychczasowa (lunit, 1 z klastra)"
REF_SAME = "ten sam model, 1 z klastra"
REF_RANDOM = "losowa"
SCORES = ("coverage_mean", "coverage_p95", "rare5_categories", "rare5_tiles", "rare1_categories", "rare1_tiles", "jsd_groups",
          "diversity_mean", "near_duplicates", "low_tissue", "nuclei", "grid_cells", "nearest_neighbour_mm", "overlapping_pairs",
          "n_selected", "fallbacks")
FEATURE_GROUPS = {"pole tkanki (biel, tłuszcz)": ("white", "fat"), "krew i barwnik": ("erythrocyte", "pigment"), "kość": ("compact_eosin",),
                  "jądra (4 cechy)": ("nuclear_area", "nuclear_density", "nuclear_size", "large_nuclei"),
                  "barwienie (3 cechy)": ("h_median", "e_median", "h_high"), "tekstura": ("texture",)}
TESTS = ("nowe seedy", "fragmenty tkanki", "usuwanie cech", "podpróbki", "warianty przestrzeni")


def key_of(model: str, k: int, allocation: str, method: str) -> str:
    return f"{model}|{k}|{allocation}|{method}"


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd over single selections (seeds, and draws where there are any) per test, part and configuration."""
    keys = ["test", "part", *CONFIG]
    grouped = scores.groupby(keys, sort=False)[list(SCORES)]
    table = grouped.mean().join(grouped.std(ddof=0).add_suffix("_sd")).reset_index()
    return table.merge(grouped.size().rename("selections").reset_index(), on=keys)


def pooled(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt(0.5 * (sd_a**2 + sd_b**2)))


def apply_rule(summary: pd.DataFrame, count: int = 3) -> tuple[pd.DataFrame, float]:
    """The frozen rule on the grid. Returns the ``count`` best configurations in order and the low-tissue limit.

    Eligible: no more low-tissue tiles than random selection has on average. Then lowest mean coverage; configurations
    within the pooled sd of it tie and are ordered by more rare 1% categories, lower divergence, cheaper model.
    """
    grid = summary[summary["test"] == "grid"]
    limit = float(grid[grid["model"] == RANDOM]["low_tissue"].iloc[0])
    candidates = grid[grid["method"].isin(["m1", "m1p", "m2", "m3"]) & (grid["model"] != NOISE)].copy()
    candidates = candidates[candidates["low_tissue"] <= limit + 1e-12]
    if candidates.empty:
        raise ValueError("no configuration has as few low-tissue tiles as random selection")
    cost = {name: position for position, name in enumerate(COST_ORDER)}
    candidates["cost"] = [cost.get(m, len(cost)) for m in candidates["model"]]
    rows = []
    remaining = candidates
    while len(rows) < count and len(remaining):
        best = remaining.loc[remaining["coverage_mean"].idxmin()]
        gap = remaining["coverage_mean"] - best["coverage_mean"]
        threshold = np.sqrt(0.5 * (remaining["coverage_mean_sd"] ** 2 + best["coverage_mean_sd"] ** 2))
        tied = remaining[(gap < threshold) | (remaining.index == best.name)]
        winner = tied.sort_values(["rare1_categories", "jsd_groups", "cost", "coverage_mean", "k"],
                                  ascending=[False, True, True, True, True]).iloc[0]
        rows.append({**winner[CONFIG].to_dict(), "place": len(rows) + 1, "coverage_mean": winner["coverage_mean"],
                     "coverage_mean_sd": winner["coverage_mean_sd"], "best_coverage_in_tie": best["coverage_mean"], "tied": len(tied),
                     "eligible": len(candidates), "all": int(((grid["method"].isin(["m1", "m1p", "m2", "m3"])) & (grid["model"] != NOISE)).sum())})
        remaining = remaining.drop(winner.name)
    return pd.DataFrame(rows), limit


def one_per_cluster(model: str, total: int) -> tuple:
    """The configuration of the earlier method: ``total`` clusters, one tile each."""
    return (model, total, "one", "one")


def references(config: Mapping, total: int) -> dict[str, tuple]:
    """The three things a configuration is compared with."""
    return {REF_RECOMMENDED: one_per_cluster(RECOMMENDED_MODEL, total), REF_SAME: one_per_cluster(config["model"], total),
            REF_RANDOM: (RANDOM, 0, "-", RANDOM)}


def feature_removal(selections: pd.DataFrame, D: np.ndarray, configs: Sequence[tuple], *, draws: int = 200) -> pd.DataFrame:
    """Mean coverage of the grid selections of ``configs`` when one group of descriptors is left out of the score."""
    rows = []
    for config in configs:
        part = selections[selections["test"] == "grid"]
        for column, value in zip(CONFIG, config):
            part = part[part[column] == value]
        if config[0] == RANDOM:
            part = part.head(draws)
        for name, features in FEATURE_GROUPS.items():
            keep = [i for i, f in enumerate(FEATURES) if f not in features]
            values = [sel.coverage_on(D[:, keep], D[np.asarray(t, dtype=int)][:, keep])[0] for t in part["tiles"]]
            rows.append({**dict(zip(CONFIG, config)), "removed": name, "coverage_mean": float(np.mean(values))})
    return pd.DataFrame(rows)


def confirm(winners: pd.DataFrame, summary: pd.DataFrame, removal: pd.DataFrame, total: int) -> pd.DataFrame:
    """One row per (winner, reference, test, part) with the two mean coverages and whether the test is passed."""
    table = summary.set_index(["test", "part", *CONFIG]).sort_index()
    removed = removal.set_index(["removed", *CONFIG]).sort_index()
    rows = []
    for winner in winners.to_dict("records"):
        config = tuple(winner[c] for c in CONFIG)
        for reference_name, reference in references(winner, total).items():
            def add(test: str, part: str, a: pd.Series, b: pd.Series, needs_margin: bool = False):
                difference = float(b["coverage_mean"] - a["coverage_mean"])
                margin = pooled(a["coverage_mean_sd"], b["coverage_mean_sd"]) if needs_margin else 0.0
                rows.append({"place": winner["place"], **dict(zip(CONFIG, config)), "reference": reference_name, "test": test, "part": part,
                             "coverage": float(a["coverage_mean"]), "reference_coverage": float(b["coverage_mean"]),
                             "difference": difference, "margin": margin, "passed": bool(difference > margin)})

            def at(test: str, part: str, which: tuple) -> pd.Series:
                found = table.loc[(test, part, *which)]
                return found.iloc[0] if isinstance(found, pd.DataFrame) else found

            add("nowe seedy", "seedy 30–59", at("confirm_seeds", "-", config), at("confirm_seeds", "-", reference), True)
            for part in sorted(summary[summary["test"] == "fragment"]["part"].unique()):
                add("fragmenty tkanki", part, at("fragment", part, config), at("fragment", part, reference))
            for group in FEATURE_GROUPS:
                a = removed.loc[(group, *config)]
                b = removed.loc[(group, *reference)]
                a, b = (x.iloc[0] if isinstance(x, pd.DataFrame) else x for x in (a, b))
                difference = float(b["coverage_mean"] - a["coverage_mean"])
                rows.append({"place": winner["place"], **dict(zip(CONFIG, config)), "reference": reference_name, "test": "usuwanie cech",
                             "part": group, "coverage": float(a["coverage_mean"]), "reference_coverage": float(b["coverage_mean"]),
                             "difference": difference, "margin": 0.0, "passed": bool(difference > 0)})
            add("podpróbki", "80% kafli", at("subsample", "-", config), at("subsample", "-", reference))
            for variant in sorted(summary[summary["test"] == "variant"]["part"].unique()):
                add("warianty przestrzeni", variant, at("variant", variant, config), at("variant", variant, reference))
    return pd.DataFrame(rows)


def verdicts(confirmation: pd.DataFrame) -> pd.DataFrame:
    """Per winner: for each reference the tests passed; ``confirmed`` = all tests against the recommendation and random."""
    rows = []
    for place, group in confirmation.groupby("place", sort=True):
        row = {"place": int(place), **group.iloc[0][CONFIG].to_dict()}
        required = []
        for reference, part in group.groupby("reference", sort=False):
            by_test = part.groupby("test", sort=False)["passed"].all()
            row[f"passed: {reference}"] = ", ".join(t for t in TESTS if by_test.get(t, False))
            row[f"failed: {reference}"] = ", ".join(t for t in TESTS if not by_test.get(t, False))
            if reference != REF_SAME:
                required.append(bool(by_test.all()))
        row["confirmed"] = all(required)
        rows.append(row)
    return pd.DataFrame(rows)


def factor_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean coverage of all grid configurations of the four models sharing one level of one factor."""
    grid = summary[(summary["test"] == "grid") & summary["method"].isin(["m1", "m1p", "m2", "m3"]) & (summary["model"] != NOISE)]
    rows = []
    for factor in ("method", "allocation", "k", "model"):
        for level, part in grid.groupby(factor, sort=True):
            rows.append({"factor": factor, "level": level, "coverage_mean": float(part["coverage_mean"].mean()),
                         "coverage_best": float(part["coverage_mean"].min()), "coverage_worst": float(part["coverage_mean"].max()),
                         "rare1_categories": float(part["rare1_categories"].mean()), "jsd_groups": float(part["jsd_groups"].mean()),
                         "low_tissue": float(part["low_tissue"].mean()), "configurations": len(part)})
    return pd.DataFrame(rows)


def stability(selections: pd.DataFrame, positions: np.ndarray, size: int, *, seeds: int = 30) -> pd.DataFrame:
    """Agreement of the tiles chosen by different seeds (one draw each), per grid configuration."""
    part = selections[(selections["test"] == "grid") & (selections["draw"] == 0) & (selections["seed"] < seeds)]
    rows = []
    for config, group in part.groupby(CONFIG, sort=False):
        picks = [np.asarray(t, dtype=int) for t in group.sort_values("seed")["tiles"]]
        if len(picks) < 2:
            continue
        shares = []
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                near = sel.overlap_matrix(positions[np.concatenate([picks[i], picks[j]])], size)[: len(picks[i]), len(picks[i]):]
                identical = len(set(picks[i].tolist()) & set(picks[j].tolist()))
                # share of the first set that has an identical or overlapping tile in the second one
                shares.append((identical / len(picks[i]), float(near.any(axis=1).mean())))
        values = np.array(shares)
        rows.append({**dict(zip(CONFIG, config)), "identical": float(values[:, 0].mean()), "identical_or_overlapping": float(values[:, 1].mean())})
    return pd.DataFrame(rows)


def median_selection(scores: pd.DataFrame) -> pd.Series:
    """The row of lower-median coverage (ties: smaller seed, then smaller draw)."""
    ordered = scores.sort_values(["coverage_mean", "seed", "draw"])
    return ordered.iloc[(len(ordered) - 1) // 2]


def explain_sets(selections: pd.DataFrame, D: np.ndarray, groups: np.ndarray, sets: Mapping[str, tuple], *, draws: int = 200
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per set: coverage and number of chosen tiles per tissue group, and the mean absolute difference per descriptor
    between a tile and the selected tile nearest to it. Averages over the grid selections of that configuration."""
    by_group, by_feature = [], []
    for name, config in sets.items():
        part = selections[selections["test"] == "grid"]
        for column, value in zip(CONFIG, config):
            part = part[part[column] == value]
        part = part.head(draws)
        distance = np.zeros(len(D))
        difference = np.zeros(D.shape[1])
        counts = np.zeros(int(groups.max()) + 1)
        for tiles in part["tiles"]:
            tiles = np.asarray(tiles, dtype=int)
            squared = (D**2).sum(axis=1)[:, None] + (D[tiles] ** 2).sum(axis=1)[None, :] - 2.0 * D @ D[tiles].T
            closest = tiles[np.argmin(squared, axis=1)]
            distance += np.sqrt(((D - D[closest]) ** 2).sum(axis=1))
            difference += np.abs(D - D[closest]).mean(axis=0)
            counts += np.bincount(groups[tiles], minlength=len(counts))
        distance, difference, counts = distance / len(part), difference / len(part), counts / len(part)
        for g in range(len(counts)):
            by_group.append({"set": name, "group": g, "coverage": float(distance[groups == g].mean()), "chosen": float(counts[g]),
                             "tiles": int((groups == g).sum())})
        by_feature += [{"set": name, "feature": f, "difference": float(v)} for f, v in zip(FEATURES, difference)]
    return pd.DataFrame(by_group), pd.DataFrame(by_feature)


def describe_groups(table: pd.DataFrame, D: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per tissue group: size, mean raw descriptors and the three descriptors that differ most from the slide."""
    rows = []
    for g in range(int(groups.max()) + 1):
        mask = groups == g
        z = D[mask].mean(axis=0)
        strongest = np.argsort(-np.abs(z))[:3]
        rows.append({"group": g, "tiles": int(mask.sum()), "share": float(mask.mean()),
                     **{f: float(table[f].to_numpy()[mask].mean()) for f in FEATURES},
                     "typical": "; ".join(f"{FEATURES[i]}{'+' if z[i] > 0 else '-'}{abs(z[i]):.1f}" for i in strongest)})
    return pd.DataFrame(rows)
