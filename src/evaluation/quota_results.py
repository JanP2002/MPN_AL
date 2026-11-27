"""The files of one run of the 50-tiles-from-clusters experiment, with the Polish names the reports use."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import quota_analysis as qa
from .tile_descriptors import RARE_FEATURES, standardize

MODELS = ["lunit-dino-s16", "midnight-12k", "phikon-v2", "dinov2-s14"]
METHOD_NAMES = {"m1": "M1: centroid + najbardziej różne", "m1p": "M1p: jak M1, bez 5% skrajnych", "m2": "M2: losowo w klastrze",
                "m3": "M3: najbliższe centroidowi", "one": "1 kafel z klastra", "no-clusters": "bez klastrowania",
                "random": "losowa", "spatial": "przestrzenna"}
METHOD_SHORT = {"m1": "M1", "m1p": "M1p", "m2": "M2", "m3": "M3", "one": "1 z klastra", "no-clusters": "bez klastrów", "random": "losowa",
                "spatial": "przestrzenna"}
ALLOCATION_NAMES = {"equal": "równy", "proportional": "proporcjonalny", "one": "po 1", "-": "–"}
MODEL_NAMES = {"random": "losowa", "spatial": "przestrzenna", "descriptor-kmeans": "pułap (cechy z pikseli)", "noise-control": "kontrola szumem"}
SET_NAMES = {"zwycięzca siatki": "zwycięzca siatki"}


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def config_name(model: str, k: int, allocation: str, method: str) -> str:
    """A configuration in words, e.g. ``midnight-12k · 8 klastrów · przydział równy · M2``."""
    if method in ("random", "spatial"):
        return f"{MODEL_NAMES[method]} (bez modelu)"
    if model == "descriptor-kmeans":
        return "pułap: KMeans na cechach z pikseli"
    if method == "one":
        return f"{model_name(model)} · {int(k)} klastrów po 1 kaflu"
    if method == "no-clusters":
        return f"{model_name(model)} · bez klastrowania"
    return f"{model_name(model)} · {int(k)} klastrów · przydział {ALLOCATION_NAMES[allocation]} · {METHOD_SHORT[method]}"


class Run:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        read = lambda name: pd.read_csv(self.folder / f"{name}.csv")
        self.summary = read("summary")
        self.winners = read("winners")
        self.confirmation = read("confirmation")
        self.verdicts = read("verdicts").fillna("")
        self.factors = read("factor_effects")
        self.stability = read("stability")
        self.metrics = read("clustering_metrics")
        self.groups = read("tissue_groups")
        self.features = read("feature_coverage")
        self.group_table = read("group_descriptions")
        self.representative = read("representative")
        self.scores = read("selection_scores")
        self.run = json.loads((self.folder / "run.json").read_text(encoding="utf-8"))
        self.manifest = pd.read_parquet(self.folder / "manifest.parquet")
        self.table = pd.read_parquet(self.folder / "descriptors.parquet")
        self.D = standardize(self.table)
        self.selections = pd.read_parquet(self.folder / "selections.parquet")
        args = self.run["args"]
        self.total = int(args["total"])
        self.ks = sorted(int(k) for k in args["ks"])
        self.seeds = int(args["seeds"])
        self.models = [m for m in MODELS if m in self.run["models"]] + [m for m in self.run["models"] if m not in MODELS]
        self.rare_features = RARE_FEATURES

    @property
    def winner(self) -> tuple:
        w = self.winners.iloc[0]
        return (w["model"], int(w["k"]), w["allocation"], w["method"])

    def grid(self, config: tuple) -> pd.Series:
        """The grid summary row of one configuration."""
        s = self.summary[self.summary["test"] == "grid"]
        for column, value in zip(qa.CONFIG, config):
            s = s[s[column] == value]
        if len(s) != 1:
            raise KeyError(f"no single summary row for {config}")
        return s.iloc[0]

    def baselines(self) -> dict[str, tuple]:
        """The comparison sets every table repeats: earlier method, whole-slide diversity, random, ceiling."""
        out = {qa.REF_RECOMMENDED: qa.one_per_cluster(qa.RECOMMENDED_MODEL, self.total)}
        model = self.winner[0]
        if model != qa.RECOMMENDED_MODEL:
            out[qa.REF_SAME] = qa.one_per_cluster(model, self.total)
        out["bez klastrowania (model zwycięzcy)"] = (model, 1, "-", "no-clusters")
        out[qa.REF_RANDOM] = ("random", 0, "-", "random")
        out["przestrzenna"] = ("spatial", 0, "-", "spatial")
        out["pułap (cechy z pikseli)"] = ("descriptor-kmeans", self.total, "one", "one")
        return out

    def chosen(self, set_name: str) -> dict:
        """The representative (median coverage) selection of one set: its row in ``selections``."""
        r = self.representative[self.representative["set"] == set_name].iloc[0]
        part = self.selections[self.selections["test"] == "grid"]
        for column in qa.CONFIG:
            part = part[part[column] == r[column]]
        row = part[(part["seed"] == r["seed"]) & (part["draw"] == r["draw"])].iloc[0]
        return {**row.to_dict(), "tiles": np.asarray(row["tiles"], dtype=int), "cluster": np.asarray(row["cluster"], dtype=int),
                "turn": np.asarray(row["turn"], dtype=int), "coverage_mean": float(r["coverage_mean"])}

    def confirmed_config(self) -> tuple[tuple, bool]:
        """The configuration the reports recommend: the first confirmed winner, otherwise the earlier recommendation."""
        confirmed = self.verdicts[self.verdicts["confirmed"] == True]  # noqa: E712 - the column holds booleans
        if len(confirmed):
            v = confirmed.sort_values("place").iloc[0]
            return (v["model"], int(v["k"]), v["allocation"], v["method"]), True
        return qa.one_per_cluster(qa.RECOMMENDED_MODEL, self.total), False
