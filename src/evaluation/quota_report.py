"""Report 1 (Polish) of the 50-tiles-from-clusters experiment, written for a reader who does not know the project:
what was asked, how the data were prepared, what every metric means, the results and why one way of choosing wins."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from . import quota_analysis as qa
from . import quota_selection as qs
from .cluster_figures import _plt, _save
from .quota_results import ALLOCATION_NAMES, METHOD_NAMES, METHOD_SHORT, Run, config_name, model_name
from .report_html import num, page, percent, picture, read_notes, table, to_pdf
from .selection_plates import FEATURE_NAMES, RARE_NAMES, Canvas, comma

MODEL_STYLE = {"dinov2-s14": {"color": "#0072B2", "marker": "o"}, "lunit-dino-s16": {"color": "#E69F00", "marker": "s"},
               "phikon-v2": {"color": "#009E73", "marker": "^"}, "midnight-12k": {"color": "#D55E00", "marker": "D"}}
METHOD_STYLE = {"m1": {"color": "#0072B2", "marker": "o"}, "m1p": {"color": "#CC79A7", "marker": "v"},
                "m2": {"color": "#E69F00", "marker": "s"}, "m3": {"color": "#009E73", "marker": "^"}}
SET_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#444444"]
METRICS = [  # column, name, direction, what it tells, how to read it
    ("coverage_mean", "pokrycie", "mniej", "Czy dla każdego kafla slajdu wśród 50 wybranych jest ktoś podobny.",
     "Dla każdego z kafli liczymy odległość (w 13 cechach) do najbliższego wybranego i uśredniamy. Jednostka to odchylenia standardowe cech; "
     "liczy się porównanie między metodami, nie sama wartość."),
    ("coverage_p95", "pokrycie, 95. percentyl", "mniej", "Czy jakaś część preparatu została pominięta.",
     "Odległość, której nie przekracza 95% kafli. Duża wartość: istnieją kafle bardzo niepodobne do wszystkich wybranych."),
    ("rare5_categories", "rzadkie kategorie (5%)", "więcej", "Czy do adnotacji trafiają rzadkie struktury.",
     "Z 6 kategorii (po 5% kafli z największą ilością kości, krwinek, barwnika, jąder, dużych jąder, bieli) liczba tych z choć jednym kaflem."),
    ("rare1_categories", "rzadkie kategorie (1%)", "więcej", "To samo dla naprawdę rzadkich, skrajnych kafli.",
     "Jak wyżej, ale próg to górny 1% kafli. Wartość 6 jest trudna do osiągnięcia."),
    ("jsd_groups", "rozbieżność Jensena-Shannona", "mniej", "Czy proporcje typów kafli w zestawie są takie jak na slajdzie.",
     "Kafle podzielono na 8 grup według cech; 0 oznacza identyczne proporcje, 1 zupełnie różne."),
    ("diversity_mean", "różnorodność wybranych", "więcej", "Jak bardzo wybrane kafle różnią się między sobą.",
     "Średnia odległość cech między parami wybranych. Sama duża wartość nie wystarcza: zestaw samych skrajności też ją ma."),
    ("near_duplicates", "pary prawie identyczne", "mniej", "Czy nie płacimy dwa razy za tę samą informację.",
     "Liczba par wybranych bliższych w cechach niż 1% losowych par kafli (na 1225 par możliwych przy 50 kaflach)."),
    ("low_tissue", "kafle mało użyteczne", "mniej", "Ile adnotacji zmarnuje się na prawie puste obrazy.",
     "Liczba wybranych kafli z ponad 70% bieli (szkło, tłuszcz)."),
    ("nuclei", "kandydaci na jądra", "opisowo", "Ile obiektów dostaje do oznaczenia adnotujący.",
     "Szacunek z barwienia hematoksyliną: gęstość jąder razy pole kafla, suma po 50 kaflach."),
    ("grid_cells", "zajęte pola slajdu", "opisowo", "Czy wybór nie skupia się w jednym miejscu.", "Liczba kwadratów ok. 0,9 mm z wybranym kaflem."),
]
FEATURE_MEANING = {
    "white": "udział niebarwionych pikseli (szkło, światło tłuszczu)", "fat": "udział okrągłych pustych pól wielkości adipocytów",
    "erythrocyte": "udział nasyconych pomarańczowo-czerwonych pikseli (krwinki czerwone)", "pigment": "udział ciemnobrązowych pikseli",
    "compact_eosin": "duże, gładkie różowe pola (beleczki kostne)", "nuclear_area": "udział pola zajętego przez kandydatów na jądra",
    "nuclear_density": "liczba kandydatów na jądra na mm²", "nuclear_size": "mediana pola jądra",
    "large_nuclei": "udział pola w jądrach większych niż 80 µm²", "h_median": "typowe natężenie hematoksyliny (jądra)",
    "e_median": "typowe natężenie eozyny (cytoplazma, kość)", "h_high": "95. percentyl hematoksyliny", "texture": "średnia siła krawędzi w obrazie"}
MODEL_FACTS = [
    ("lunit-dino-s16", "ViT-S/16, 21 mln", "384", "histopatologia: 19 mln kafli z TCGA (metoda DINO)"),
    ("phikon-v2", "ViT-L/16", "1024", "histopatologia: 456 mln kafli, 60 tys. preparatów (DINOv2)"),
    ("midnight-12k", "ViT-g/14", "3072 (token CLS + średnia tokenów)", "histopatologia: 12 tys. preparatów TCGA (DINOv2, zmodyfikowane)"),
    ("dinov2-s14", "ViT-S/14", "384", "zwykłe zdjęcia, nie histologia (DINOv2); punkt odniesienia"),
]


def short_set(name: str) -> str:
    return {"zwycięzca siatki": "zwycięzca siatki", qa.REF_RECOMMENDED: "lunit, 1 z klastra (dotychczasowa)",
            qa.REF_SAME: "model zwycięzcy, 1 z klastra"}.get(name, name)


# ---- figures ----------------------------------------------------------------------------------


def method_illustration(path: Path) -> Path:
    """The four ways of taking 8 tiles from one cluster, on invented 2-D points (not data)."""
    plt = _plt()
    rng = np.random.default_rng(3)
    points = np.concatenate([rng.normal((0, 0), 0.55, (170, 2)), rng.normal((1.7, 0.9), 0.3, (35, 2)), [[-2.1, 1.7], [2.2, -1.7]]])
    vectors = np.column_stack([points, np.full(len(points), 6.0)])  # nearly flat on the sphere: cosine behaves like plane distance
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.zeros(len(points), dtype=int)
    centre = unit.mean(axis=0, keepdims=True)
    free = np.eye(len(points), dtype=bool)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1))
    for ax, method in zip(axes, qs.METHODS):
        picks = qs.select(unit, labels, centre, 8, allocation="equal", method=method, conflicts=free, rng=np.random.default_rng(5))
        ax.scatter(points[:, 0], points[:, 1], s=9, color="#b9bdc4", linewidths=0)
        ax.scatter(*points.mean(axis=0), marker="+", s=160, color="black")
        order = np.argsort(picks.turn)
        for rank, tile in enumerate(picks.tiles[order], start=1):
            ax.scatter(*points[tile], s=150, color=METHOD_STYLE[method]["color"], edgecolors="black", linewidths=0.8, zorder=3)
            ax.annotate(str(rank), points[tile], ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
        ax.set_title(METHOD_NAMES[method], fontsize=10.5)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    return _save(fig, path)


def _reference_lines(r: Run, ax, column: str):
    lines = [(qa.one_per_cluster(qa.RECOMMENDED_MODEL, r.total), "lunit, 1 z klastra", "#E69F00", "--"),
             (("random", 0, "-", "random"), "losowa", "#000000", ":"),
             (("descriptor-kmeans", r.total, "one", "one"), "pułap", "#56B4E9", "-.")]
    for config, name, colour, style in lines:
        ax.axhline(r.grid(config)[column], color=colour, linestyle=style, linewidth=1.6, label=name)


def _legend(fig, axes, columns: int):
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=columns, fontsize=9.5, frameon=False)


def curves_coverage(r: Run, path: Path) -> Path:
    plt = _plt()
    grid = r.summary[r.summary["test"] == "grid"]
    allocations = list(qs.ALLOCATIONS)
    fig, axes = plt.subplots(len(allocations), len(qs.METHODS), figsize=(16, 7.4), sharey=True, squeeze=False)
    for row, allocation in enumerate(allocations):
        for ax, method in zip(axes[row], qs.METHODS):
            for model in r.models:
                part = grid[(grid["model"] == model) & (grid["method"] == method) & (grid["allocation"] == allocation)].sort_values("k")
                ax.plot(part["k"], part["coverage_mean"], label=model, linewidth=2, markersize=5, **MODEL_STYLE.get(model, {}))
                ax.fill_between(part["k"], part["coverage_mean"] - part["coverage_mean_sd"], part["coverage_mean"] + part["coverage_mean_sd"],
                                color=MODEL_STYLE.get(model, {}).get("color"), alpha=0.08, linewidth=0)
            _reference_lines(r, ax, "coverage_mean")
            ax.set_title(f"{METHOD_SHORT[method]}, przydział {ALLOCATION_NAMES[allocation]}", fontsize=10.5)
            ax.set_xticks(r.ks), ax.grid(alpha=0.25), ax.set_xlabel("liczba klastrów K")
        axes[row][0].set_ylabel("pokrycie (mniej = lepiej)")
    values = grid[grid["method"].isin(qs.METHODS) & grid["model"].isin(r.models)]["coverage_mean"]
    references = [r.grid(("descriptor-kmeans", r.total, "one", "one"))["coverage_mean"], r.grid(("random", 0, "-", "random"))["coverage_mean"]]
    axes[0][0].set_ylim(min(references[0], values.min()) - 0.04, min(max(values.max(), references[1]) + 0.04, references[1] + 0.45))
    _legend(fig, axes, 7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path)


def curves_metrics(r: Run, path: Path) -> Path:
    """Other metrics against K, per method, averaged over the four models (solid: equal places, dashed: proportional)."""
    plt = _plt()
    panels = [("coverage_p95", "pokrycie, 95. percentyl (mniej)"), ("rare1_categories", "rzadkie kategorie 1% z 6 (więcej)"),
              ("jsd_groups", "rozbieżność JS (mniej)"), ("low_tissue", "kafle mało użyteczne z 50 (mniej)"),
              ("near_duplicates", "pary prawie identyczne (mniej)"), ("diversity_mean", "różnorodność wybranych (więcej)")]
    grid = r.summary[(r.summary["test"] == "grid") & r.summary["model"].isin(r.models)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (column, title) in zip(axes.ravel(), panels):
        for method in qs.METHODS:
            for allocation, style in zip(qs.ALLOCATIONS, ("-", "--")):
                part = grid[(grid["method"] == method) & (grid["allocation"] == allocation)].groupby("k")[column].mean()
                ax.plot(part.index, part.values, linestyle=style, linewidth=1.8, markersize=4.5,
                        label=f"{METHOD_SHORT[method]}" if allocation == "equal" else None, **METHOD_STYLE[method])
        _reference_lines(r, ax, column)
        ax.set_title(title, fontsize=10.5), ax.set_xticks(r.ks), ax.grid(alpha=0.25), ax.set_xlabel("liczba klastrów K")
    _legend(fig, axes, 7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path)


def heatmaps(r: Run, path: Path) -> Path:
    plt = _plt()
    grid = r.summary[(r.summary["test"] == "grid") & r.summary["model"].isin(r.models) & r.summary["method"].isin(qs.METHODS)]
    low, high = grid["coverage_mean"].min(), grid["coverage_mean"].quantile(0.97)
    fig, axes = plt.subplots(2, 4, figsize=(16, 1.4 + 1.1 * len(r.models)), squeeze=False)
    for row, allocation in enumerate(qs.ALLOCATIONS):
        for ax, method in zip(axes[row], qs.METHODS):
            pivot = grid[(grid["method"] == method) & (grid["allocation"] == allocation)].pivot(index="model", columns="k", values="coverage_mean")
            values = pivot.reindex(index=r.models, columns=r.ks).to_numpy()
            ax.imshow(values, cmap="Blues_r", vmin=low, vmax=high, aspect="auto")  # darker = lower = better
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    ax.text(j, i, comma(values[i, j], 2), ha="center", va="center", fontsize=8.5,
                            color="white" if values[i, j] < low + 0.4 * (high - low) else "black")
            ax.set_xticks(range(len(r.ks)), r.ks), ax.set_yticks(range(len(r.models)), r.models if ax is axes[row][0] else [""] * len(r.models), fontsize=9)
            ax.set_title(f"{METHOD_SHORT[method]}, {ALLOCATION_NAMES[allocation]}", fontsize=10.5), ax.set_xlabel("K")
    fig.tight_layout()
    return _save(fig, path)


def factors_figure(r: Run, path: Path) -> Path:
    plt = _plt()
    names = {"method": "metoda w klastrze", "allocation": "przydział miejsc", "k": "liczba klastrów", "model": "model"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    for ax, (factor, title) in zip(axes, names.items()):
        part = r.factors[r.factors["factor"] == factor].reset_index(drop=True)
        labels = [METHOD_SHORT.get(v, ALLOCATION_NAMES.get(v, v)) for v in part["level"].astype(str)]
        y = np.arange(len(part))
        ax.hlines(y, part["coverage_best"], part["coverage_worst"], color="#9aa3ad", linewidth=3, label="od najlepszej do najgorszej konfiguracji")
        ax.scatter(part["coverage_mean"], y, s=70, color="#0072B2", zorder=3, label="średnia")
        ax.set_yticks(y, labels), ax.invert_yaxis(), ax.set_title(title, fontsize=11), ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("pokrycie (mniej = lepiej)")
    _legend(fig, axes, 2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def distributions(r: Run, sets: dict[str, tuple], path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.52 * len(sets)))
    rng = np.random.default_rng(0)
    grid = r.scores[r.scores["test"] == "grid"]
    for y, (name, config) in enumerate(sets.items()):
        part = grid
        for column, value in zip(qa.CONFIG, config):
            part = part[part[column] == value]
        values = part["coverage_mean"].to_numpy()
        ax.scatter(values, y + rng.uniform(-0.22, 0.22, len(values)), s=8, color="#0072B2" if name.startswith("zwyc") or name.startswith("2.") or name.startswith("3.") else "#777777",
                   alpha=0.45, linewidths=0)
        ax.vlines(values.mean(), y - 0.33, y + 0.33, color="black", linewidth=2)
    ax.set_yticks(range(len(sets)), list(sets), fontsize=9.5), ax.invert_yaxis(), ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("pokrycie pojedynczego wyboru (mniej = lepiej); kreska = średnia")
    fig.tight_layout()
    return _save(fig, path)


def confirmation_figure(r: Run, path: Path) -> Path:
    """Differences against the earlier recommendation and random selection, per test; to the right = the winner is better."""
    plt = _plt()
    data = r.confirmation[r.confirmation["reference"] != qa.REF_SAME]
    places = sorted(data["place"].unique())
    fig, axes = plt.subplots(1, len(places), figsize=(5.4 * len(places), 7.2), sharey=True, sharex=True, squeeze=False)
    first = data[(data["place"] == places[0]) & (data["reference"] == qa.REF_RECOMMENDED)]
    labels = [f"{t}: {p}" for t, p in zip(first["test"], first["part"])]
    for ax, place in zip(axes[0], places):
        for offset, (reference, colour) in enumerate(((qa.REF_RECOMMENDED, "#E69F00"), (qa.REF_RANDOM, "#444444"))):
            part = data[(data["place"] == place) & (data["reference"] == reference)]
            y = np.arange(len(part)) + (offset - 0.5) * 0.36
            ax.barh(y, part["difference"], height=0.34, color=colour, label=f"wobec: {reference}")
        config = data[data["place"] == place].iloc[0]
        ax.set_title(f"{int(place)}. {config['model']}\n{int(config['k'])} klastrów, {ALLOCATION_NAMES[config['allocation']]}, {METHOD_SHORT[config['method']]}", fontsize=10)
        ax.axvline(0, color="black", linewidth=1), ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("przewaga w pokryciu (> 0: wariant lepszy)")
    axes[0][0].set_yticks(range(len(labels)), labels, fontsize=9), axes[0][0].invert_yaxis()
    _legend(fig, axes, 2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, path)


def groups_figure(r: Run, group_names: dict[int, str], path: Path) -> Path:
    plt = _plt()
    sets = list(dict.fromkeys(r.groups["set"]))
    order = r.group_table.sort_values("tiles", ascending=False)["group"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    height = 0.8 / len(sets)
    for ax, (column, title) in zip(axes, (("coverage", "pokrycie kafli z tej grupy (mniej = lepiej)"), ("chosen", "ile z 50 kafli wybrano z grupy"))):
        for i, name in enumerate(sets):
            part = r.groups[r.groups["set"] == name].set_index("group").loc[order]
            ax.barh(np.arange(len(order)) - 0.4 + height * (i + 0.5), part[column], height=height * 0.92, color=SET_COLOURS[i % 4], label=short_set(name))
        if column == "chosen":
            share = r.group_table.set_index("group").loc[order, "share"] * r.total
            ax.scatter(share, np.arange(len(order)), marker="|", s=500, color="#D55E00", linewidths=2.5, label="proporcjonalnie do slajdu", zorder=3)
        ax.set_title(title, fontsize=11), ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(len(order)), [group_names[g] for g in order], fontsize=9.5), axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, path)


def features_figure(r: Run, path: Path) -> Path:
    plt = _plt()
    sets = list(dict.fromkeys(r.features["set"]))
    names = list(dict.fromkeys(r.features["feature"]))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(sets):
        part = r.features[r.features["set"] == name].set_index("feature").loc[names]
        ax.scatter(part["difference"], np.arange(len(names)) + (i - (len(sets) - 1) / 2) * 0.14, s=44, color=SET_COLOURS[i % 4],
                   marker="osD^"[i % 4], label=short_set(name))
    ax.set_yticks(range(len(names)), [FEATURE_NAMES[f] for f in names], fontsize=9.5), ax.invert_yaxis(), ax.grid(alpha=0.25)
    ax.set_xlabel("średnia różnica cechy między kaflem a najbliższym wybranym (mniej = lepiej, w odchyleniach standardowych)")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def clustering_figure(r: Run, path: Path) -> Path:
    plt = _plt()
    metrics = r.metrics.groupby(["model", "k"])[["silhouette", "davies_bouldin", "calinski_harabasz"]].mean().reset_index()
    ks = [*r.ks, r.total]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (column, title) in zip(axes, (("silhouette", "silhouette kosinusowa (więcej)"), ("davies_bouldin", "Davies-Bouldin (mniej; w obrębie modelu)"),
                                          ("calinski_harabasz", "Calinski-Harabasz (więcej; w obrębie modelu)"))):
        for model in [*r.models, "noise-control"]:
            part = metrics[metrics["model"] == model].set_index("k").reindex(ks)
            style = MODEL_STYLE.get(model, {"color": "#888888", "marker": "x"})
            ax.plot(range(len(ks)), part[column], label=model_name(model), linewidth=2, **style)
        ax.set_xticks(range(len(ks)), ks), ax.set_title(title, fontsize=11), ax.grid(alpha=0.25), ax.set_xlabel("liczba klastrów (ostatni punkt: dotychczasowe 50)")
    _legend(fig, axes, 5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def name_groups(r: Run) -> dict[int, str]:
    """A short name of every tissue group from its two most unusual descriptors, with its share of the slide."""
    names = {}
    z = pd.DataFrame(r.D, columns=list(r.table.columns[r.table.columns != "tile_id"])).assign(group=r.manifest["group"]).groupby("group").mean()
    for row in r.group_table.itertuples():
        values = z.loc[row.group]
        strongest = values.abs().sort_values(ascending=False).head(2).index
        words = ", ".join(f"{'dużo' if values[f] > 0 else 'mało'}: {FEATURE_NAMES[f]}" for f in strongest)
        names[int(row.group)] = f"G{row.group + 1} ({percent(row.share)}) {words}"
    return names


def group_strip(r: Run, image_of: Callable[[str], np.ndarray], group_names: dict[int, str], *, count: int = 8, side: int = 220) -> np.ndarray:
    """Example tiles of every tissue group: the ones nearest the group's mean, not overlapping each other."""
    positions = r.manifest[["x", "y"]].to_numpy(float)
    size = int(r.manifest["width"].iloc[0])
    order = r.group_table.sort_values("tiles", ascending=False)["group"].tolist()
    left, gap = 520, 8
    canvas = Canvas(left + count * (side + gap), 20 + len(order) * (side + gap))
    for row, g in enumerate(order):
        members = np.flatnonzero(r.manifest["group"].to_numpy() == g)
        distance = ((r.D[members] - r.D[members].mean(axis=0)) ** 2).sum(axis=1)
        chosen: list[int] = []
        for i in members[np.argsort(distance, kind="stable")]:
            if all(np.abs(positions[i] - positions[j]).max() >= size for j in chosen):
                chosen.append(int(i))
            if len(chosen) == count:
                break
        y = 20 + row * (side + gap)
        head, _, tail = group_names[g].partition(") ")
        canvas.text(8, y + side // 2 - 30, head + ")", 28, bold=True)
        for line, text in enumerate(tail.split(", ")):
            canvas.text(8, y + side // 2 + 8 + 28 * line, text, 20)
        for column, i in enumerate(chosen):
            canvas.tile(image_of(r.manifest["tile_id"].iloc[i]), left + column * (side + gap), y, side)
    return canvas.array()


def worst_covered(r: Run, image_of: Callable[[str], np.ndarray], sets: list[str], *, count: int = 8, side: int = 220) -> np.ndarray:
    """Per set (its median selection): the tiles farthest from every chosen one, i.e. what an annotator would never see."""
    positions = r.manifest[["x", "y"]].to_numpy(float)
    size = int(r.manifest["width"].iloc[0])
    left, gap = 430, 8
    canvas = Canvas(left + count * (side + gap), 20 + len(sets) * (side + 40 + gap))
    for row, name in enumerate(sets):
        tiles = r.chosen(name)["tiles"]
        squared = (r.D**2).sum(axis=1)[:, None] + (r.D[tiles] ** 2).sum(axis=1)[None, :] - 2.0 * r.D @ r.D[tiles].T
        distance = np.sqrt(np.clip(squared.min(axis=1), 0, None))
        far: list[int] = []
        for i in np.argsort(-distance, kind="stable"):
            if all(np.abs(positions[i] - positions[j]).max() >= size for j in far):
                far.append(int(i))
            if len(far) == count:
                break
        y = 20 + row * (side + 40 + gap)
        for line, text in enumerate(short_set(name).split(", ")):
            canvas.text(8, y + side // 2 - 20 + 32 * line, text, 25, bold=True)
        for column, i in enumerate(far):
            x = left + column * (side + gap)
            canvas.tile(image_of(r.manifest["tile_id"].iloc[i]), x, y, side)
            canvas.text(x, y + side + 6, f"odległość {comma(distance[i], 1)}", 19)
    return canvas.array()


# ---- tables -----------------------------------------------------------------------------------


def result_row(r: Run, label: str, config: tuple) -> list:
    s = r.grid(config)
    spread = f" ± {num(s['coverage_mean_sd'], 3)}" if s["selections"] > 1 else ""
    return [label, num(s["coverage_mean"], 3) + spread, num(s["coverage_p95"], 2), num(s["rare5_categories"], 1), num(s["rare1_categories"], 1),
            num(s["jsd_groups"], 3), num(s["near_duplicates"], 1), num(s["low_tissue"], 1), num(s["nuclei"], 0), num(s["grid_cells"], 1)]


RESULT_HEADER = ["wariant", "pokrycie ± rozrzut", "95. percentyl", "rzadkie 5% (z 6)", "rzadkie 1% (z 6)", "rozbieżność JS", "pary prawie identyczne",
                 "mało użyteczne (z 50)", "jądra", "pola slajdu"]


def _diagram() -> str:
    steps = ["Preparat WSI (trepanobiopsja szpiku), H&amp;E", "2668 kafli 1024 × 1024 px po kontroli jakości", "Model SSL zamienia kafel w wektor (embedding)",
             "KMeans dzieli kafle na K klastrów", "Z klastrów wybieramy 50 kafli", "Ocena zestawu na 13 cechach liczonych z pikseli"]
    boxes = "<span class='arrow'>→</span>".join(f"<span class='step'>{s}</span>" for s in steps)
    return f"<div class='flow'>{boxes}</div>"


EXTRA_STYLE = """
.flow { display:flex; align-items:stretch; gap:6px; margin:12px 0; break-inside:avoid; }
.step { flex:1; border:1px solid var(--line); background:#f5f6f8; border-radius:8px; padding:8px; font-size:12px; text-align:center; }
.arrow { align-self:center; color:var(--mute); }
.read { background:#f5f6f8; border-radius:6px; padding:6px 12px; margin:6px 0; font-size:12.5px; break-inside:avoid; break-after:avoid; }
.note { break-inside:avoid; }
@media print { figure.short img { max-height: 126mm; } }
"""


def build(results: str | Path, image_of: Callable[[str], np.ndarray], out: str | Path, *, notes: str | Path | None = None, pdf: bool = True) -> Path:
    r = Run(results)
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    text = read_notes(notes)
    note = lambda key: f"<div class='note'><b>Wniosek.</b> {text[key]}</div>" if key in text else ""
    how = lambda shows, reading: f"<div class='read'><b>Co pokazuje:</b> {shows} <b>Jak czytać:</b> {reading}</div>"
    cache: dict[str, np.ndarray] = {}

    def small(tile_id: str) -> np.ndarray:
        if tile_id not in cache:
            cache[tile_id] = np.asarray(Image.fromarray(image_of(tile_id)).resize((256, 256), Image.LANCZOS))
        return cache[tile_id]

    winner = r.winner
    winner_name = config_name(*winner)
    recommended, confirmed = r.confirmed_config()
    baselines = r.baselines()
    ranked, limit = qa.apply_rule(r.summary, 10)
    group_names = name_groups(r)
    n_tiles = len(r.manifest)
    args = r.run["args"]
    mpp = float(r.manifest["mpp"].iloc[0])
    size = int(r.manifest["width"].iloc[0])
    regions = r.manifest["region"].value_counts().sort_index()
    eligible = int(ranked.iloc[0]["eligible"])
    everything = int(ranked.iloc[0]["all"])

    key_rows = [result_row(r, f"zwycięzca siatki: {winner_name}", winner)] + [result_row(r, short_set(n), c) for n, c in baselines.items()]
    top_rows = [result_row(r, f"{int(w['place'])}. {config_name(w['model'], w['k'], w['allocation'], w['method'])}",
                           tuple(w[c] for c in qa.CONFIG)) for w in ranked.to_dict("records")]
    base_rows = [result_row(r, short_set(n), c) for n, c in baselines.items()]

    sets = {}
    for w in r.winners.to_dict("records"):
        prefix = "zwycięzca: " if w["place"] == 1 else f"{w['place']}. "
        sets[prefix + config_name(w["model"], w["k"], w["allocation"], w["method"])] = tuple(w[c] for c in qa.CONFIG)
    sets.update({short_set(n): c for n, c in baselines.items()})

    stability = r.stability.set_index(qa.CONFIG)
    stability_rows = []
    for name, config in sets.items():
        if config in stability.index:
            s = stability.loc[config]
            stability_rows.append([name, percent(s["identical"]), percent(s["identical_or_overlapping"])])

    confirm_tables = ""
    for w in r.winners.to_dict("records"):
        part = r.confirmation[r.confirmation["place"] == w["place"]]
        first = part[part["reference"] == qa.REF_RECOMMENDED]
        rows = []
        for line in first.itertuples():
            row = [line.test, line.part, num(line.coverage, 3)]
            for reference in (qa.REF_RECOMMENDED, qa.REF_SAME, qa.REF_RANDOM):
                m = part[(part["reference"] == reference) & (part["test"] == line.test) & (part["part"] == line.part)].iloc[0]
                row.append(f"{num(m['reference_coverage'], 3)}  {'✓' if m['passed'] else '✗'}")
            rows.append(row)
        verdict = r.verdicts[r.verdicts["place"] == w["place"]].iloc[0]
        result = "potwierdzony" if verdict["confirmed"] else "niepotwierdzony"
        confirm_tables += (f"<div class='keep'><h3>{int(w['place'])}. {config_name(w['model'], w['k'], w['allocation'], w['method'])}: {result}</h3>"
                           + table(["test", "część", "pokrycie wariantu", qa.REF_RECOMMENDED, qa.REF_SAME, "losowa"], rows) + "</div>")

    factor_rows = [[{"method": "metoda", "allocation": "przydział", "k": "K", "model": "model"}[f.factor],
                    METHOD_NAMES.get(str(f.level), ALLOCATION_NAMES.get(str(f.level), str(f.level))), num(f.coverage_mean, 3), num(f.coverage_best, 3),
                    num(f.coverage_worst, 3), num(f.rare1_categories, 1), num(f.jsd_groups, 3), num(f.low_tissue, 1)] for f in r.factors.itertuples()]
    one_rows = [result_row(r, config_name(*qa.one_per_cluster(m, r.total)), qa.one_per_cluster(m, r.total))
                for m in [*r.models, "noise-control"]]
    why_title = ("Dlaczego ten wariant? Z czego bierze się różnica" if confirmed else
                 "Dlaczego dotychczasowa metoda zostaje: z czego biorą się różnice")
    group_rows = []
    for g in r.group_table.sort_values("tiles", ascending=False).itertuples():
        group_rows.append([group_names[g.group], int(g.tiles), percent(g.white), percent(g.compact_eosin), percent(g.erythrocyte),
                           num(g.nuclear_density / 1000, 1)])
    chosen_sets = list(dict.fromkeys(r.groups["set"]))
    k_quota = r.ks[len(r.ks) // 2]
    quota_rows = [[f"K = {k}", ", ".join(map(str, qs.quotas([1000 - i for i in range(k)], r.total, "equal")))] for k in r.ks]

    f = {
        "illustration": method_illustration(figures / "f1_metody.png"),
        "coverage": curves_coverage(r, figures / "w1_pokrycie.png"),
        "metrics": curves_metrics(r, figures / "w2_miary.png"),
        "heat": heatmaps(r, figures / "w3_mapy.png"),
        "factors": factors_figure(r, figures / "w4_czynniki.png"),
        "distributions": distributions(r, sets, figures / "w5_rozklady.png"),
        "clustering": clustering_figure(r, figures / "w6_klastrowanie.png"),
        "confirm": confirmation_figure(r, figures / "w7_potwierdzenia.png"),
        "groups": groups_figure(r, group_names, figures / "w8_grupy.png"),
        "features": features_figure(r, figures / "w9_cechy.png"),
    }
    figure = lambda key, caption: picture(f[key], caption, lossless=True, css="short")
    strip = group_strip(r, small, group_names)
    worst = worst_covered(r, small, [s for s in ("zwycięzca siatki", qa.REF_RECOMMENDED, qa.REF_RANDOM) if s in set(r.representative["set"])])
    Image.fromarray(strip).save(figures / "f2_grupy_przyklady.png")
    Image.fromarray(worst).save(figures / "w10_najgorzej_pokryte.png")

    w_row = r.grid(winner)
    ref_row = r.grid(qa.one_per_cluster(qa.RECOMMENDED_MODEL, r.total))
    random_row = r.grid(("random", 0, "-", "random"))
    def versus(other: pd.Series) -> str:
        change = (other["coverage_mean"] - w_row["coverage_mean"]) / other["coverage_mean"]
        return f"zwycięzca siatki o {num(100 * abs(change), 1)}% {'lepszy' if change > 0 else 'gorszy'}"
    status = ("<b>potwierdzony</b> we wszystkich testach wobec dotychczasowej rekomendacji i losowania" if confirmed and recommended == winner else
              "<b>niepotwierdzony</b>: nie przeszedł wszystkich testów (sekcja 8)")
    recommendation = (f"{config_name(*recommended)}" if confirmed else
                      f"{config_name(*recommended)} (dotychczasowa metoda): żaden z nowych wariantów nie przeszedł wszystkich testów")

    body = f"""
<h1>Jak wybrać 50 kafli do adnotacji? Kilka kafli z 5–10 klastrów</h1>
<p class="mute">Jeden preparat (slajd 24070_25, jeden pacjent), {n_tiles} kafli. Raport 1 z 2: pytanie, metoda i wyniki. Obrazy preparatu pacjenta:
do użytku wewnętrznego. Raport zakłada, że czytelnik nie zna projektu.</p>

<h2>1. Streszczenie</h2>
<div class="box">{text.get('streszczenie', '')}
<ul><li>Zwycięzca siatki: <b>{winner_name}</b>, {status}.</li>
<li>Pokrycie (mniej = lepiej) zwycięzcy siatki: {num(w_row['coverage_mean'], 3)}. Dotychczasowa metoda: {num(ref_row['coverage_mean'], 3)}
({versus(ref_row)}). Losowa: {num(random_row['coverage_mean'], 3)} ({versus(random_row)}).</li>
<li>Rekomendacja według reguł protokołu: <b>{recommendation}</b>.</li></ul></div>
<div class="keep"><h3>T1. Najważniejsze liczby (średnie; strzałki kierunków w słowniku miar)</h3>{table(RESULT_HEADER, key_rows, mark_rows=[0])}</div>

<h2>2. Po co to robimy</h2>
<p>Preparat histologiczny po zeskanowaniu (WSI) to obraz wielkości miliardów pikseli, dlatego dzieli się go na małe kwadraty, <b>kafle</b>. Żeby nauczyć model
rozpoznawać komórki, specjalista musi ręcznie oznaczyć wybrane kafle. To kosztowne, więc w <b>Active Learning</b> oznacza się je porcjami (rundami), a o
tym, które kafle trafią do porcji, decyduje algorytm. W tym eksperymencie porcja ma zawsze <b>{r.total} kafli</b>.</p>
<p>Dobry zestaw jest <b>różnorodny</b> (obejmuje różne obrazy, także rzadkie), <b>reprezentatywny</b> (nie przekręca proporcji preparatu), bez powtórek i bez
prawie pustych obrazów. Nie wiemy z góry, które kafle są „właściwe”, więc porównujemy sposoby wyboru według mierzalnych przybliżeń tych własności.</p>
<p><b>Pytania:</b> (1) czy lepiej podzielić kafle na mało klastrów (5–10) i brać z każdego po kilka, niż jak dotąd na {r.total} klastrów i brać po jednym;
(2) ile klastrów, jak rozdzielić miejsca, którą metodą wybierać wewnątrz klastra i którym modelem; (3) dlaczego.</p>
{_diagram()}

<h2>3. Dane i ich przygotowanie</h2>
{table(["cecha", "wartość"], [
    ["kafle", f"{n_tiles} obrazów {size} × {size} px, {num(mpp, 4)} µm/px (pole {num(size * mpp, 0)} × {num(size * mpp, 0)} µm)"],
    ["kontrola jakości (etap wcześniejszy)", "kafel przyjęty, gdy tkanka zajmuje co najmniej 50% i artefakty najwyżej 2% powierzchni"],
    ["nakładanie", f"krok 512 px, więc sąsiednie kafle dzielą połowę powierzchni; {int(r.manifest['lattice'].sum())} kafli tworzy siatkę bez nakładania"],
    ["fragmenty tkanki", f"{len(regions)}; dwa największe mają {int(regions.iloc[0])} i {int(regions.iloc[1]) if len(regions) > 1 else 0} kafli"],
])}
<h3>Od obrazu do wektora (embeddingu)</h3>
<ol>
<li>Modele przyjmują obraz 224 × 224 px. Kafel 1024 px nie jest zmniejszany w całości, tylko <b>czytany jako 9 okien</b> 507 × 507 px (112 µm), a każde okno
przeskalowane do 224 px przy 0,5 µm/px. Wszystkie modele dostają dokładnie te same wycinki.</li>
<li>Kolory są normalizowane średnimi i odchyleniami właściwymi dla danego modelu.</li>
<li>Embedding kafla to <b>średnia 9 wektorów</b>. Okna nakładają się mniej więcej w połowie, więc środek kafla liczy się w średniej częściej niż rogi.</li>
<li>Wektor jest normalizowany do długości 1 (L2). Dzięki temu odległość w KMeans jest bliska odległości kosinusowej (kąt między wektorami). Nie stosujemy
PCA ani centrowania w wariancie głównym.</li>
</ol>
<h3>Cechy z pikseli: niezależny „sędzia”</h3>
<p>Gdyby oceniać zestaw w przestrzeni modelu, każdy model oceniałby sam siebie. Dlatego dla każdego kafla policzono 13 prostych cech bezpośrednio z pikseli
(w połowie rozdzielczości; biel szkła wyznaczona dla całego slajdu). Cechy przeskalowano do wspólnej skali (odchylenia od średniej slajdu, przycięte do ±5).
To przybliżenia: nie rozpoznają atypii ani typów komórek.</p>
<div class="keep">{table(["cecha", "co mierzy"], [[FEATURE_NAMES[k], v] for k, v in FEATURE_MEANING.items()])}</div>
<h3>Grupy kafli</h3>
<p>Żeby móc mówić o „typach” obrazów, kafle podzielono (KMeans na 13 cechach, bez udziału modeli) na 8 grup. Nazwa grupy to dwie cechy, którymi najbardziej
odbiega od średniej; to nie są rozpoznania histologiczne.</p>
{picture(strip, "F2. Przykłady kafli z 8 grup (najbliższe środka grupy). W nawiasie udział grupy w slajdzie.", width=2200)}
<div class="keep">{table(["grupa", "kafli", "biel", "kość", "krwinki", "jądra, tys./mm²"], group_rows)}</div>

<h2 class="page">4. Modele</h2>
{table(["model", "architektura", "wymiar wektora", "na czym uczony"], [list(m) for m in MODEL_FACTS if m[0] in r.models])}
<p>Wszystkie są <b>zamrożone</b>: nie douczaliśmy ich na naszych danych. Uczono je samonadzorowanie (bez etykiet). Dodatkowo
<b>kontrola szumem</b> podstawia losowe wektory zamiast embeddingów: wynik modelu lepszy od niej oznacza, że wektory niosą informację o obrazie.</p>

<h2>5. Jak wybieramy 50 kafli</h2>
<ol>
<li><b>Klastrowanie.</b> KMeans dzieli embeddingi na K = {", ".join(map(str, r.ks))} klastrów; powtarzamy dla {r.seeds} różnych startów losowych (seedów).</li>
<li><b>Przydział miejsc.</b> <i>Równy:</i> każdy klaster dostaje {r.total} // K kafli, reszta trafia do największych. <i>Proporcjonalny:</i> liczba kafli
proporcjonalna do wielkości klastra (metoda największych reszt), co najmniej 1.</li>
<li><b>Wybór w klastrze</b> jedną z czterech metod (rysunek F1).</li>
<li><b>Zakaz nakładania.</b> Żadne dwa wybrane kafle nie dzielą powierzchni. Klastry biorą po jednym kaflu na zmianę; jeśli któremuś zabraknie wolnych kafli,
jego miejsca przejmują największe klastry.</li>
</ol>
<div class="keep">{table(["liczba klastrów", "przydział równy (od największego klastra)"], quota_rows)}</div>
{figure("illustration", "F1. Cztery metody na wymyślonych punktach 2-D (ilustracja, nie dane): + to środek klastra, liczby to kolejność wyboru.")}
<div class="keep">{table(["metoda", "zasada", "czego się spodziewać"], [
    [METHOD_NAMES["m1"], "najpierw kafel najbliższy środkowi, potem zawsze najbardziej różny od już wziętych z tego klastra", "różnorodność, ale też skrajności i artefakty"],
    [METHOD_NAMES["m1p"], "jak M1, po odrzuceniu 5% kafli najdalszych od środka", "różnorodność bez najbardziej nietypowych"],
    [METHOD_NAMES["m2"], f"losowo wewnątrz klastra ({args['draws']} losowań na seed)", "zachowane proporcje wewnątrz klastra"],
    [METHOD_NAMES["m3"], "kolejne kafle najbliższe środkowi", "kafle typowe i podobne do siebie"]])}</div>
<h3>Z czym porównujemy</h3>
{table(["odniesienie", "opis"], [
    ["dotychczasowa metoda", f"{r.total} klastrów, z każdego kafel najbliższy środkowi; wcześniej rekomendowana z modelem lunit-dino-s16"],
    ["bez klastrowania", "metoda M1 na całym slajdzie naraz: sprawdza, czy klastry w ogóle pomagają"],
    ["losowa", f"{args['random_draws']} losowych zestawów {r.total} kafli bez nakładania"],
    ["przestrzenna", "kafle rozłożone równomiernie po slajdzie, bez żadnego modelu"],
    ["pułap", "KMeans wprost na cechach, którymi oceniamy: pokazuje, jak daleko można dojść na tej skali; to nie jest konkurent"]])}
<p>Siatka to {len(r.models)} modele × {len(r.ks)} wartości K × 2 przydziały × 4 metody = <b>{everything} konfiguracji</b>, każda w {r.seeds} seedach.</p>

<h2 class="page">6. Słownik miar i reguła wyboru</h2>
<div class="keep">{table(["miara", "lepiej", "co nam mówi", "jak jest liczona"], [[m[1], m[2], m[3], m[4]] for m in METRICS])}</div>
<p><b>± rozrzut</b> to odchylenie standardowe pomiędzy pojedynczymi wyborami (seedami, a w M2 także losowaniami). Dwie metody, które różnią się mniej niż
o ten rozrzut, uznajemy za <b>remis</b>.</p>
<div class="box"><b>Reguła zapisana przed obliczeniami:</b> (1) odpadają konfiguracje, które średnio biorą więcej kafli mało użytecznych niż losowanie
(próg: {num(limit, 1)} z {r.total}; spełnia go {eligible} z {everything}); (2) wygrywa najniższe pokrycie; (3) w remisie decydują kolejno: więcej rzadkich
kategorii 1%, mniejsza rozbieżność JS, tańszy model. Trzy najlepsze przechodzą testy potwierdzające, bo przy tylu konfiguracjach ktoś może wygrać
przypadkiem.</div>

<h2 class="page">7. Wyniki siatki</h2>
<div class="keep"><h3>T2. Dziesięć najlepszych konfiguracji według reguły</h3>{table(RESULT_HEADER, top_rows, mark_rows=[0])}</div>
<div class="keep"><h3>T3. Odniesienia, te same miary</h3>{table(RESULT_HEADER, base_rows)}</div>
<p class="mute">W regule remisowało {int(r.winners.iloc[0]['tied'])} konfiguracji o pokryciu do {num(r.winners.iloc[0]['best_coverage_in_tie'], 3)};
zwycięzcę wyłoniły w tej grupie kryteria dodatkowe.</p>
{note('siatka')}
{how("pokrycie wobec liczby klastrów, osobno dla każdej metody i przydziału.", "linie to modele, pasma to rozrzut; poziome linie to odniesienia z 50 kaflami. Im niżej, tym lepiej.")}
{figure("coverage", "W1. Pokrycie wobec liczby klastrów, zawsze 50 kafli.")}
{note('krzywe')}
{how("pozostałe miary wobec K, uśrednione po czterech modelach.", "kolor to metoda, linia ciągła: przydział równy, przerywana: proporcjonalny. Kierunek lepszego wyniku jest w tytule panelu.")}
{figure("metrics", "W2. Pozostałe miary jakości wobec liczby klastrów, zawsze 50 kafli.")}
{note('miary')}
{how("pokrycie każdej konfiguracji.", "ciemniejsze pole to niższe, czyli lepsze pokrycie. Wspólna skala kolorów dla wszystkich paneli.")}
{figure("heat", "W3. Model × liczba klastrów dla każdej metody i przydziału.")}
{note('mapy')}
<h3>Który wybór ma największe znaczenie?</h3>
{how("średnie pokrycie wszystkich konfiguracji, które mają wspólną jedną cechę.", "im bardziej punkty w panelu są od siebie oddalone, tym ważniejsza decyzja; szara kreska pokazuje cały zakres.")}
{figure("factors", "W4. Wpływ metody, przydziału, liczby klastrów i modelu.")}
<div class="keep">{table(["czynnik", "poziom", "pokrycie średnie", "najlepsze", "najgorsze", "rzadkie 1%", "rozbieżność JS", "mało użyteczne"], factor_rows)}</div>
{note('czynniki')}
{how("rozrzut pojedynczych wyborów trzech najlepszych wariantów i odniesień.", "każdy punkt to jeden zestaw 50 kafli; nakładające się chmury oznaczają, że pojedynczy wybór jednej metody bywa gorszy od drugiej.")}
{figure("distributions", "W5. Rozkład pokrycia po seedach i losowaniach.")}
<div class="keep"><h3>T4. Czy inny seed daje te same kafle?</h3>{table(["wariant", "identyczne kafle", "identyczne lub nakładające się"], stability_rows)}
<p class="mute">Średnio po parach seedów: jaka część pierwszego zestawu powtarza się w drugim.</p></div>
{note('stabilnosc')}
{how("klasyczne miary jakości samych klastrów (bez wyboru kafli).", "silhouette blisko 0 oznacza klastry, które się zlewają; Davies-Bouldin i Calinski-Harabasz porównujemy tylko wzdłuż jednej linii.")}
{figure("clustering", "W6. Miary klastrowania.")}
{note('klastrowanie')}

<h2 class="page">8. Czy zwycięzca wygrywa naprawdę? Testy potwierdzające</h2>
{table(["test", "po co"], [
    ["nowe seedy (30–59)", "czy przewaga nie wynikała z konkretnych startów losowych (wymagana przewaga większa niż rozrzut)"],
    ["fragmenty tkanki", "cała procedura osobno na dwóch kawałkach tkanki: czy działa na innym obszarze"],
    ["usuwanie cech", "ocena bez jednej z 6 grup cech: czy wynik nie zależy od jednej cechy „sędziego”"],
    ["podpróbki", "20 losowych 80% kafli: czy wynik nie zależy od kilku kafli"],
    ["warianty przestrzeni", "PCA 128, 10 startów KMeans, tylko kafle bez nakładania: czy nie zależy od ustawień"]])}
{how("przewagę wariantu nad dotychczasową rekomendacją i losowaniem w każdym teście.", "słupek w prawo: wariant lepszy; w lewo: gorszy.")}
{figure("confirm", "W7. Testy potwierdzające dla trzech najlepszych.")}
<p>W tabelach: pokrycie odniesienia i ✓, jeśli wariant zalicza test wobec niego. Wariant jest potwierdzony, gdy ma ✓ wszędzie w kolumnach
„dotychczasowa” i „losowa”.</p>
{confirm_tables}
{note('potwierdzenie')}

<h2 class="page">9. {why_title}</h2>
{text.get('dlaczego', '')}
<div class="keep"><h3>T5. Który model? Po 1 kaflu z klastra, każdy model i kontrola szumem</h3>{table(RESULT_HEADER, one_rows)}</div>
{note('modele')}
{how("jak dobrze każdy zestaw pokrywa poszczególne grupy kafli i ile miejsc im przydziela.", "lewy panel: krótszy słupek to lepsze pokrycie grupy; prawy: pomarańczowa kreska to liczba miejsc proporcjonalna do udziału grupy w slajdzie.")}
{figure("groups", "W8. Pokrycie i liczba wybranych kafli w grupach tkanki (średnio po seedach).")}
{note('grupy')}
{how("które z 13 cech są w zestawie najlepiej odwzorowane.", "punkt bardziej w lewo: dla typowego kafla istnieje wybrany o bliższej wartości tej cechy.")}
{figure("features", "W9. Pokrycie rozbite na cechy.")}
{note('cechy')}
{how("kafle, które są najmniej podobne do czegokolwiek w zestawie (seed o medianowym pokryciu).", "to obrazy, których adnotujący przy tym wyborze nie zobaczy w podobnej postaci.")}
{picture(worst, "W10. Najgorzej pokryte kafle trzech zestawów.", width=2200)}
{note('przyklady')}

<h2 class="page">10. Wnioski i rekomendacja</h2>
<div class="box">{text.get('wnioski', f'<p>Rekomendacja według reguł: {recommendation}.</p>')}</div>
<h3>Czego te wyniki nie dowodzą</h3>
<ul>
<li>Badaliśmy <b>jeden preparat jednego pacjenta</b>; fragmenty i podpróbki pochodzą z tego samego preparatu.</li>
<li>Oceniają 13 cech z pikseli, nie patomorfolog. Wariant lepszy na tych cechach nie musi dawać lepszego zbioru do adnotacji.</li>
<li>Nie sprawdziliśmy, czy po adnotacji model uczy się lepiej; to pokaże dopiero eksperyment Active Learning.</li>
<li>Analiza jest eksploracyjna: wcześniej znaliśmy na tym slajdzie wyniki wyboru po jednym kaflu z klastra.</li>
</ul>
{text.get('ograniczenia', '')}
<h2>11. Słowniczek i odtworzenie</h2>
{table(["pojęcie", "znaczenie"], [
    ["WSI", "zeskanowany cały preparat"], ["kafel", "kwadratowy wycinek preparatu"], ["embedding", "wektor liczb opisujący obraz, wynik modelu"],
    ["SSL", "uczenie samonadzorowane: bez ręcznych etykiet"], ["KMeans", "algorytm dzielący punkty na K grup wokół środków"],
    ["centroid", "środek klastra"], ["seed", "ustawienie generatora losowego; inny seed to inny start algorytmu"],
    ["odległość kosinusowa", "miara kąta między wektorami: 0 dla identycznego kierunku"], ["pułap", "wynik metody, która zna cechy oceny; punkt skali"]])}
<p>Protokół: <code>docs/architecture/quota-selection-experiment.md</code>. Kod: <code>scripts/run_quota_selection.py</code> (obliczenia),
<code>scripts/build_quota_reports.py</code> (raporty). Wszystkie liczby: pliki CSV w katalogu <code>analysis/</code>
(m.in. <code>summary.csv</code> ze wszystkimi konfiguracjami).</p>
"""
    target = out / "raport_eksperyment_i_wyniki.html"
    target.write_text(page("50 kafli z klastrów: eksperyment i wyniki", body, extra_style=EXTRA_STYLE), encoding="utf-8")
    if pdf:
        to_pdf(target, out / "raport_eksperyment_i_wyniki.pdf", timeout=600)
    return target
