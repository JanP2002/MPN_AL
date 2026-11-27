"""Report 1 of the K selection experiment (Polish): tables and charts from the files ``k_cli`` writes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster_figures import _plt, _save
from .report_html import num, page, percent, picture, read_notes, table, to_pdf

MODELS = ["lunit-dino-s16", "midnight-12k", "phikon-v2", "dinov2-s14"]
NOISE = "noise-control"
BASELINES = ["random", "spatial", "descriptor-kmeans"]
NAMES = {"random": "losowa", "spatial": "przestrzenna", "descriptor-kmeans": "pułap (KMeans na cechach)", NOISE: "kontrola szumem"}
# One colour per method in every chart; baselines also differ by line style, so colour is never the only cue.
STYLES = {"dinov2-s14": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
          "lunit-dino-s16": {"color": "#E69F00", "marker": "s", "linestyle": "-"},
          "phikon-v2": {"color": "#009E73", "marker": "^", "linestyle": "-"},
          "midnight-12k": {"color": "#D55E00", "marker": "D", "linestyle": "-"},
          NOISE: {"color": "#888888", "marker": "x", "linestyle": "--"},
          "random": {"color": "#000000", "marker": "", "linestyle": ":"},
          "spatial": {"color": "#CC79A7", "marker": "", "linestyle": "-."},
          "descriptor-kmeans": {"color": "#56B4E9", "marker": "", "linestyle": "--"}}


def variant_names(run: dict, lattice_tiles: int) -> dict[str, str]:
    """Variant key -> Polish name, in the order of the tables, for the settings this run used."""
    args = run["args"]
    pca, extra = args["pca"], args["n_init"][1:]
    spaces = {"lattice": f"{lattice_tiles} kafli bez nakładania"}
    if pca:
        spaces[f"pca{pca}"] = f"PCA {pca}"
    names = {"l2": "główny", "free": "nakładanie dozwolone", **spaces}
    for n in extra:
        names[f"l2-ninit{n}"] = f"n_init = {n}"
        names.update({f"{key}-ninit{n}": f"{value}, n_init = {n}" for key, value in spaces.items()})
    if args["subsamples"]:
        share = round(100 * args["fraction"])
        names[f"subsample{share}"] = f"podpróbki {share}% kafli"
    return names


def label(method: str) -> str:
    return NAMES.get(method, method)


class Results:
    """The tables of one run of the grid search."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        read = lambda name: pd.read_csv(self.folder / f"{name}.csv")
        self.summary = read("k_summary")
        self.scores = read("scores_gap")
        self.choice = read("k_choice")
        self.order = read("model_order")
        self.marginal = read("marginal_gains")
        self.nesting = read("nesting")
        self.stability = read("selection_stability")
        self.ari = read("seed_stability")
        self.metrics = read("clustering_metrics")
        self.seeds = read("representative_seeds")
        self.run = json.loads((self.folder / "run.json").read_text(encoding="utf-8"))
        self.ks = [int(k) for k in self.run["ks"]]
        self.models = [m for m in MODELS if m in self.run["models"]] + [m for m in self.run["models"] if m not in MODELS]
        self.lattice_tiles = int(pd.read_parquet(self.folder / "manifest.parquet", columns=["lattice"])["lattice"].sum())
        self.names = variant_names(self.run, self.lattice_tiles)
        self.variants = [v for v in self.names if v in set(self.summary["variant"])]

    def rows(self, method: str, variant: str = "l2") -> pd.DataFrame:
        variant = "baseline" if method in BASELINES else variant
        return self.summary[(self.summary["method"] == method) & (self.summary["variant"] == variant)].set_index("k").reindex(self.ks)

    def top_models(self, count: int = 2) -> list[str]:
        return self.order[self.order["variant"] == "l2"].sort_values("rank")["method"].tolist()[:count]


def metric_table(r: Results, methods: list[str], column: str, digits: int, *, sd: bool = False, best: str | None = None,
                 as_percent: bool = False) -> str:
    """Model x K table of one summary column; ``best`` (``min``/``max``) highlights the best model in every K."""
    values = {m: r.rows(m) for m in methods}
    rows = []
    for method in methods:
        cells = [label(method)]
        for k in r.ks:
            value = values[method].loc[k, column]
            text = percent(value) if as_percent else num(value, digits)
            if sd and np.isfinite(value):
                text += f" ± {num(values[method].loc[k, column + '_sd'], digits)}"
            cells.append(text)
        rows.append(cells)
    marks = []
    if best:
        candidates = [i for i, m in enumerate(methods) if m in r.models]
        for c, k in enumerate(r.ks, start=1):
            series = pd.Series({i: values[methods[i]].loc[k, column] for i in candidates}).dropna()
            if len(series):
                marks.append((int(series.idxmin() if best == "min" else series.idxmax()), c))
    return table(["metoda"] + [f"K = {k}" for k in r.ks], rows, mark_cells=marks)


# ---- figures ----------------------------------------------------------------------------------


def _legend(fig, ax, count: int = 4):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, [label(x) for x in labels], loc="lower center", ncol=count, fontsize=9, frameon=False)


def fig_curves(r: Results, panels: list[tuple[str, str]], path: Path, methods: list[str], *, columns: int = 3) -> Path:
    plt = _plt()
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.3 * rows), squeeze=False)
    for ax, (column, title) in zip(axes.ravel(), panels):
        for method in methods:
            data = r.rows(method)
            ax.plot(r.ks, data[column], label=method, linewidth=2, markersize=5, **STYLES.get(method, {}))
            if f"{column}_sd" in data and method != "random":
                ax.fill_between(r.ks, data[column] - data[f"{column}_sd"], data[column] + data[f"{column}_sd"],
                                color=STYLES.get(method, {}).get("color"), alpha=0.10, linewidth=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("K"), ax.set_xticks(r.ks), ax.grid(alpha=0.25), ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    _legend(fig, axes.ravel()[0])
    fig.tight_layout(rect=(0, 0.09 if rows == 1 else 0.07, 1, 1))
    return _save(fig, path)


def fig_heatmaps(r: Results, panels: list[tuple[str, np.ndarray, int]], row_names: list[str], path: Path, *, columns: int = 3) -> Path:
    """Sequential single-hue heatmaps (darker = larger) with the values written in the cells."""
    plt = _plt()
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.8 * columns, (0.9 + 0.5 * len(row_names)) * rows), squeeze=False)
    for ax, (title, values, digits) in zip(axes.ravel(), panels):
        low, high = np.nanmin(values), np.nanmax(values)
        ax.imshow(values, cmap="Blues", aspect="auto", vmin=low - 0.05 * (high - low + 1e-9), vmax=high)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    dark = (values[i, j] - low) / (high - low + 1e-9) > 0.6
                    ax.text(j, i, num(values[i, j], digits), ha="center", va="center", fontsize=8.5, color="white" if dark else "black")
        ax.set_xticks(range(len(r.ks)), r.ks), ax.set_yticks(range(len(row_names)), row_names, fontsize=8.5)
        ax.set_xlabel("K"), ax.set_title(title, fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def matrix(r: Results, frame: pd.DataFrame, methods: list[str], column: str, key: str = "method") -> np.ndarray:
    pivot = frame.pivot(index=key, columns="k", values=column)
    return pivot.reindex(index=methods, columns=r.ks).to_numpy(dtype=float)


def fig_distributions(r: Results, path: Path) -> Path:
    """Coverage of every seed against the distribution of random draws, one panel per K."""
    plt = _plt()
    methods = [*r.models, NOISE, "descriptor-kmeans"]
    fig, axes = plt.subplots(2, -(-len(r.ks) // 2), figsize=(4.6 * -(-len(r.ks) // 2), 7.2), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes.ravel(), r.ks):
        base = r.scores[(r.scores["k"] == k)]
        random = base[(base["method"] == "random") & (base["variant"] == "baseline")]["coverage_mean"]
        parts = ax.violinplot([random], positions=[0], widths=0.8, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#9a9a9a"), body.set_alpha(0.55)
        for position, method in enumerate(methods, start=1):
            variant = "baseline" if method in BASELINES else "l2"
            values = base[(base["method"] == method) & (base["variant"] == variant)]["coverage_mean"].to_numpy()
            ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=11, color=STYLES[method]["color"], alpha=0.7, linewidths=0)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color="black", linewidth=1.4)
        ax.axhline(random.mean(), color="#000000", linestyle=":", linewidth=1)
        ax.set_ylim(r.scores["coverage_mean"].quantile(0.001) - 0.05, np.quantile(random, 0.98) + 0.05)
        ax.set_xticks(range(len(methods) + 1), ["losowa"] + [label(m).replace(" (KMeans na cechach)", "") for m in methods], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"K = {k}: pokrycie (mniej = lepiej)", fontsize=10), ax.grid(axis="y", alpha=0.25), ax.tick_params(axis="y", labelsize=8)
    for ax in axes.ravel()[len(r.ks):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def fig_marginal(r: Results, path: Path) -> Path:
    plt = _plt()
    methods = [*r.models, "random", "descriptor-kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    steps = [f"{a}→{b}" for a, b in zip(r.ks[:-1], r.ks[1:])]
    width = 0.8 / len(methods)
    for ax, (column, title, wanted) in zip(axes, (("coverage_drop", "spadek pokrycia po dodaniu jednego kafla", methods),
                                                  ("gap_change", "zmiana G po dodaniu jednego kafla", r.models))):
        w = 0.8 / len(wanted)
        for i, method in enumerate(wanted):
            data = r.marginal[r.marginal["method"] == method].sort_values("k_from")
            ax.bar(np.arange(len(steps)) - 0.4 + w * (i + 0.5), data[column], width=w * 0.9, color=STYLES[method]["color"], label=method)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(steps)), steps), ax.set_title(title, fontsize=10), ax.grid(axis="y", alpha=0.25), ax.set_xlabel("K → K + 1")
    _legend(fig, axes[0], 6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def fig_ranks(r: Results, path: Path) -> Path:
    """Place of every model among the models: per K in the main configuration, and per variant (G averaged over K)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    main = r.summary[(r.summary["variant"] == "l2") & r.summary["method"].isin(r.models)]
    per_k = main.pivot(index="k", columns="method", values="gap").rank(axis=1, ascending=False)
    means = r.summary[r.summary["method"].isin(r.models) & r.summary["variant"].isin(r.variants)]
    per_variant = means.groupby(["variant", "method"])["gap"].mean().unstack().reindex(r.variants).rank(axis=1, ascending=False)
    for ax, ranks, ticks, title in ((axes[0], per_k, [f"K = {k}" for k in per_k.index], "miejsce wg G w każdym K (główny)"),
                                    (axes[1], per_variant, [r.names[v] for v in per_variant.index], "miejsce wg średniego G w wariantach")):
        for method in r.models:
            ax.plot(range(len(ranks)), ranks[method], label=method, linewidth=2, markersize=8, **STYLES[method])
        ax.set_xticks(range(len(ranks)), ticks, rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(1, len(r.models) + 1)), ax.invert_yaxis(), ax.set_title(title, fontsize=10), ax.grid(alpha=0.25)
    _legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


def fig_inertia(r: Results, path: Path) -> Path:
    """Inertia against K, one panel per model (own scale: spaces differ), with the drop from each K to the next."""
    plt = _plt()
    methods = [*r.models, NOISE]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.5 * len(methods), 3.4), squeeze=False)
    main = r.metrics[r.metrics["variant"] == "l2"]
    for ax, method in zip(axes[0], methods):
        data = main[main["method"] == method].groupby("k")["inertia"].mean().reindex(r.ks)
        ax.plot(r.ks, data / data.iloc[0], linewidth=2, markersize=5, **STYLES[method])
        for k, before, after in zip(r.ks[1:], data.iloc[:-1], data.iloc[1:]):
            ax.annotate(f"−{100 * (before - after) / data.iloc[0]:.1f}%".replace(".", ","), (k, after / data.iloc[0]), fontsize=7,
                        textcoords="offset points", xytext=(0, 6), ha="center")
        ax.set_title(label(method), fontsize=10), ax.set_xticks(r.ks), ax.grid(alpha=0.25), ax.tick_params(labelsize=8), ax.set_xlabel("K")
        ax.margins(y=0.15)
    axes[0][0].set_ylabel("inercja / inercja przy najmniejszym K", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def fig_tradeoff(r: Results, path: Path) -> Path:
    """G against the stability of the chosen tiles, one point per (model, K), labelled with K."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    stable = r.stability[r.stability["variant"] == "l2"].set_index(["method", "k"])["identical_or_overlapping"]
    for method in r.models:
        data = r.rows(method)
        x = [stable.loc[(method, k)] for k in r.ks]
        style = {**STYLES[method], "linestyle": "-"}
        ax.plot(x, data["gap"], label=method, linewidth=1, alpha=0.8, markersize=7, **style)
        for xi, yi, k in zip(x, data["gap"], r.ks):
            ax.annotate(str(k), (xi, yi), fontsize=8, textcoords="offset points", xytext=(5, 4))
    ax.set_xlabel("stabilność: udział kafli identycznych lub nakładających się między seedami")
    ax.set_ylabel("G (więcej = lepiej)"), ax.grid(alpha=0.25), ax.legend(frameon=False, fontsize=9)
    ax.set_title("jakość i stabilność wyboru (liczba przy punkcie to K)", fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def make_figures(r: Results, folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    everything = [*r.models, NOISE, *BASELINES]
    stable = r.stability[r.stability["variant"] == "l2"]
    main = r.summary[r.summary["variant"] == "l2"]
    metrics = r.metrics[r.metrics["variant"] == "l2"].groupby(["method", "k"])[["silhouette"]].mean().reset_index()
    methods = [*r.models, NOISE]
    names = [label(m) for m in methods]
    figures = {
        "curves": fig_curves(r, [("coverage_mean", "pokrycie (mniej = lepiej)"), ("gap", "G: część drogi losowa → pułap (więcej)"),
                                 ("coverage_p95", "pokrycie, 95. percentyl (mniej)"), ("rare_categories", "rzadkie kategorie z 6 (więcej)"),
                                 ("jsd_excess", "JS ponad minimum dla K (mniej)"), ("grid_cells", "zajęte pola siatki slajdu (więcej)")],
                             folder / "w1_curves.png", everything),
        "heatmaps": fig_heatmaps(r, [("G (więcej = lepiej)", matrix(r, main, methods, "gap"), 2),
                                     ("stabilność wyboru (identyczne lub nakładające się)", matrix(r, stable, methods, "identical_or_overlapping"), 2),
                                     ("silhouette (kosinusowa)", matrix(r, metrics, methods, "silhouette"), 3)], names, folder / "w2_heatmaps.png"),
        "distributions": fig_distributions(r, folder / "w3_distributions.png"),
        "marginal": fig_marginal(r, folder / "w4_marginal.png"),
        "ranks": fig_ranks(r, folder / "w5_ranks.png"),
        "variants": fig_heatmaps(r, [(f"G: {m}", matrix(r, r.summary[(r.summary["method"] == m) & r.summary["variant"].isin(r.variants)]
                                                        .assign(v=lambda t: t["variant"]), r.variants, "gap", key="v"), 2) for m in r.models],
                                 [r.names[v] for v in r.variants], folder / "w5b_variants.png", columns=2),
        "inertia": fig_inertia(r, folder / "w6_inertia.png"),
        "tradeoff": fig_tradeoff(r, folder / "w7_tradeoff.png"),
    }
    return figures


# ---- the report -------------------------------------------------------------------------------


def build(results: str | Path, out: str | Path, *, notes: str | Path | None = None, pdf: bool = True) -> Path:
    r = Results(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = make_figures(r, out / "figures")
    text = read_notes(notes)
    note = lambda key: f"<div class='note'>{text[key]}</div>" if key in text else ""
    figure = lambda key, caption: picture(f[key], caption, lossless=True)
    with_base = [*r.models, NOISE, "random", "spatial", "descriptor-kmeans"]
    with_noise = [*r.models, NOISE]
    choice = r.choice[r.choice["variant"] == "l2"].set_index("method")
    order = r.order[r.order["variant"] == "l2"].sort_values("rank")
    seeds = r.run["args"]["seeds"]
    slide_tiles = r.run["tiles"]
    args = r.run["args"]
    inits = ", ".join([f"{args['n_init'][0]} (główne)"] + [str(n) for n in args["n_init"][1:]])
    spaces = "; ".join(["L2 w pełnym wymiarze (główna)", f"{r.lattice_tiles} kafli bez nakładania"] + ([f"PCA {args['pca']}"] if args["pca"] else []))

    choice_rows = []
    for method in with_noise:
        c = choice.loc[method]
        choice_rows.append([label(method), int(c.recommended_k), int(c.best_k), c.tied_k.replace(" ", ", "), num(c.recommended_gap, 2),
                            num(c.best_gap, 2), num(c.gap_range, 2), "tak" if c.all_tied else "nie"])
    order_rows = [[int(o.rank), o.method, f"{num(o.gap, 3)} ± {num(o.gap_sd, 3)}",
                   "–" if not isinstance(o.tied_with, str) else o.tied_with] for o in order.itertuples()]

    variant_rows, variant_marks = [], []
    for i, variant in enumerate(r.variants):
        part = r.choice[r.choice["variant"] == variant].set_index("method")
        cells = [r.names[variant]]
        for method in r.models:
            c = part.loc[method]
            cells.append(f"{int(c.recommended_k)} ({c.tied_k.replace(' ', ', ')})")
        variant_rows.append(cells)
    mean_rows = []
    for variant in r.variants:
        part = r.order[r.order["variant"] == variant].set_index("method")
        mean_rows.append([r.names[variant]] + [f"{num(part.loc[m, 'gap'], 3)} ({int(part.loc[m, 'rank'])}.)" for m in r.models])

    marginal_rows = []
    for method in with_base:
        data = r.marginal[r.marginal["method"] == method].sort_values("k_from")
        marginal_rows.append([label(method)] + [f"{num(row.coverage_drop, 3)} / {num(row.gap_change, 2)}" for row in data.itertuples()])
    step_header = [f"{a} → {b}" for a, b in zip(r.ks[:-1], r.ks[1:])]

    ari = r.ari[r.ari["variant"] == "l2"].set_index(["method", "k"])
    stable = r.stability[r.stability["variant"].isin(["l2", "baseline"])].set_index(["method", "k"])
    stability_rows = [[label(m)] + [f"{num(ari.loc[(m, k), 'ari_seed'], 2)} / {percent(stable.loc[(m, k), 'identical'])} / "
                                    f"{percent(stable.loc[(m, k), 'identical_or_overlapping'])}" for k in r.ks] for m in with_noise]
    nest = r.nesting.set_index(["method", "k_from"])
    nest_rows = [[label(m)] + [f"{percent(nest.loc[(m, a), 'identical'])} / {percent(nest.loc[(m, a), 'identical_or_overlapping'])}"
                               for a in r.ks[:-1]] for m in [*with_noise, "random", "descriptor-kmeans"]]
    cluster = r.metrics[r.metrics["variant"] == "l2"].groupby(["method", "k"])[["silhouette", "davies_bouldin", "calinski_harabasz", "inertia"]].mean()
    cluster_tables = "".join(
        f"<div class='keep'><h3>{title}</h3>" + table(["metoda"] + [f"K = {k}" for k in r.ks],
                                                    [[label(m)] + [num(cluster.loc[(m, k), column], digits) for k in r.ks] for m in with_noise])
        + "</div>" for column, title, digits in (("silhouette", "Silhouette kosinusowa (więcej = lepiej)", 3),
                                               ("davies_bouldin", "Davies-Bouldin (mniej; tylko w obrębie modelu)", 2),
                                               ("calinski_harabasz", "Calinski-Harabasz (więcej; tylko w obrębie modelu)", 0),
                                               ("inertia", "Inercja (tylko w obrębie modelu)", 1)))
    jsd_floor = r.rows("random")["jsd_floor"]

    body = f"""
<h1>Ile kafli w rundzie? Dobór K = {r.ks[0]}–{r.ks[-1]}</h1>
<p class="mute">Slajd 24070_25 (jeden pacjent), {slide_tiles} kafli. Grid Search: 4 modele i kontrola szumem, {seeds} seedów. Analiza eksploracyjna
(K = 5 i 10 były znane przed zapisaniem reguły). Zawiera wyniki na preparacie pacjenta: do użytku wewnętrznego.</p>

<h2>1. Odpowiedź</h2>
<div class="box">{text.get('answer', '<p>Wynik reguły w tabeli T8.</p>')}</div>
<div class="keep"><h3>T8. Wynik reguły wyboru K (konfiguracja główna)</h3>
{table(["model", "zalecane K", "najlepsze K", "K w remisie", "G przy zalecanym", "G przy najlepszym", "rozpiętość G po K", "dane nie rozróżniają K"], choice_rows)}
<p class="mute">Zalecane K to najmniejsze K, którego średnie G różni się od najlepszego o mniej niż łączne odchylenie po seedach.</p></div>
<div class="keep"><h3>T9. Kolejność modeli (G uśrednione po K = {r.ks[0]}–{r.ks[-1]})</h3>
{table(["miejsce", "model", "średnie G ± odchylenie po seedach", "remis z"], order_rows, mark_rows=[0, 1])}</div>
{note('after_answer')}

<h2 class="page">2. Co i jak porównano</h2>
{table(["wymiar", "wartości", "rola"], [
    ["K (kafle w rundzie)", ", ".join(map(str, r.ks)), "badane"],
    ["model", ", ".join(r.models) + ", kontrola szumem", "badane"],
    ["seed KMeans", f"0–{seeds - 1}", "powtórzenia"],
    ["przestrzeń", spaces, "odporność"],
    ["n_init", inits, "odporność"],
    ["nakładanie w wyborze", "zakazane (główne), dozwolone", "odporność"],
    ["próbka kafli", f"wszystkie (główne); {args['subsamples']} podpróbek po {round(100 * args['fraction'])}%", "odporność"],
    ["odniesienia", f"losowa ({args['random_draws']} losowań), przestrzenna ({args['spatial_draws']}), pułap: KMeans na cechach z pikseli ({seeds} seedów)",
     "skala"]])}
<p><b>Pokrycie</b>: średnia, po wszystkich kaflach, odległość w przestrzeni 13 cech z pikseli do najbliższego wybranego kafla (mniej = lepiej).
Maleje z każdym dodanym kaflem, dlatego różne K porównuje <b>G = (losowa − model) / (losowa − pułap)</b> przy tym samym K: 0 to poziom losowania,
1 to poziom pułapu, wartość ujemna to gorzej niż losowo. Protokół: <code>docs/architecture/k-selection-experiment.md</code>.</p>

<h2 class="page">3. Wyniki wobec K (konfiguracja główna)</h2>
<div class="keep"><h3>T1. Pokrycie ± odchylenie (mniej = lepiej; wyróżniony najlepszy model w K)</h3>
{metric_table(r, with_base, "coverage_mean", 3, sd=True, best="min")}</div>
<div class="keep"><h3>T2. G ± odchylenie (więcej = lepiej)</h3>{metric_table(r, with_base, "gap", 2, sd=True, best="max")}</div>
<div class="keep"><h3>T2b. Jaką część losowych selekcji pobija pojedynczy seed (średnio po seedach)</h3>
{metric_table(r, [*with_noise, "spatial"], "beats_random", 0, best="max", as_percent=True)}
<h3>T2c. To samo dla najgorszego seeda</h3>{metric_table(r, [*with_noise, "spatial"], "beats_random_min", 0, best="max", as_percent=True)}</div>
{note('coverage')}
{figure("curves", "W1. Miary selekcji wobec K. Pasma: odchylenie po seedach (dla losowej pominięte, bo jest bardzo szerokie).")}
{figure("heatmaps", "W2. Model × K: G, stabilność wyboru i silhouette. Ciemniej = większa wartość.")}
{figure("distributions", "W3. Pokrycie każdego seeda (punkty, kreska = średnia) na tle rozkładu 1000 selekcji losowych (szary).")}
<div class="keep"><h3>T3. Co daje jeden kafel więcej: spadek pokrycia / zmiana G</h3>{table(["metoda"] + step_header, marginal_rows)}</div>
{figure("marginal", "W4. Zysk z dodania jednego kafla: spadek pokrycia (lewy) i zmiana G (prawy).")}
{note('marginal')}

<h2 class="page">4. Miary pomocnicze (nie rozstrzygają)</h2>
<div class="keep"><h3>T4a. Rzadkie kategorie z 6 (więcej)</h3>{metric_table(r, with_base, "rare_categories", 1, best="max")}</div>
<div class="keep"><h3>T4b. Pokrycie, 95. percentyl (mniej)</h3>{metric_table(r, with_base, "coverage_p95", 2, best="min")}</div>
<div class="keep"><h3>T4c. Rozbieżność Jensena-Shannona ponad minimum osiągalne dla K (mniej)</h3>
{metric_table(r, with_base, "jsd_excess", 3, best="min")}
<p class="mute">Minimum dla 8 grup kafli: {", ".join(f"K = {k}: {num(jsd_floor.loc[k], 3)}" for k in r.ks)}.</p></div>
<div class="keep"><h3>T4d. Zajęte pola siatki slajdu (więcej)</h3>{metric_table(r, with_base, "grid_cells", 1, best="max")}</div>
<div class="keep"><h3>T4e. Średnia odległość między wybranymi w przestrzeni cech (więcej = bardziej różne)</h3>
{metric_table(r, with_base, "diversity_mean", 2)}</div>
<div class="keep"><h3>T5. Stabilność między seedami: ARI klastrowań / kafle identyczne / identyczne lub nakładające się</h3>
{table(["metoda"] + [f"K = {k}" for k in r.ks], stability_rows)}</div>
<div class="keep"><h3>T5b. Ile kafli z wyboru K wraca w wyborze K + 1 (ten sam seed): identyczne / identyczne lub nakładające się</h3>
{table(["metoda"] + step_header, nest_rows)}</div>
{note('stability')}
{figure("tradeoff", "W7. Jakość wyboru (G) wobec jego stabilności między seedami.")}

<h2 class="page">5. Miary klastrowania</h2>
{cluster_tables}
{figure("inertia", "W6. Inercja KMeans wobec K (każdy model we własnej skali); liczby: spadek w stosunku do inercji przy najmniejszym K.")}
{note('clustering')}

<h2 class="page">6. Odporność wyniku</h2>
<div class="keep"><h3>T7a. Zalecane K (w nawiasie K w remisie) w każdym wariancie</h3>{table(["wariant"] + r.models, variant_rows)}</div>
<div class="keep"><h3>T7b. G uśrednione po K (w nawiasie miejsce według reguły)</h3>{table(["wariant"] + r.models, mean_rows)}</div>
{figure("ranks", "W5. Miejsce modeli w poszczególnych K i w wariantach.")}
{figure("variants", "W5b. G w każdym wariancie i K.")}
{note('robustness')}

<div class="keep"><h2>7. Ograniczenia</h2>
<ul>
<li>Jeden preparat i jeden pacjent: rozrzut po seedach i podpróbkach nie mówi nic o zmienności między pacjentami.</li>
<li>Miary liczone są z przybliżonych cech z pikseli (kolor, komórkowość, struktury), nie z oceny patomorfologa.</li>
<li>Analiza eksploracyjna: K = 5 i 10 były znane przed zapisaniem reguły.</li>
<li>Przy K ≤ 10 rzadkie kategorie i proporcje typów kafli mierzone są zgrubnie.</li>
<li>Badano jedną rundę po K kafli; wybór w kilku kolejnych rundach to osobne pytanie.</li>
<li>W wariantach bez nakładania i podpróbek skala G (losowa i pułap) jest liczona na wszystkich kaflach.</li>
</ul></div>
{note('limits')}
<h2>Odtworzenie</h2>
<p><code>scripts/run_k_selection.py</code> (obliczenia), <code>scripts/build_k_reports.py</code> (raporty). Tabele CSV leżą w katalogu
wyników <code>analysis/</code>.</p>
"""
    target = out / "raport_dobor_K.html"
    target.write_text(page("Dobór K: ile kafli w rundzie", body), encoding="utf-8")
    if pdf:
        to_pdf(target, out / "raport_dobor_K.pdf")
    return target
