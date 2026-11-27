"""Report 2 (Polish): the best way of choosing 50 tiles shown in pictures, next to the earlier method and random selection.

Every set is its selection of median coverage. Tile numbers 1-50 (cluster by cluster, in the order of picking) are the
same on the 2-D map, the board and the slide map.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import quota_analysis as qa
from . import quota_selection as qs
from .cluster_figures import _plt, _save
from .clustering import fit_kmeans
from .plates_report import _save_png
from .quota_report import EXTRA_STYLE, RESULT_HEADER, name_groups, result_row
from .quota_results import ALLOCATION_NAMES, METHOD_NAMES, METHOD_SHORT, Run, config_name
from .report_html import num, page, percent, picture, read_notes, table, to_pdf
from .selection_figures import slide_mosaic
from .selection_plates import Canvas, comma, font

logger = logging.getLogger("mpn_al.quota_plates")

# Distinct colours for up to 10 clusters (matplotlib tab10), repeated for larger numbers.
TAB10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
         (188, 189, 34), (23, 190, 207)]
GREY = (110, 110, 110)


def colour(cluster: int) -> tuple[int, int, int]:
    return TAB10[cluster % len(TAB10)]


def hex_colour(cluster: int) -> str:
    return "#%02x%02x%02x" % colour(cluster)


class Pictures:
    def __init__(self, run: Run, image_of: Callable[[str], np.ndarray], vectors: Mapping[str, np.ndarray]):
        self.r = run
        self.vectors = dict(vectors)
        self.positions = run.manifest[["x", "y"]].to_numpy(float)
        self.size = int(run.manifest["width"].iloc[0])
        self.maps = pd.read_parquet(run.folder / "projections.parquet")
        self.labels = np.load(run.folder / "labels.npz")
        self._image_of = image_of
        self._cache: dict[int, np.ndarray] = {}

    def image(self, index: int) -> np.ndarray:
        index = int(index)
        if index not in self._cache:
            full = self._image_of(self.r.manifest["tile_id"].iloc[index])
            self._cache[index] = np.asarray(Image.fromarray(full).resize((384, 384), Image.LANCZOS))
        return self._cache[index]

    # ---- the clusters of a chosen set ----------------------------------------------------------

    def clustering(self, chosen: dict) -> np.ndarray | None:
        """For every tile of the slide the number (0 = largest) of its cluster in the chosen set's clustering."""
        if chosen["method"] in ("random", "spatial", "no-clusters"):
            return None
        labels = self.labels[f"{chosen['model']}|{int(chosen['k'])}|{int(chosen['seed'])}"]
        number = {int(c): n for n, c in enumerate(qs.cluster_order(labels))}
        return np.array([number[int(c)] for c in labels])

    def cluster_table(self, chosen: dict) -> pd.DataFrame:
        clusters = self.clustering(chosen)
        rows = []
        for n in range(int(clusters.max()) + 1):
            members = clusters == n
            part = self.r.table[members]
            rows.append({"cluster": n, "size": int(members.sum()), "share": float(members.mean()), "chosen": int((chosen["cluster"] == n).sum()),
                         "white": float(part["white"].mean()), "nuclear_density": float(part["nuclear_density"].mean()),
                         "compact_eosin": float(part["compact_eosin"].mean()), "erythrocyte": float(part["erythrocyte"].mean()),
                         "pigment": float(part["pigment"].mean()), "chosen_from_cluster": int(members[chosen["tiles"]].sum())})
        return pd.DataFrame(rows)

    # ---- figures ------------------------------------------------------------------------------

    def umap(self, chosen: dict, path: Path, title: str) -> Path:
        """Left: the 2-D map coloured by cluster with the 50 chosen tiles numbered. Right: the same coloured by tissue group."""
        plt = _plt()
        data = self.maps[self.maps["model"] == chosen["model"]].set_index("tile_id").loc[self.r.manifest["tile_id"]]
        coords = data[["x", "y"]].to_numpy()
        method = data["method"].iloc[0]
        clusters = self.clustering(chosen)
        fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), gridspec_kw={"width_ratios": [1.25, 1]})
        ax = axes[0]
        ax.scatter(coords[:, 0], coords[:, 1], s=6, c=[hex_colour(int(c)) for c in clusters], alpha=0.35, linewidths=0)
        few = clusters.max() < 10
        for number, (tile, cluster, turn) in enumerate(zip(chosen["tiles"], chosen["cluster"], chosen["turn"]), start=1):
            ax.annotate(str(number), coords[tile], ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=3,
                        bbox={"boxstyle": "square,pad=0.2" if turn == 0 and few else "circle,pad=0.2", "fc": "white",
                              "ec": hex_colour(int(cluster)), "lw": 2.0})
        if few:
            sizes = np.bincount(clusters)
            handles = [plt.Line2D([], [], marker="o", linestyle="", color=hex_colour(n), label=f"klaster {n + 1}: {size} kafli") for n, size in enumerate(sizes)]
            ax.legend(handles=handles, fontsize=8.5, loc="best", frameon=True)
        ax.set_title(f"kolor = klaster, liczby = {len(chosen['tiles'])} wybranych"
                     + (" (kwadrat: pierwszy wzięty z klastra)" if few and chosen["method"] != "one" else ""), fontsize=10.5)
        names = name_groups(self.r)
        groups = self.r.manifest["group"].to_numpy()
        ax = axes[1]
        for g in self.r.group_table.sort_values("tiles", ascending=False)["group"]:
            mask = groups == g
            ax.scatter(coords[mask, 0], coords[mask, 1], s=6, color=hex_colour(int(g)), label=names[int(g)], alpha=0.7, linewidths=0)
        ax.scatter(coords[chosen["tiles"], 0], coords[chosen["tiles"], 1], s=46, facecolors="none", edgecolors="black", linewidths=1.2)
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
        ax.set_title("ta sama mapa, kolor = grupa kafli wg cech z pikseli; kółka = wybrane", fontsize=10.5)
        for ax in axes:
            ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f"{title}: mapa {method.upper() if method == 'umap' else method}", fontsize=12)
        fig.tight_layout()
        return _save(fig, path)

    def sizes(self, chosen: dict, path: Path, title: str) -> Path:
        plt = _plt()
        info = self.cluster_table(chosen)
        few = len(info) <= 10
        fig, axes = plt.subplots(1, 2, figsize=(15, 1.5 + 0.34 * len(info) if few else 5), sharey=few,
                                 gridspec_kw=None if few else {"width_ratios": [2.6, 1]})
        if len(info) <= 10:
            y = np.arange(len(info))
            colours = [hex_colour(n) for n in info["cluster"]]
            axes[0].barh(y, info["size"], color=colours)
            for i, row in enumerate(info.itertuples()):
                axes[0].text(row.size, i, f" {row.size} ({percent(row.share)})", va="center", fontsize=9)
            axes[0].set_yticks(y, [f"klaster {n + 1}" for n in info["cluster"]]), axes[0].invert_yaxis()
            axes[0].set_title("liczba kafli w klastrze", fontsize=11), axes[0].set_xlim(0, info["size"].max() * 1.25)
            axes[1].barh(y, info["chosen"], color=colours, label="wybrano")
            axes[1].scatter(info["share"] * self.r.total, y, marker="|", s=420, color="black", linewidths=2.5, label="proporcjonalnie do wielkości", zorder=3)
            for i, row in enumerate(info.itertuples()):
                axes[1].text(max(row.chosen, row.share * self.r.total) + 0.15, i, str(row.chosen), va="center", fontsize=9)
            axes[1].set_title(f"ile z {self.r.total} kafli wybrano", fontsize=11)
            axes[1].set_xlim(0, max(info["chosen"].max(), info["share"].max() * self.r.total) * 1.12)
            axes[1].legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
        else:
            axes[0].bar(np.arange(len(info)) + 1, info["size"], color="#0072B2")
            for number, value in enumerate(info["size"], start=1):
                axes[0].text(number, value, f" {value}", rotation=90, ha="center", va="bottom", fontsize=7)
            axes[0].set_ylim(0, info["size"].max() * 1.18)
            axes[0].set_title("liczba kafli w każdym klastrze (od największego)", fontsize=11), axes[0].set_xlabel("klaster")
            axes[1].hist(info["size"], bins=15, color="#0072B2", edgecolor="white")
            axes[1].set_title("rozkład wielkości klastrów", fontsize=11), axes[1].set_xlabel("kafli w klastrze"), axes[1].set_ylabel("liczba klastrów")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        return _save(fig, path)

    def board_by_cluster(self, chosen: dict, title: str, *, side: int = 300) -> list[np.ndarray]:
        """The chosen tiles grouped by cluster, with a header per cluster; split into pages that fit a landscape sheet.

        Clusters that give at most six tiles stand two next to each other, so that the pages are used well."""
        info = self.cluster_table(chosen)
        gap, head, below, divider = 10, 86, 34, 50
        counts = np.bincount(chosen["cluster"], minlength=len(info))
        columns = int(min(10, max(5, counts.max())))
        across = 2 if columns <= 6 else 1
        block_width = gap + columns * (side + gap)
        blocks = []
        for row in info.itertuples():
            mine = np.flatnonzero(chosen["cluster"] == row.cluster)
            lines = max(1, -(-len(mine) // columns))
            canvas = Canvas(block_width, head + lines * (side + below + gap))
            canvas.draw.rectangle([0, 6, 16, head - 10], fill=colour(row.cluster))
            canvas.text(26, 8, f"Klaster {row.cluster + 1}: {row.size} kafli ({percent(row.share)} slajdu), wybrano {len(mine)}", 30, bold=True)
            canvas.text(26, 48, f"średnio w klastrze: biel {comma(100 * row.white)}%, jądra {comma(row.nuclear_density / 1000, 1)} tys./mm², "
                        f"kość {comma(100 * row.compact_eosin)}%, krwinki {comma(100 * row.erythrocyte)}%", 23, colour=GREY)
            for place, i in enumerate(mine):
                line, column = divmod(place, columns)
                x, y = gap + column * (side + gap), head + line * (side + below + gap)
                canvas.tile(self.image(chosen["tiles"][i]), x, y, side, frame=colour(row.cluster), width=7)
                first = chosen["turn"][i] == 0 and chosen["method"] in ("m1", "m1p", "m3")
                canvas.text(x, y + side + 5, f"{i + 1}" + ("  (najbliższy środka)" if first else ""), 22, bold=first)
            blocks.append(canvas.array())
        rows = []
        for start in range(0, len(blocks), across):
            group = blocks[start : start + across]
            height = max(b.shape[0] for b in group)
            padded = [np.pad(b, ((0, height - b.shape[0]), (0, 0), (0, 0)), constant_values=255) for b in group]
            while len(padded) < across:
                padded.append(np.full((height, block_width, 3), 255, np.uint8))
            spacer = np.full((height, divider, 3), 255, np.uint8)
            rows.append(np.concatenate([part for b in padded for part in (b, spacer)][:-1], axis=1))
        width = rows[0].shape[1]
        pages, current, used = [], [], 0
        limit = int(width * 0.60)
        for block in rows:
            if current and used + block.shape[0] > limit:
                pages.append(current)
                current, used = [], 0
            current.append(block)
            used += block.shape[0]
        pages.append(current)
        out = []
        for i, group in enumerate(pages):
            header = Canvas(width, 64)
            header.text(gap, 14, f"{title}" + (f" (strona {i + 1} z {len(pages)})" if len(pages) > 1 else ""), 32, bold=True)
            out.append(np.concatenate([header.array(), *group], axis=0))
        return out

    def board_grid(self, chosen: dict, title: str, *, by: str = "cluster", side: int = 300, columns: int = 10) -> np.ndarray:
        """A plain board: ``by="cluster"`` (from the largest cluster, size written under the tile) or ``by="density"``."""
        tiles = chosen["tiles"]
        if by == "density":
            tiles = tiles[np.argsort(self.r.table["nuclear_density"].to_numpy()[tiles], kind="stable")]
            sizes = None
        else:
            info = self.cluster_table(chosen).set_index("cluster")
            sizes = [int(info.loc[c, "size"]) for c in chosen["cluster"]]
        gap, below = 10, 34
        rows = -(-len(tiles) // columns)
        canvas = Canvas(gap + columns * (side + gap), 64 + rows * (side + below + gap))
        canvas.text(gap, 14, title, 32, bold=True)
        for n, index in enumerate(tiles):
            row, column = divmod(n, columns)
            x, y = gap + column * (side + gap), 64 + row * (side + below + gap)
            canvas.tile(self.image(index), x, y, side)
            canvas.text(x, y + side + 5, f"{n + 1}" + (f"  · klaster: {sizes[n]} kafli" if sizes else ""), 21)
        return canvas.array()

    def mosaic(self) -> tuple[np.ndarray, tuple[int, int]]:
        def tiny(index: int) -> np.ndarray:
            full = self._image_of(self.r.manifest["tile_id"].iloc[index])
            return np.asarray(Image.fromarray(full).resize((64, 64), Image.BILINEAR))

        return slide_mosaic(self.positions, self.size, tiny)

    def _cores(self, canvas: np.ndarray, origin, scale: float) -> np.ndarray:
        """The two largest tissue fragments cropped and placed side by side, left to right as on the slide."""
        regions = self.r.manifest["region"].to_numpy()
        crops = []
        for region in (0, 1):
            xy = self.positions[regions == region]
            if not len(xy):
                continue
            x0, y0 = ((xy.min(axis=0) - origin) * scale - 60).astype(int)
            x1, y1 = ((xy.max(axis=0) + self.size - origin) * scale + 60).astype(int)
            crops.append((x0, canvas[max(y0, 0) : y1, max(x0, 0) : x1]))
        if not crops:
            return canvas
        crops = [crop for _, crop in sorted(crops, key=lambda item: item[0])]
        width = sum(c.shape[1] for c in crops) + 30 * (len(crops) - 1)
        height = max(c.shape[0] for c in crops)
        out = np.full((height, width, 3), 255, np.uint8)
        x = 0
        for crop in crops:
            out[: crop.shape[0], x : x + crop.shape[1]] = crop
            x += crop.shape[1] + 30
        return out

    def slide_selected(self, mosaic, origin, chosen: dict, title: str, *, scale: float = 1 / 32) -> np.ndarray:
        faded = Image.fromarray((mosaic.astype(np.float32) * 0.6 + 255 * 0.4).astype(np.uint8))
        draw = ImageDraw.Draw(faded)
        side = int(round(self.size * scale))
        for number, (index, cluster) in enumerate(zip(chosen["tiles"], chosen["cluster"]), start=1):
            x = int((self.positions[index, 0] - origin[0]) * scale)
            y = int((self.positions[index, 1] - origin[1]) * scale)
            outline = colour(int(cluster)) if cluster >= 0 else (0, 90, 220)
            draw.rectangle([x - 6, y - 6, x + side + 6, y + side + 6], outline=outline, width=9)
            draw.text((x + side // 2, y + side // 2), str(number), font=font(34, True), fill=(0, 0, 0), anchor="mm", stroke_width=5,
                      stroke_fill=(255, 255, 255))
        body = self._cores(np.asarray(faded), np.array(origin), scale)
        header = Canvas(body.shape[1], 80)
        header.text(20, 16, title, 46, bold=True)
        return np.concatenate([header.array(), body], axis=0)

    def slide_clusters(self, mosaic, origin, chosen: dict, title: str, *, scale: float = 1 / 32) -> np.ndarray:
        """Every tile of the slide as a small square in the colour of its cluster."""
        clusters = self.clustering(chosen)
        canvas = Image.fromarray((mosaic.astype(np.float32) * 0.25 + 255 * 0.75).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        side = int(round(self.size * scale))
        for index, cluster in enumerate(clusters):
            x = (self.positions[index, 0] - origin[0]) * scale + side / 2
            y = (self.positions[index, 1] - origin[1]) * scale + side / 2
            half = side / 4
            draw.rectangle([x - half, y - half, x + half, y + half], fill=colour(int(cluster)))
        body = self._cores(np.asarray(canvas), np.array(origin), scale)
        header = Canvas(body.shape[1], 90)
        header.text(20, 16, title, 46, bold=True)
        x = 20 + int(header.draw.textlength(title, font=font(46, True))) + 60
        for n in range(int(clusters.max()) + 1):
            header.draw.rectangle([x, 28, x + 34, 62], fill=colour(n))
            header.text(x + 42, 24, str(n + 1), 36)
            x += 120
        return np.concatenate([header.array(), body], axis=0)

    def methods_in_cluster(self, chosen: dict, *, side: int = 250, columns: int = 10) -> tuple[np.ndarray, list[int]]:
        """What each of the four methods takes from the largest, a middle and the smallest cluster of the winner's clustering."""
        model, k, seed, allocation = chosen["model"], int(chosen["k"]), int(chosen["seed"]), chosen["allocation"]
        vectors = self.vectors[model]
        fitted = fit_kmeans(vectors, k, seed)
        stored = self.labels[f"{model}|{k}|{seed}"]
        if not np.array_equal(fitted.labels_, stored):
            raise ValueError("the clustering cannot be reproduced from these embeddings")
        conflicts = np.zeros((len(vectors), len(vectors)), dtype=bool)
        for start in range(0, len(vectors), 512):
            delta = np.abs(self.positions[start : start + 512, None, :] - self.positions[None, :, :])
            conflicts[start : start + 512] = (delta < self.size).all(axis=2)
        picks = {method: qs.select(vectors, fitted.labels_, fitted.cluster_centers_, self.r.total, allocation=allocation, method=method,
                                   conflicts=conflicts, rng=np.random.default_rng((10_000, k, seed, 0, qs.ALLOCATIONS.index(allocation))))
                 for method in qs.METHODS}
        clusters = list(dict.fromkeys([0, k // 2, k - 1]))
        order = qs.cluster_order(fitted.labels_)
        gap, label_width = 8, 400
        longest = max(int((p.cluster == c).sum()) for p in picks.values() for c in clusters)
        columns = min(columns, longest)
        blocks = []
        for c in clusters:
            size = int((fitted.labels_ == order[c]).sum())
            lines = {m: max(1, -(-int((p.cluster == c).sum()) // columns)) for m, p in picks.items()}
            canvas = Canvas(label_width + columns * (side + gap) + gap, 60 + sum(lines.values()) * (side + gap) + 20)
            canvas.draw.rectangle([0, 8, 16, 46], fill=colour(c))
            canvas.text(28, 12, f"Klaster {c + 1} ({size} kafli): co bierze każda metoda", 30, bold=True)
            y = 60
            for method, p in picks.items():
                mine = p.tiles[p.cluster == c]
                canvas.text(8, y + side // 2 - 16, METHOD_NAMES[method], 22, bold=method == chosen["method"])
                for place, tile in enumerate(mine):
                    line, column = divmod(place, columns)
                    canvas.tile(self.image(tile), label_width + column * (side + gap), y + line * (side + gap), side)
                y += lines[method] * (side + gap)
            blocks.append(canvas.array())
        return blocks, clusters


def blinded_pack(pictures: Pictures, sets: Mapping[str, dict], out: Path, *, seed: int = 23, pdf: bool = True) -> dict[str, str]:
    """Unnamed boards of the compared sets, ordered by nuclear density, with a form; the key stays outside the pack."""
    folder = out / "dla_patomorfologa"
    folder.mkdir(parents=True, exist_ok=True)
    names = list(sets)
    order = np.random.default_rng(seed).permutation(len(names))
    key = {}
    for letter, position in zip("PQR", order):
        name = names[int(position)]
        key[letter] = name
        _save_png(folder / f"zestaw_{letter}.png", pictures.board_grid(sets[name], f"Zestaw {letter}", by="density", side=384))
    (out / "blind_key_50.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    questions = ("Różnorodność zestawu (1 = mała, 5 = duża)", "Jak dobrze oddaje to, co jest w preparacie (1–5)",
                 "Numery kafli nieprzydatnych do adnotacji", "Numery kafli prawie identycznych ze sobą", "Czego w zestawie brakuje",
                 "Co jest w nadmiarze")
    letters = sorted(key)
    rows = "".join(f"<tr><td>{q}</td>" + "<td></td>" * len(letters) + "</tr>" for q in questions)
    head = "".join(f"<th>zestaw {x}</th>" for x in letters)
    total = pictures.r.total
    body = (f"<h1>Ocena zestawów po {total} kafli ({', '.join(letters)})</h1><p>Zestawy z tego samego preparatu, wybrane różnymi sposobami; sposoby "
            "ujawnimy po ocenie. Kafle są uporządkowane od najmniej do najbardziej komórkowych (pomiar automatyczny). Powiększenia: pliki "
            "<code>zestaw_*.png</code>. Obrazy preparatu pacjenta: do użytku wewnętrznego.</p>"
            f"<table><thead><tr><th>pytanie</th>{head}</tr></thead><tbody>{rows}</tbody></table>"
            "<p><b>Kolejność zestawów od najlepszego do adnotacji:</b> ______________________</p><p><b>Uzasadnienie i uwagi:</b></p>"
            "<p style='height:110px;border:1px solid #d7d9dd;border-radius:6px'></p><p>Imię i nazwisko: ______________________ &nbsp;&nbsp; Data: ______________</p>"
            + "".join(picture(folder / f"zestaw_{x}.png", f"Zestaw {x}", css="page") for x in letters))
    form = folder / "formularz_oceny.html"
    form.write_text(page("Ocena zestawów kafli", body, extra_style="td{height:44px} td:first-child{width:34%}"), encoding="utf-8")
    if pdf:
        to_pdf(form, folder / "formularz_oceny.pdf")
    with zipfile.ZipFile(out / "pakiet_dla_patomorfologa_PQR.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(folder.iterdir()):
            archive.write(file, file.name)
    return key


def build(results: str | Path, image_of: Callable[[str], np.ndarray], vectors: Mapping[str, np.ndarray], out: str | Path, *,
          notes: str | Path | None = None, pdf: bool = True) -> Path:
    r = Run(results)
    out = Path(out)
    plates = out / "plansze"
    plates.mkdir(parents=True, exist_ok=True)
    p = Pictures(r, image_of, vectors)
    text = read_notes(notes)
    note = lambda key: f"<div class='note'><b>Obserwacje.</b> {text[key]}</div>" if key in text else ""
    how = lambda shows, reading: f"<div class='read'><b>Co pokazuje:</b> {shows} <b>Jak czytać:</b> {reading}</div>"

    winner = r.chosen("zwycięzca siatki")
    earlier = r.chosen(qa.REF_RECOMMENDED)
    random = r.chosen(qa.REF_RANDOM)
    winner_name = config_name(*r.winner)
    earlier_name = config_name(*qa.one_per_cluster(qa.RECOMMENDED_MODEL, r.total))
    recommended, confirmed = r.confirmed_config()

    def plate(name: str, rgb: np.ndarray, caption: str, css: str = "") -> str:
        _save_png(plates / f"{name}.png", rgb)
        logger.info("plate %s", name)
        return picture(rgb, caption, width=2800, quality=86, css=css)

    mosaic, origin = p.mosaic()
    logger.info("slide mosaic ready")
    info = p.cluster_table(winner)
    cluster_rows = [[f"{c.cluster + 1}", c.size, percent(c.share), c.chosen, num(c.share * r.total, 1), percent(c.white),
                     num(c.nuclear_density / 1000, 1), percent(c.compact_eosin), percent(c.erythrocyte), num(100 * c.pigment, 2) + "%"]
                    for c in info.itertuples()]
    earlier_info = p.cluster_table(earlier)

    def sections(chosen: dict, name: str, label: str, prefix: str, notes_prefix: str) -> str:
        umap_path = p.umap(chosen, plates / f"{prefix}_umap.png", label)
        sizes_path = p.sizes(chosen, plates / f"{prefix}_klastry.png", f"{label}: wielkości klastrów")
        out_html = (f"<h2 class='page'>{name}</h2><p>{label}. Seed {chosen['seed']}" + (f", losowanie {chosen['draw']}" if chosen["method"] == "m2" else "")
                    + f", pokrycie {num(chosen['coverage_mean'], 3)} (mediana wśród wszystkich seedów tego wariantu).</p>"
                    + how("rozmieszczenie wszystkich kafli w płaszczyźnie tak, by podobne dla modelu leżały blisko.",
                          "to tylko rzut do 2 wymiarów: klastry liczono w pełnej przestrzeni, a odległości między wyspami są umowne.")
                    + note(f"{notes_prefix}_umap")
                    + picture(umap_path, f"Mapa 2-D z zaznaczonymi {r.total} wybranymi kaflami.", lossless=True, css="short"))
        if chosen["method"] == "one":
            boards = [p.board_grid(chosen, f"{label}: {r.total} kafli, po jednym z klastra (od największego klastra)")]
        else:
            boards = p.board_by_cluster(chosen, f"{label}: {r.total} kafli według klastrów")
        out_html += how("wybrane kafle uporządkowane według klastra, od największego.",
                        "numer kafla jest ten sam na mapie 2-D i na mapie slajdu; nagłówek klastra podaje jego wielkość i średnie cechy.")
        out_html += note(f"{notes_prefix}_plansza") + "".join(
            plate(f"{prefix}_plansza_{i + 1}", rgb, f"Plansza kafli wg klastrów ({i + 1}/{len(boards)}).", css="page short" if i else "short")
            for i, rgb in enumerate(boards))
        out_html += (f"<h3 class='page'>Gdzie na preparacie leżą wybrane kafle</h3>"
                     + how("dwa największe fragmenty tkanki (obrócone tak jak na skanie) z ramkami wybranych kafli.",
                           "kolor ramki to klaster; skupienie ramek w jednym miejscu oznaczałoby wybór jednostronny przestrzennie.")
                     + note(f"{notes_prefix}_mapa")
                     + plate(f"{prefix}_mapa_wsi", p.slide_selected(mosaic, origin, chosen, f"{label}: wybrane kafle"),
                             "Mapa rozmieszczenia wybranych kafli na WSI.", css="short"))
        if chosen["method"] != "one":
            out_html += plate(f"{prefix}_mapa_klastrow", p.slide_clusters(mosaic, origin, chosen, f"{label}: klaster każdego kafla"),
                              "Przynależność wszystkich kafli do klastrów na WSI.", css="short")
        out_html += (f"<div class='keep'><h3>Liczebność klastrów</h3>{note(f'{notes_prefix}_liczebnosc')}"
                     + picture(sizes_path, "Liczebność klastrów i liczba wybranych kafli.", lossless=True, css="short") + "</div>")
        return out_html

    winner_first = confirmed and recommended == r.winner
    letters = ("A", "B") if winner_first else ("B", "A")
    winner_title = f"{letters[0]}. Najlepszy z nowych wariantów (5–10 klastrów)" + (": rekomendowany" if winner_first else "")
    earlier_title = f"{letters[1]}. Dotychczasowa metoda: po 1 kaflu z {r.total} klastrów" + ("" if winner_first else " (rekomendowana)")
    body_winner = sections(winner, winner_title, winner_name, "A_zwyciezca", "zwyciezca")
    body_winner += ("<div class='keep'><h3>Opis klastrów tego wariantu</h3>"
                    + table(["klaster", "kafli", "udział", "wybrano", "proporcjonalnie", "biel", "jądra, tys./mm²", "kość", "krwinki", "barwnik"],
                            cluster_rows) + "</div>")
    method_blocks, shown_clusters = p.methods_in_cluster(winner)
    methods_html = (how("ten sam podział na klastry i te same liczby miejsc, cztery metody wyboru.",
                        "każdy wiersz to inna metoda; pogrubiona nazwa to metoda pokazanego wariantu.") + note("metody")
                    + "".join(plate(f"C_metody_klaster_{c + 1}", block, f"Klaster {c + 1}: porównanie metod wyboru.", css="page short" if i else "short")
                              for i, (block, c) in enumerate(zip(method_blocks, shown_clusters))))
    body_earlier = sections(earlier, earlier_title, earlier_name, "B_dotychczasowa", "dotychczasowa")
    body_earlier += (f"<p>Wielkości {len(earlier_info)} klastrów: od {int(earlier_info['size'].min())} do {int(earlier_info['size'].max())} kafli "
                     f"(mediana {num(earlier_info['size'].median(), 0)}).</p>")
    first_part, second_part = (body_winner, body_earlier) if winner_first else (body_earlier, body_winner)

    rows = [result_row(r, "zwycięzca siatki", r.winner), result_row(r, "dotychczasowa (lunit, 1 z klastra)",
                                                                    qa.one_per_cluster(qa.RECOMMENDED_MODEL, r.total)),
            result_row(r, "losowa", ("random", 0, "-", "random"))]
    shared = []
    sets = {"zwycięzca": winner, "dotychczasowa": earlier, "losowa": random}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]]["tiles"], sets[names[j]]["tiles"]
            near = (np.abs(p.positions[a][:, None, :] - p.positions[b][None, :, :]) < p.size).all(axis=2)
            shared.append([f"{names[i]} / {names[j]}", len(set(a.tolist()) & set(b.tolist())), int(near.any(axis=1).sum())])

    key = blinded_pack(p, sets, out, pdf=pdf)
    logger.info("blinded pack: %s", key)

    status = (f"<b>Rekomendowany jest nowy wariant</b>: {winner_name}. Przeszedł wszystkie testy potwierdzające." if confirmed and recommended == r.winner else
              f"<b>Rekomendowana pozostaje {'dotychczasowa metoda' if not confirmed else 'inna konfiguracja'}</b>: {config_name(*recommended)}. "
              f"Najlepszy z nowych wariantów ({winner_name}) nie przeszedł testów potwierdzających (raport 1, sekcja 8). "
              "Pokazujemy najpierw metodę rekomendowaną, potem nowy wariant, żeby można je było porównać.")
    body = f"""
<h1>Wybrany wariant w obrazach: które 50 kafli trafia do adnotacji</h1>
<p class="mute">Jeden preparat (slajd 24070_25, jeden pacjent), {len(r.manifest)} kafli. Raport 2 z 2; metodę i liczby opisuje raport 1
(„Eksperyment i wyniki”). Obrazy preparatu pacjenta: do użytku wewnętrznego.</p>
<div class="warn"><b>Przed rozesłaniem:</b> osoba oceniająca zaślepione zestawy P, Q, R (a także wcześniejsze A, B, C) powinna to zrobić <b>zanim</b>
zobaczy ten raport.</div>
<h2>Jak to czytać (dla osoby, która nie zna projektu)</h2>
<p>Preparat pocięto na kwadratowe kafle. Mamy wybrać z nich <b>{r.total}</b> do ręcznej adnotacji tak, by były różnorodne i oddawały preparat.
Model komputerowy zamienia każdy kafel w wektor, algorytm KMeans dzieli kafle na <b>klastry</b> (grupy podobnych dla modelu), a z klastrów bierzemy
kafle.</p>
<div class="box">{status}</div>
<ul>
<li>Pokazujemy zestaw o <b>medianowym pokryciu</b> wśród wszystkich powtórzeń, a nie najładniejszy.</li>
<li>Numery 1–{r.total} są te same na mapie 2-D, planszy i mapie slajdu danego zestawu.</li>
<li>Metody wyboru w klastrze: {"; ".join(METHOD_NAMES[m] for m in qs.METHODS)}.</li>
<li><b>Pokrycie</b> (mniej = lepiej): jak podobny jest typowy kafel slajdu do najbliższego z wybranych.</li>
</ul>
<div class="keep"><h3>Trzy porównywane zestawy: średnie po powtórzeniach</h3>{table(RESULT_HEADER, rows, mark_rows=[0])}</div>
{text.get('podsumowanie', '')}
{first_part}
{second_part}
<h2 class="page">C. Czym różnią się metody wyboru wewnątrz klastra</h2>
{methods_html}
<h2 class="page">D. Selekcja losowa i porównanie</h2>
{plate("D_losowa_plansza", p.board_grid(random, f"Losowa: {r.total} kafli (od najmniej do najbardziej komórkowych)", by="density"), "Selekcja losowa (mediana).")}
{plate("D_losowa_mapa", p.slide_selected(mosaic, origin, random, "Losowa: wybrane kafle"), "Mapa rozmieszczenia, selekcja losowa.", css="page")}
<div class="keep"><h3>Ile kafli mają wspólnych pokazane zestawy</h3>{table(["para", f"identyczne (z {r.total})", "identyczne lub nakładające się"], shared)}</div>
{note('porownanie')}
<h2>Zaślepiona ocena</h2>
<p>Pakiet <code>pakiet_dla_patomorfologa_PQR.zip</code> zawiera trzy zestawy bez nazw i formularz. Klucz leży poza pakietem (<code>blind_key_50.json</code>).
Pliki plansz w pełnej rozdzielczości: katalog <code>plansze/</code>.</p>
<h2>Ograniczenia</h2>
<ul><li>Jeden preparat i jeden pacjent.</li><li>Obserwacje z oglądania plansz to wrażenia z jednego powtórzenia, nie pomiary; ich autor nie jest patomorfologiem.</li></ul>
"""
    target = out / "raport_wybrany_wariant.html"
    target.write_text(page("50 kafli: wybrany wariant w obrazach", body, extra_style=EXTRA_STYLE), encoding="utf-8")
    if pdf:
        to_pdf(target, out / "raport_wybrany_wariant.pdf", timeout=600)
    return target
