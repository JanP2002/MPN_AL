"""Plates that compare what the best models select for K = 5-10 (``docs/architecture/k-selection-experiment.md``).

Rules: every plate shows the seed of median coverage (never a hand-picked one); a tile's number is the place of its
cluster counted from the largest, the same on the board, the slide map and the 2-D map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from . import selection as sel
from .cluster_figures import _plt, _save
from .selection_figures import slide_mosaic
from .tile_descriptors import FEATURES, RARE_FEATURES, standardize

GREEN, ORANGE, BLUE, GREY, BLACK = (0, 140, 60), (230, 130, 0), (0, 90, 220), (110, 110, 110), (20, 20, 20)
CATEGORIES = ("ten sam kafel", "nakładają się", "podobne cechy", "różne")
CATEGORY_COLOURS = (GREEN, ORANGE, (90, 90, 200), GREY)
FEATURE_NAMES = {"white": "biel", "fat": "tłuszcz", "erythrocyte": "erytrocyty", "pigment": "barwnik", "compact_eosin": "zbita eozynofilowa (kość)",
                 "nuclear_area": "pole jąder", "nuclear_density": "gęstość jąder", "nuclear_size": "wielkość jąder", "large_nuclei": "duże jądra",
                 "h_median": "hematoksylina (mediana)", "e_median": "eozyna (mediana)", "h_high": "hematoksylina (95. percentyl)", "texture": "tekstura"}
RARE_NAMES = {"compact_eosin": "kość", "erythrocyte": "krew", "pigment": "barwnik", "nuclear_density": "gęste jądra",
              "large_nuclei": "duże jądra", "white": "dużo bieli"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """DejaVu Sans, shipped with matplotlib: it has the Polish letters the built-in OpenCV fonts lack."""
    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


class Canvas:
    def __init__(self, width: int, height: int):
        self.image = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def tile(self, rgb: np.ndarray, x: int, y: int, side: int, *, frame: tuple | None = None, width: int = 8):
        self.image.paste(Image.fromarray(rgb).resize((side, side), Image.LANCZOS), (x, y))
        if frame:
            self.draw.rectangle([x, y, x + side - 1, y + side - 1], outline=frame, width=width)

    def text(self, x: int, y: int, text: str, size: int = 20, *, bold: bool = False, colour: tuple = BLACK, anchor: str = "la"):
        self.draw.text((x, y), text, font=font(size, bold), fill=colour, anchor=anchor)

    def array(self) -> np.ndarray:
        return np.asarray(self.image)


def comma(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def cluster_order(labels: np.ndarray) -> np.ndarray:
    """Cluster ids from the largest to the smallest, the order :func:`selection.representatives` serves them in."""
    clusters, sizes = np.unique(labels, return_counts=True)
    return clusters[np.argsort(-sizes, kind="stable")]


def match_pairs(a: Sequence[int], b: Sequence[int], positions: np.ndarray, size: int, D: np.ndarray, threshold: float) -> list[tuple[int, int, int]]:
    """One-to-one pairs ``(tile of a, tile of b, category)``: identical first, then overlapping, then by descriptor distance."""
    free_a, free_b = [int(i) for i in a], [int(i) for i in b]
    pairs: list[tuple[int, int, int]] = []
    for i in list(free_a):
        if i in free_b:
            pairs.append((i, i, 0))
            free_a.remove(i), free_b.remove(i)
    for i in list(free_a):
        hits = [j for j in free_b if sel.overlaps(positions, i, [j], size)[0]]
        if hits:
            j = min(hits, key=lambda h: float(((positions[h] - positions[i]) ** 2).sum()))
            pairs.append((i, j, 1))
            free_a.remove(i), free_b.remove(j)
    while free_a and free_b:
        distance = np.sqrt(((D[free_a][:, None, :] - D[free_b][None, :, :]) ** 2).sum(axis=2))
        x, y = np.unravel_index(np.argmin(distance), distance.shape)
        pairs.append((free_a[x], free_b[y], 2 if distance[x, y] <= threshold else 3))
        free_a.pop(int(x)), free_b.pop(int(y))
    return sorted(pairs, key=lambda p: (p[2], float(np.linalg.norm(D[p[0]] - D[p[1]]))))


def progression_columns(rows: Sequence[Sequence[int]], positions: np.ndarray, size: int) -> list[dict[int, tuple[int, int, int]]]:
    """Places the tiles of growing K so that a tile that stays keeps its column.

    ``rows`` are the selections for increasing K (cluster order). Returns per row ``{column: (tile, rank, kept)}`` with
    ``kept`` 2 = the same tile as in the row above, 1 = overlaps one of its tiles, 0 = new.
    """
    out: list[dict[int, tuple[int, int, int]]] = []
    previous: dict[int, tuple[int, int, int]] = {}
    for tiles in rows:
        current: dict[int, tuple[int, int, int]] = {}
        waiting = [(int(t), rank) for rank, t in enumerate(tiles)]
        for kind in (2, 1):
            for tile, rank in list(waiting):
                for column, (old, _, _) in previous.items():
                    same = old == tile if kind == 2 else bool(sel.overlaps(positions, tile, [old], size)[0])
                    if same and column not in current:
                        current[column] = (tile, rank, kind)
                        waiting.remove((tile, rank))
                        break
        column = 0
        for tile, rank in waiting:
            while column in current:
                column += 1
            current[column] = (tile, rank, 0)
        out.append(current)
        previous = current
    return out


class Study:
    """Everything the plates need, read from a grid-search folder, the tile images and the embeddings."""

    def __init__(self, results, image_of: Callable[[str], np.ndarray], vectors: Mapping[str, np.ndarray]):
        self.r = results
        folder = results.folder
        self.manifest = pd.read_parquet(folder / "manifest.parquet")
        self.table = pd.read_parquet(folder / "descriptors.parquet")
        self.D = standardize(self.table)
        self.positions = self.manifest[["x", "y"]].to_numpy(dtype=float)
        self.size = int(self.manifest["width"].iloc[0])
        self.flags = sel.rare_flags(self.table, RARE_FEATURES)
        self.threshold = float(np.quantile(np.sqrt(((self.D[::3][:, None, :] - self.D[1::3][None, :, :]) ** 2).sum(axis=2)), 0.05))
        picks = pd.read_parquet(folder / "selections.parquet")
        picks = picks[(picks["constraint"] == "no-overlap") & picks["variant"].isin(["l2", "baseline"])].sort_values("rank")
        self.selected = {(m, int(k), int(s)): g["tile_index"].to_numpy()
                         for (m, k, s), g in picks.groupby(["method", "k", "seed"], sort=False)}
        self.labels = np.load(folder / "labels_l2.npz")
        self.maps = pd.read_parquet(folder / "projections.parquet") if (folder / "projections.parquet").exists() else None
        self.vectors = dict(vectors)
        self.models = results.top_models(2)
        self.sets = [*self.models, "random"]
        self.ks = results.ks
        self._seeds = {(r.method, int(r.k)): int(r.seed) for r in results.seeds.itertuples()}
        self._image_of = image_of
        self._cache: dict[int, np.ndarray] = {}

    # ---- access -------------------------------------------------------------------------------

    def image(self, index: int) -> np.ndarray:
        index = int(index)
        if index not in self._cache:
            full = self._image_of(self.manifest["tile_id"].iloc[index])
            self._cache[index] = np.asarray(Image.fromarray(full).resize((512, 512), Image.LANCZOS))
        return self._cache[index]

    def seed(self, method: str, k: int) -> int:
        return self._seeds[(method, k)]

    def pick(self, method: str, k: int, seed: int | None = None) -> np.ndarray:
        return self.selected[(method, k, self.seed(method, k) if seed is None else seed)]

    def shares(self, method: str, k: int, seed: int | None = None) -> np.ndarray:
        """Share of the slide's tiles in the cluster of each number (empty for the random selection)."""
        if method not in self.models:
            return np.array([])
        labels = self.labels[f"{method}|{k}|{self.seed(method, k) if seed is None else seed}"]
        return np.array([(labels == c).mean() for c in cluster_order(labels)])

    def facts(self, index: int) -> str:
        row = self.table.iloc[int(index)]
        return f"biel {comma(100 * row['white'])}% · jądra {comma(row['nuclear_density'] / 1000, 1)} tys. · kość {comma(100 * row['compact_eosin'])}%"

    def position(self, index: int) -> str:
        x, y = self.positions[int(index)]
        return f"x{int(x)} y{int(y)}"

    # ---- numbers ------------------------------------------------------------------------------

    def contents(self) -> pd.DataFrame:
        """What a selection contains, averaged over all seeds (random: all draws), per (method, k)."""
        rows = []
        for method in self.sets:
            for k in self.ks:
                picks = [v for (m, kk, _), v in self.selected.items() if m == method and kk == k]
                row = {"method": method, "k": k, "selections": len(picks),
                       "white": float(np.mean([self.table["white"].to_numpy()[p].mean() for p in picks])),
                       "nuclear_density": float(np.mean([self.table["nuclear_density"].to_numpy()[p].mean() for p in picks]))}
                for column, name in enumerate(RARE_FEATURES):
                    row[f"tiles_{name}"] = float(np.mean([self.flags[p, column].sum() for p in picks]))
                    row[f"has_{name}"] = float(np.mean([self.flags[p, column].any() for p in picks]))
                rows.append(row)
        return pd.DataFrame(rows)

    def pair_counts(self) -> pd.DataFrame:
        """Categories of the matched tiles of the two best models: representative seeds, and the mean over equal seeds."""
        a, b = self.models
        rows = []
        for k in self.ks:
            shown = match_pairs(self.pick(a, k), self.pick(b, k), self.positions, self.size, self.D, self.threshold)
            seeds = sorted({s for (m, kk, s) in self.selected if m == a and kk == k} & {s for (m, kk, s) in self.selected if m == b and kk == k})
            every = [match_pairs(self.pick(a, k, s), self.pick(b, k, s), self.positions, self.size, self.D, self.threshold) for s in seeds]
            for c, name in enumerate(CATEGORIES):
                rows.append({"k": k, "category": name, "shown": sum(p[2] == c for p in shown),
                             "mean_over_seeds": float(np.mean([sum(p[2] == c for p in pairs) for pairs in every]))})
        return pd.DataFrame(rows)

    def frequency(self, method: str, k: int) -> pd.Series:
        """How many seeds chose each tile (tiles never chosen are absent), most frequent first."""
        chosen = np.concatenate([v for (m, kk, s), v in self.selected.items() if m == method and kk == k])
        counts = pd.Series(chosen).value_counts()
        return counts.sort_index().sort_values(ascending=False, kind="stable")

    # ---- plates -------------------------------------------------------------------------------

    def overview(self, k: int, *, side: int = 384) -> np.ndarray:
        """P1: one row per set (two best models, random), ``k`` large tiles each."""
        left, gap, below = 250, 12, 66
        canvas = Canvas(left + k * (side + gap) + gap, 62 + 3 * (side + below + gap))
        canvas.text(gap, 12, f"K = {k}: wybór dwóch najlepszych modeli i losowy (seed o medianowym pokryciu)", 30, bold=True)
        a, b = (set(map(int, self.pick(m, k))) for m in self.models)
        for row, method in enumerate(self.sets):
            y = 62 + row * (side + below + gap)
            picks = self.pick(method, k)
            shares = self.shares(method, k)
            if method == "random":
                density = self.table["nuclear_density"].to_numpy()
                picks = np.array(sorted(picks, key=lambda i: density[i]))
            canvas.text(gap, y + side // 2 - 24, method if method != "random" else "losowa", 28, bold=True)
            canvas.text(gap, y + side // 2 + 16, f"seed {self.seed(method, k)}" if method != "random" else f"losowanie {self.seed(method, k)}", 20, colour=GREY)
            other = b if method == self.models[0] else a
            for column, index in enumerate(picks):
                x = left + column * (side + gap)
                frame = None
                if method != "random":
                    if int(index) in other:
                        frame = GREEN
                    elif sel.overlaps(self.positions, int(index), list(other), self.size).any():
                        frame = ORANGE
                canvas.tile(self.image(index), x, y, side, frame=frame)
                title = f"{column + 1}" + (f" · klaster {comma(100 * shares[column])}% kafli" if len(shares) else "")
                canvas.text(x, y + side + 6, title, 22, bold=True)
                canvas.text(x, y + side + 36, self.facts(index), 19, colour=GREY)
        return canvas.array()

    def pairs(self, k: int, *, side: int = 384) -> np.ndarray:
        """P4: the two best models side by side, tiles matched one to one."""
        a, b = self.models
        pairs = match_pairs(self.pick(a, k), self.pick(b, k), self.positions, self.size, self.D, self.threshold)
        numbers = [{int(t): n + 1 for n, t in enumerate(self.pick(m, k))} for m in self.models]
        left, gap = 250, 14
        canvas = Canvas(left + len(pairs) * (side + gap) + gap, 62 + 2 * (side + 36) + 56)
        canvas.text(gap, 12, f"K = {k}: pary kafli {a} i {b} (dopasowanie jeden do jednego)", 30, bold=True)
        for row, method in enumerate(self.models):
            y = 62 + row * (side + 36)
            canvas.text(gap, y + side // 2 - 14, method, 28, bold=True)
            for column, pair in enumerate(pairs):
                x = left + column * (side + gap)
                canvas.tile(self.image(pair[row]), x, y, side, frame=CATEGORY_COLOURS[pair[2]], width=10)
                canvas.text(x, y + side + 6, f"nr {numbers[row][pair[row]]} · {self.position(pair[row])}", 18, colour=GREY)
        for column, pair in enumerate(pairs):
            canvas.text(left + column * (side + gap), 62 + 2 * (side + 36) + 8, CATEGORIES[pair[2]], 26, bold=True,
                        colour=CATEGORY_COLOURS[pair[2]])
        return canvas.array()

    def progression(self, method: str, *, side: int = 300) -> np.ndarray:
        """P2: the selections for K = 5..10 of one seed; a tile that stays keeps its column."""
        seed = self.seed(method, 0)
        rows = [self.pick(method, k, seed) for k in self.ks]
        placed = progression_columns(rows, self.positions, self.size)
        columns = 1 + max(max(row) for row in placed)
        left, gap, below = 130, 12, 34
        canvas = Canvas(left + columns * (side + gap) + gap, 62 + len(self.ks) * (side + below + gap))
        title = f"{method if method != 'random' else 'losowa'}: jak zmienia się wybór wraz z K (seed {seed}); zielony: ten sam kafel co wiersz wyżej, pomarańczowy: nakłada się"
        canvas.text(gap, 12, title, 26, bold=True)
        for r, (k, row) in enumerate(zip(self.ks, placed)):
            y = 62 + r * (side + below + gap)
            canvas.text(gap, y + side // 2 - 16, f"K = {k}", 30, bold=True)
            shares = self.shares(method, k, seed)
            for column, (tile, rank, kept) in row.items():
                x = left + column * (side + gap)
                canvas.tile(self.image(tile), x, y, side, frame={2: GREEN, 1: ORANGE, 0: None}[kept] if r else None)
                caption = f"nr {rank + 1}" + (f" · {comma(100 * shares[rank])}%" if len(shares) else "")
                canvas.text(x, y + side + 5, caption, 20, colour=GREY)
        return canvas.array()

    def cluster_cards(self, method: str, k: int, *, near: int = 8, far: int = 4, side: int = 200) -> np.ndarray:
        """P3: per cluster its representative, the members closest to the centre and the farthest ones."""
        seed = self.seed(method, k)
        labels = self.labels[f"{method}|{k}|{seed}"]
        picks = self.pick(method, k)
        vectors = self.vectors[method]
        unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        left, gap, split = 330, 8, 40
        canvas = Canvas(left + (1 + near + far) * (side + gap) + 2 * split + gap, 100 + k * (side + gap + 6))
        canvas.text(gap, 10, f"{method}, K = {k} (seed {seed}): co reprezentuje każdy wybrany kafel", 28, bold=True)
        x_near = left + side + gap + split
        x_far = x_near + near * (side + gap) + split
        for x, heading in ((left, "wybrany"), (x_near, f"{near} najbliższych środka klastra (bez nakładających się)"), (x_far, f"{far} najdalszych")):
            canvas.text(x, 58, heading, 22, bold=True, colour=GREY)
        for number, cluster in enumerate(cluster_order(labels)):
            y = 100 + number * (side + gap + 6)
            members = np.flatnonzero(labels == cluster)
            centre = vectors[members].mean(axis=0)
            order = members[np.argsort(-(unit[members] @ (centre / max(np.linalg.norm(centre), 1e-12))), kind="stable")]
            chosen = int(picks[number])
            used = [chosen]

            def distinct(candidates, count):
                found = []
                for i in candidates:
                    if len(found) == count:
                        break
                    if not sel.overlaps(self.positions, int(i), used, self.size).any():
                        found.append(int(i))
                        used.append(int(i))
                return found

            closest = distinct(order, near)
            farthest = distinct(order[::-1], far)
            row = self.table.iloc[members]
            for line, text in enumerate((f"klaster {number + 1}: {len(members)} kafli ({comma(100 * len(members) / len(labels))}%)",
                                         "średnio w klastrze:", f"biel {comma(100 * row['white'].mean())}%",
                                         f"jądra {comma(row['nuclear_density'].mean() / 1000, 1)} tys./mm²",
                                         f"kość {comma(100 * row['compact_eosin'].mean())}%", f"erytrocyty {comma(100 * row['erythrocyte'].mean())}%")):
                canvas.text(gap, y + 6 + 30 * line, text, 22 if line == 0 else 19, bold=line == 0, colour=BLACK if line == 0 else GREY)
            canvas.tile(self.image(chosen), left, y, side, frame=BLUE)
            for offset, i in enumerate(closest):
                canvas.tile(self.image(i), x_near + offset * (side + gap), y, side)
            for offset, i in enumerate(farthest):
                canvas.tile(self.image(i), x_far + offset * (side + gap), y, side)
        return canvas.array()

    def seed_variability(self, method: str, k: int, *, seeds: int = 8, side: int = 220) -> np.ndarray:
        """P5: the selections of the first seeds (no choosing), then the tiles most often chosen over all seeds."""
        density = self.table["nuclear_density"].to_numpy()
        frequency = self.frequency(method, k)
        total = sum(1 for (m, kk, _) in self.selected if m == method and kk == k)
        left, gap = 170, 8
        canvas = Canvas(left + k * (side + gap) + gap, 62 + seeds * (side + gap) + side + 80)
        canvas.text(gap, 12, f"{method}, K = {k}: kolejne seedy (kafle od najmniej do najbardziej komórkowych)", 24, bold=True)
        for row in range(seeds):
            y = 62 + row * (side + gap)
            canvas.text(gap, y + side // 2 - 12, f"seed {row}", 24, bold=True)
            for column, index in enumerate(sorted(self.pick(method, k, row), key=lambda i: density[i])):
                canvas.tile(self.image(index), left + column * (side + gap), y, side)
        y = 62 + seeds * (side + gap) + 24
        canvas.text(gap, y + side // 2 - 24, "najczęstsze", 24, bold=True)
        canvas.text(gap, y + side // 2 + 10, f"z {total} seedów", 20, colour=GREY)
        for column, (index, count) in enumerate(frequency.head(k).items()):
            x = left + column * (side + gap)
            canvas.tile(self.image(index), x, y, side, frame=BLUE, width=5)
            canvas.text(x, y + side + 6, f"{count} / {total} seedów", 20, colour=GREY)
        return canvas.array()

    def seed_variability_both(self, ks: Mapping[str, int]) -> np.ndarray:
        """The two models' P5 plates next to each other (``ks``: the K shown for each model)."""
        plates = [self.seed_variability(m, ks[m]) for m in self.models]
        height = max(p.shape[0] for p in plates)
        padded = [np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in plates]
        divider = np.full((height, 60, 3), 255, np.uint8)
        divider[:, 28:32] = 150
        return np.concatenate([padded[0], divider, padded[1]], axis=1)

    def mosaic(self) -> tuple[np.ndarray, tuple[int, int]]:
        def small(index: int) -> np.ndarray:  # read directly: caching every tile of the slide would need gigabytes
            full = self._image_of(self.manifest["tile_id"].iloc[index])
            return np.asarray(Image.fromarray(full).resize((64, 64), Image.BILINEAR))

        canvas, origin = slide_mosaic(self.positions, self.size, small)
        return canvas, origin

    def slide_numbers(self, mosaic, origin, k: int, *, scale: float = 1 / 32) -> np.ndarray:
        """P6: where the numbered tiles of the two best models and of the random selection lie."""
        panels = []
        faded = Image.fromarray((mosaic.astype(np.float32) * 0.72 + 255 * 0.28).astype(np.uint8))
        side = int(round(self.size * scale))
        for method in self.sets:
            panel = faded.copy()
            draw = ImageDraw.Draw(panel)
            for number, index in enumerate(self.pick(method, k), start=1):
                x = int((self.positions[index, 0] - origin[0]) * scale)
                y = int((self.positions[index, 1] - origin[1]) * scale)
                draw.rectangle([x - 8, y - 8, x + side + 8, y + side + 8], outline=BLUE, width=8)
                if method != "random":
                    draw.text((x + side // 2, y - 14), str(number), font=font(96, True), fill=BLACK, anchor="md",
                              stroke_width=8, stroke_fill=(255, 255, 255))
            draw.rectangle([0, 0, panel.width - 1, panel.height - 1], outline=GREY, width=4)
            draw.text((30, 20), f"{method if method != 'random' else 'losowa'}, K = {k}", font=font(70, True), fill=BLACK)
            panels.append(np.asarray(panel))
        return np.concatenate(panels, axis=0)

    def slide_frequency(self, mosaic, origin, *, scale: float = 1 / 32) -> np.ndarray:
        """How often a tile is chosen over all seeds and all K: darker and larger squares = more often."""
        panels = []
        faded = (mosaic.astype(np.float32) * 0.45 + 255 * 0.55).astype(np.uint8)
        side = int(round(self.size * scale))
        counts = {m: sum((self.frequency(m, k).reindex(range(len(self.table)), fill_value=0) for k in self.ks)) for m in self.models}
        highest = max(c.max() for c in counts.values())
        for method in self.models:
            panel = Image.fromarray(faded.copy())
            draw = ImageDraw.Draw(panel)
            for index, count in counts[method].sort_values().items():
                if count <= 0:
                    continue
                strength = float(np.sqrt(count / highest))
                x = (self.positions[index, 0] - origin[0]) * scale + side / 2
                y = (self.positions[index, 1] - origin[1]) * scale + side / 2
                half = 5 + 20 * strength
                colour = tuple(int((1 - strength) * c + strength * d) for c, d in zip((190, 215, 245), (0, 40, 130)))
                draw.rectangle([x - half, y - half, x + half, y + half], fill=colour)
            draw.rectangle([0, 0, panel.width - 1, panel.height - 1], outline=GREY, width=4)
            draw.text((30, 20), f"{method}: częstość wyboru kafla (wszystkie seedy i K)", font=font(70, True), fill=BLACK)
            panels.append(np.asarray(panel))
        return np.concatenate(panels, axis=0)

    def space_maps(self, k: int, path: Path) -> Path:
        """P7: the 2-D maps of the two best models, coloured by cluster, with the numbered representatives."""
        if self.maps is None:
            raise ValueError("no projections.parquet in the results folder")
        plt = _plt()
        colours = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        for ax, method in zip(axes, self.models):
            data = self.maps[self.maps["model"] == method].set_index("tile_id").loc[self.manifest["tile_id"]]
            coords = data[["x", "y"]].to_numpy()
            labels = self.labels[f"{method}|{k}|{self.seed(method, k)}"]
            order = cluster_order(labels)
            number_of = {int(c): n for n, c in enumerate(order)}
            ax.scatter(coords[:, 0], coords[:, 1], s=5, c=[colours(number_of[int(c)]) for c in labels], linewidths=0)
            for number, index in enumerate(self.pick(method, k), start=1):
                ax.annotate(str(number), coords[index], ha="center", va="center", fontsize=9, fontweight="bold",
                            bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": "black", "lw": 1.2})
            ax.set_title(f"{method}, K = {k} ({data['method'].iloc[0]}); kolor = klaster, liczba = wybrany kafel", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        return _save(fig, path)

    def profiles(self, path: Path) -> Path:
        """P8: mean standardised descriptor of the chosen tiles (all seeds) minus nothing: 0 is the slide's average."""
        plt = _plt()
        fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
        values = {}
        for method in self.sets:
            values[method] = np.stack([np.mean([self.D[v].mean(axis=0) for (m, kk, _), v in self.selected.items() if m == method and kk == k], axis=0)
                                       for k in self.ks], axis=1)
        limit = max(np.abs(v).max() for v in values.values())
        for ax, method in zip(axes, self.sets):
            image = ax.imshow(values[method], cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            for i in range(values[method].shape[0]):
                for j in range(values[method].shape[1]):
                    ax.text(j, i, comma(values[method][i, j], 2), ha="center", va="center", fontsize=8,
                            color="white" if abs(values[method][i, j]) > 0.65 * limit else "black")
            ax.set_xticks(range(len(self.ks)), self.ks), ax.set_xlabel("K")
            ax.set_title(method if method != "random" else "losowa", fontsize=11)
        axes[0].set_yticks(range(len(FEATURES)), [FEATURE_NAMES[f] for f in FEATURES], fontsize=9)
        fig.colorbar(image, ax=axes, shrink=0.8, label="odchylenia standardowe od średniej slajdu (0 = średnia)")
        return _save(fig, path)

    def blind_set(self, indices: Sequence[int], title: str, *, side: int = 512, columns: int = 5) -> np.ndarray:
        density = self.table["nuclear_density"].to_numpy()
        ordered = sorted(indices, key=lambda i: density[i])
        gap = 16
        rows = -(-len(ordered) // columns)
        canvas = Canvas(gap + columns * (side + gap), 70 + rows * (side + 44 + gap))
        canvas.text(gap, 14, title, 36, bold=True)
        for n, index in enumerate(ordered):
            row, column = divmod(n, columns)
            x, y = gap + column * (side + gap), 70 + row * (side + 44 + gap)
            canvas.tile(self.image(index), x, y, side)
            canvas.text(x, y + side + 6, str(n + 1), 28, bold=True)
        return canvas.array()
