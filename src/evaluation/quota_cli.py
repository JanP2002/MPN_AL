"""50 tiles from 5-10 clusters: the grid, the baselines, the confirmation tests and the analysis tables.

    python scripts/run_quota_selection.py --tiles-parquet prep/tiles.parquet --descriptors old/descriptors.parquet \
        --embeddings dinov2-s14=.../embeddings.parquet --embeddings lunit-dino-s16=... --out outputs/selection/<id>_quota

Protocol: ``docs/architecture/quota-selection-experiment.md``. Needs embeddings and tile descriptors, not the images.
Every finished stage is saved in ``<out>/parts``; running the same command again continues from there.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from . import quota_analysis as qa
from . import quota_selection as qs
from . import selection as sel
from .cluster_cli import parse_named
from .cluster_figures import project_2d
from .clustering import fit_kmeans, internal_metrics, preprocess
from .selection_cli import aligned_vectors
from .tile_descriptors import standardize

logger = logging.getLogger("mpn_al.quota")
NOISE = qa.NOISE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles-parquet", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--embeddings", action="append", required=True, type=parse_named, metavar="NAME=FILE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10], help="numbers of clusters")
    parser.add_argument("--total", type=int, default=50, help="tiles every selection holds")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--draws", type=int, default=20, help="random draws per seed of the random-within-cluster method")
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument("--spatial-draws", type=int, default=100)
    parser.add_argument("--variant-seeds", type=int, default=10, help="seeds of the fragment and space-variant tests")
    parser.add_argument("--subsamples", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--pca", type=int, default=128)
    parser.add_argument("--variant-n-init", type=int, default=10)
    parser.add_argument("--confirm", type=int, default=3, help="best configurations that go through the confirmation tests")
    parser.add_argument("--check-labels", type=Path, help="labels_l2.npz of an earlier run to compare the clusterings with")
    parser.add_argument("--projection", choices=["auto", "umap", "tsne", "none"], default="auto")
    parser.add_argument("--threads", type=int, default=1, help="threads of KMeans and BLAS (small data run faster on one)")
    return parser


class Study:
    """The slide, the embedding spaces and the lists every stage appends to."""

    def __init__(self, args, manifest: pd.DataFrame, table: pd.DataFrame, spaces: dict[str, np.ndarray]):
        self.args = args
        self.manifest = manifest
        self.table = table
        self.spaces = spaces
        self.D = standardize(table)
        self.positions = manifest[["x", "y"]].to_numpy(dtype=float)
        self.size = int(manifest["width"].iloc[0])
        self.conflicts = sel.overlap_matrix(self.positions, self.size)
        self.groups = KMeans(n_clusters=min(8, len(self.D)), n_init=10, random_state=0).fit_predict(self.D)
        self.everything = np.arange(len(manifest))
        self.scorer = qs.ValueScorer(manifest, table, self.D, self.groups)
        self.reset()

    def reset(self) -> None:
        self.rows: list[dict] = []
        self.picks: list[dict] = []
        self.metrics: list[dict] = []
        self.labels: dict[str, np.ndarray] = {}

    def record(self, scorer: qs.ValueScorer, key: dict, picks: qs.Picks) -> None:
        self.rows.append({**key, **scorer.score(picks.tiles), "fallbacks": picks.fallbacks})
        self.picks.append({**key, "tiles": picks.tiles.tolist(), "cluster": picks.cluster.tolist(), "turn": picks.turn.tolist()})

    # ---- selections ---------------------------------------------------------------------------

    def quota(self, test: str, part: str, model: str, vectors: np.ndarray, candidates: np.ndarray, scorer, k: int, seed: int,
              combos: Sequence[tuple[str, str]], *, n_init: int = 1, keep_labels: bool = False) -> None:
        """One clustering, every requested (allocation, method) on it."""
        sub = vectors[candidates]
        conflicts = self.conflicts[np.ix_(candidates, candidates)]
        model_fit = fit_kmeans(sub, k, seed, n_init=n_init)
        if keep_labels:
            self.labels[f"{model}|{k}|{seed}"] = model_fit.labels_.astype(np.int16)
            self.metrics.append({"model": model, "k": k, "seed": seed, "inertia": float(model_fit.inertia_),
                                 **internal_metrics(sub, model_fit.labels_, sample_size=10000)})
        for allocation, method in combos:
            draws = self.args.draws if method == qs.RANDOM_METHOD else 1
            for draw in range(draws):
                rng = np.random.default_rng((10_000, k, seed, draw, qs.ALLOCATIONS.index(allocation)))
                picks = qs.select(sub, model_fit.labels_, model_fit.cluster_centers_, min(self.args.total, len(sub)),
                                  allocation=allocation, method=method, conflicts=conflicts, rng=rng)
                picks.tiles = candidates[picks.tiles]
                self.record(scorer, {"test": test, "part": part, "model": model, "k": k, "allocation": allocation, "method": method,
                                     "seed": seed, "draw": draw}, picks)

    def one_each(self, test: str, part: str, model: str, vectors: np.ndarray, candidates: np.ndarray, scorer, seed: int,
                 *, n_init: int = 1, keep_labels: bool = False, metric: str = "cosine") -> None:
        """The earlier method: as many clusters as tiles, the tile nearest each centre."""
        sub = vectors[candidates]
        conflicts = self.conflicts[np.ix_(candidates, candidates)]
        total = min(self.args.total, len(sub) - 1)
        model_fit = fit_kmeans(sub, total, seed, n_init=n_init)
        if keep_labels:
            self.labels[f"{model}|{total}|{seed}"] = model_fit.labels_.astype(np.int16)
            self.metrics.append({"model": model, "k": total, "seed": seed, "inertia": float(model_fit.inertia_),
                                 **internal_metrics(sub, model_fit.labels_, sample_size=10000)})
        local, fallbacks = sel.representatives(sub, model_fit.labels_, model_fit.cluster_centers_, conflicts=conflicts, metric=metric)
        order = qs.cluster_order(model_fit.labels_)[: len(local)]
        number = {int(c): n for n, c in enumerate(order)}
        picks = qs.Picks(candidates[local], np.array([number[int(c)] for c in order]), np.zeros(len(local), dtype=int), fallbacks)
        self.record(scorer, {"test": test, "part": part, "model": model, "k": self.args.total, "allocation": "one", "method": "one",
                             "seed": seed, "draw": 0}, picks)

    def random(self, test: str, part: str, candidates: np.ndarray, scorer, draws: int, seed: int, *, spatial: bool = False) -> None:
        name = "spatial" if spatial else qa.RANDOM
        conflicts = self.conflicts[np.ix_(candidates, candidates)]
        rng = np.random.default_rng(seed)
        total = min(self.args.total, len(candidates))
        for draw in range(draws):
            local = (sel.spatial_selection(self.positions[candidates], total, rng, conflicts) if spatial
                     else sel.random_selection(len(candidates), total, rng, conflicts))
            picks = qs.Picks(candidates[local], np.full(len(local), -1), np.zeros(len(local), dtype=int), 0)
            self.record(scorer, {"test": test, "part": part, "model": name, "k": 0, "allocation": "-", "method": name,
                                 "seed": draw, "draw": 0}, picks)

    # ---- stages -------------------------------------------------------------------------------

    def baselines(self) -> None:
        total = self.args.total
        self.random("grid", "-", self.everything, self.scorer, self.args.random_draws, 1000 + total)
        self.random("grid", "-", self.everything, self.scorer, self.args.spatial_draws, 2000 + total, spatial=True)
        for seed in range(self.args.seeds):
            self.one_each("grid", "-", "descriptor-kmeans", self.D, self.everything, self.scorer, seed, metric="euclidean")

    def grid(self, model: str) -> None:
        vectors = self.spaces[model]
        combos = [(a, m) for a in qs.ALLOCATIONS for m in qs.METHODS]
        for seed in range(self.args.seeds):
            self.one_each("grid", "-", model, vectors, self.everything, self.scorer, seed, keep_labels=True)
            for k in self.args.ks:
                self.quota("grid", "-", model, vectors, self.everything, self.scorer, k, seed, combos, keep_labels=True)
        picks = qs.one_cluster(vectors, self.args.total, self.conflicts)
        self.record(self.scorer, {"test": "grid", "part": "-", "model": model, "k": 1, "allocation": "-", "method": "no-clusters",
                                  "seed": 0, "draw": 0}, picks)

    def confirmation(self, winners: pd.DataFrame) -> None:
        """New seeds, the two largest tissue fragments, subsamples and space variants for the winners and their references."""
        args = self.args
        configs = [tuple(w[c] for c in qa.CONFIG) for w in winners.to_dict("records")]
        ones = sorted({qa.RECOMMENDED_MODEL} | {c[0] for c in configs})
        regions = self.manifest["region"].to_numpy()
        lattice = np.flatnonzero(self.manifest["lattice"].to_numpy())
        reduced = {m: preprocess(self.spaces[m], pca_dim=args.pca) for m in ones} if args.pca else {}

        def run(test: str, part: str, candidates: np.ndarray, scorer, seeds: Sequence[int], spaces: dict, n_init: int = 1,
                fit_seed: Callable[[int], int] = lambda s: s) -> None:
            for seed in seeds:
                for model, k, allocation, method in configs:
                    self._quota_as(test, part, model, spaces[model], candidates, scorer, k, fit_seed(seed), seed, allocation, method, n_init)
                for model in ones:
                    before = len(self.rows)
                    self.one_each(test, part, model, spaces[model], candidates, scorer, fit_seed(seed), n_init=n_init)
                    for table in (self.rows, self.picks):
                        for row in table[before:]:
                            row["seed"] = seed

        fresh = range(args.seeds, 2 * args.seeds)
        run("confirm_seeds", "-", self.everything, self.scorer, fresh, self.spaces)
        self.random("confirm_seeds", "-", self.everything, self.scorer, args.random_draws, 7000)
        for number in range(2):
            candidates = np.flatnonzero(regions == number)
            if len(candidates) <= args.total:
                logger.warning("tissue fragment %d has only %d tiles: skipped", number, len(candidates))
                continue
            part = f"fragment {number + 1} ({len(candidates)} kafli)"
            scorer = qs.ValueScorer(self.manifest, self.table, self.D, self.groups, tiles=candidates)
            run("fragment", part, candidates, scorer, range(args.variant_seeds), self.spaces)
            self.random("fragment", part, candidates, scorer, 200, 7100 + number)
        keep = int(round(args.fraction * len(self.manifest)))
        for repetition in range(args.subsamples):
            candidates = np.sort(np.random.default_rng(5000 + repetition).choice(len(self.manifest), size=keep, replace=False))
            run("subsample", "-", candidates, self.scorer, [repetition], self.spaces, fit_seed=lambda s: 0)
            before = len(self.rows)
            self.random("subsample", "-", candidates, self.scorer, 20, 7200 + repetition)
            for table in (self.rows, self.picks):
                for row in table[before:]:
                    row["seed"] += 20 * repetition
        if reduced:
            run("variant", f"PCA {args.pca}", self.everything, self.scorer, range(args.variant_seeds), reduced)
            self.random("variant", f"PCA {args.pca}", self.everything, self.scorer, 200, 7300)
        run("variant", f"n_init = {args.variant_n_init}", self.everything, self.scorer, range(args.variant_seeds), self.spaces,
            n_init=args.variant_n_init)
        self.random("variant", f"n_init = {args.variant_n_init}", self.everything, self.scorer, 200, 7300)
        part = f"{len(lattice)} kafli bez nakładania"
        run("variant", part, lattice, self.scorer, range(args.variant_seeds), self.spaces)
        self.random("variant", part, lattice, self.scorer, 200, 7400)

    def _quota_as(self, test, part, model, vectors, candidates, scorer, k, fit_seed, seed, allocation, method, n_init) -> None:
        before = len(self.rows)
        self.quota(test, part, model, vectors, candidates, scorer, k, fit_seed, [(allocation, method)], n_init=n_init)
        for table in (self.rows, self.picks):
            for row in table[before:]:
                row["seed"] = seed

    def stage(self, folder: Path, name: str, signature: str, compute: Callable[[], None]) -> dict:
        """Runs ``compute`` unless its result is already saved with the same settings."""
        path = folder / f"{name}.pkl"
        if path.exists():
            saved = pd.read_pickle(path)
            if saved["signature"] != signature:
                raise ValueError(f"{path} was written with other settings: use another --out or delete the parts folder")
            logger.info("stage %s: loaded", name)
            return saved
        self.reset()
        started = time.perf_counter()
        compute()
        result = {"signature": signature, "rows": self.rows, "picks": self.picks, "metrics": self.metrics, "labels": self.labels}
        folder.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(result, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        logger.info("stage %s: %d selections in %.0f s", name, len(self.rows), time.perf_counter() - started)
        return result


def analyse(out: Path, study: Study, scores: pd.DataFrame, selections: pd.DataFrame, winners: pd.DataFrame) -> None:
    """Every table the reports read."""
    summary = qa.summarize(scores)
    summary.to_csv(out / "summary.csv", index=False)
    winners.to_csv(out / "winners.csv", index=False)
    configs = [tuple(w[c] for c in qa.CONFIG) for w in winners.to_dict("records")]
    total = study.args.total
    compared = list(dict.fromkeys(configs + [r for w in winners.to_dict("records") for r in qa.references(w, total).values()]))
    removal = qa.feature_removal(selections, study.D, compared)
    removal.to_csv(out / "feature_removal.csv", index=False)
    confirmation = qa.confirm(winners, summary, removal, total)
    confirmation.to_csv(out / "confirmation.csv", index=False)
    verdicts = qa.verdicts(confirmation)
    verdicts.to_csv(out / "verdicts.csv", index=False)
    qa.factor_effects(summary).to_csv(out / "factor_effects.csv", index=False)
    qa.stability(selections, study.positions, study.size, seeds=study.args.seeds).to_csv(out / "stability.csv", index=False)

    first = winners.iloc[0]
    sets = {"zwycięzca siatki": configs[0], **qa.references(first, total)}
    if first["model"] == qa.RECOMMENDED_MODEL:
        del sets[qa.REF_SAME]  # the same configuration as the recommendation
    groups, features = qa.explain_sets(selections, study.D, study.groups, sets)
    groups.to_csv(out / "tissue_groups.csv", index=False)
    features.to_csv(out / "feature_coverage.csv", index=False)
    qa.describe_groups(study.table, study.D, study.groups).to_csv(out / "group_descriptions.csv", index=False)

    chosen = []
    grid = scores[scores["test"] == "grid"]
    for name, config in sets.items():
        part = grid
        for column, value in zip(qa.CONFIG, config):
            part = part[part[column] == value]
        row = qa.median_selection(part)
        chosen.append({"set": name, **dict(zip(qa.CONFIG, config)), "seed": int(row["seed"]), "draw": int(row["draw"]),
                       "coverage_mean": float(row["coverage_mean"])})
    pd.DataFrame(chosen).to_csv(out / "representative.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    with threadpool_limits(limits=args.threads):
        return run(parser, args)


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    started = time.perf_counter()
    out = args.out
    if (out / "blind_key.json").exists() or (out / "k_summary.csv").exists():
        parser.error(f"{out} holds the results of another experiment: choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    args.ks = sorted(set(args.ks))
    manifest = sel.tile_manifest(pd.read_parquet(args.tiles_parquet))
    if manifest["slide_id"].nunique() != 1:
        parser.error("this experiment handles the tiles of one slide")
    table = pd.read_parquet(args.descriptors).set_index("tile_id").loc[manifest["tile_id"]].reset_index()
    if qa.RECOMMENDED_MODEL not in [name for name, _ in args.embeddings]:
        parser.error(f"the embeddings of {qa.RECOMMENDED_MODEL} are needed: it is the reference of the confirmation tests")
    spaces = {}
    for name, path in args.embeddings:
        if name is None:
            parser.error("give embeddings as NAME=FILE")
        spaces[name] = aligned_vectors(path, manifest["tile_id"].tolist())
        logger.info("%s: %d x %d", name, *spaces[name].shape)
    models = list(spaces)
    dimension = min(v.shape[1] for v in spaces.values())
    spaces[NOISE] = preprocess(np.random.default_rng(0).normal(size=(len(manifest), dimension)))
    study = Study(args, manifest, table, spaces)

    signature = json.dumps({k: v for k, v in vars(args).items() if k not in ("out", "embeddings", "tiles_parquet", "descriptors",
                                                                             "check_labels", "projection", "threads")}, default=str, sort_keys=True)
    signature += f"|{len(manifest)}|{','.join(models)}"
    parts = out / "parts"
    stages = [study.stage(parts, "baselines", signature, study.baselines)]
    stages += [study.stage(parts, f"grid_{model}", signature, lambda m=model: study.grid(m)) for model in spaces]
    grid_scores = pd.DataFrame([r for s in stages for r in s["rows"]])
    winners, limit = qa.apply_rule(qa.summarize(grid_scores), args.confirm)
    logger.info("rule: %s", winners[[*qa.CONFIG, "coverage_mean", "tied"]].to_dict("records"))
    key = signature + "|" + ";".join(qa.key_of(*[w[c] for c in qa.CONFIG]) for w in winners.to_dict("records"))
    stages.append(study.stage(parts, "confirmation", key, lambda: study.confirmation(winners)))

    scores = pd.DataFrame([r for s in stages for r in s["rows"]])
    selections = pd.DataFrame([r for s in stages for r in s["picks"]])
    labels = {k: v for s in stages for k, v in s["labels"].items()}
    scores.to_csv(out / "selection_scores.csv", index=False)
    selections.to_parquet(out / "selections.parquet", index=False)
    pd.DataFrame([r for s in stages for r in s["metrics"]]).to_csv(out / "clustering_metrics.csv", index=False)
    np.savez_compressed(out / "labels.npz", **labels)
    table.to_parquet(out / "descriptors.parquet", index=False)
    manifest.assign(group=study.groups).to_parquet(out / "manifest.parquet", index=False)
    analyse(out, study, scores, selections, winners)

    checked = None
    if args.check_labels:
        earlier = np.load(args.check_labels)
        common = sorted(set(earlier.files) & set(labels))
        checked = {"compared": len(common), "identical": int(sum(np.array_equal(earlier[k].astype(int), labels[k].astype(int)) for k in common))}
        logger.info("clusterings identical to the earlier run: %(identical)d of %(compared)d", checked)

    if args.projection != "none":
        frames = []
        for name in models:
            coords, method = project_2d(spaces[name], method=args.projection)
            frames.append(pd.DataFrame({"tile_id": manifest["tile_id"], "model": name, "x": coords[:, 0], "y": coords[:, 1], "method": method}))
            logger.info("%s: %s map done", name, method)
        pd.concat(frames, ignore_index=True).to_parquet(out / "projections.parquet", index=False)

    (out / "run.json").write_text(json.dumps(
        {"args": {k: ([str(x) for x in v] if k == "embeddings" else (str(v) if isinstance(v, Path) else v)) for k, v in vars(args).items()},
         "tiles": len(manifest), "models": models, "low_tissue_limit": limit, "labels_check": checked,
         "seconds": round(time.perf_counter() - started, 1)}, indent=2), encoding="utf-8")
    logger.info("done in %.0f s: %s", time.perf_counter() - started, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
