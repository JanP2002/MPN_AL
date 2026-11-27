"""Grid search over the number of tiles per round (``docs/architecture/k-selection-experiment.md``).

    python scripts/run_k_selection.py --tiles-parquet prep/tiles.parquet --descriptors old/descriptors.parquet \
        --embeddings dinov2-s14=.../embeddings.parquet --embeddings lunit-dino-s16=... --out outputs/selection/<id>_smallK

Needs the embeddings and the tile descriptors of the selection experiment, not the images. Written to ``--out``:
``selection_scores.csv``, ``selections.parquet``, ``clustering_metrics.csv``, ``labels_l2.npz`` and
``projections.parquet`` (for the boards), ``descriptors.parquet``, ``manifest.parquet``, ``run.json`` and the analysis
tables of :func:`analyse` (``k_summary.csv``, ``k_choice.csv``, ``model_order.csv``, ...).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import k_analysis as ka
from . import selection as sel
from .cluster_cli import parse_named
from .cluster_figures import project_2d
from .clustering import fit_kmeans, preprocess
from .selection_cli import NOISE, Experiment, aligned_vectors, seed_stability
from .tile_descriptors import standardize

logger = logging.getLogger("mpn_al.k_selection")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles-parquet", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True, help="descriptors.parquet of the selection experiment")
    parser.add_argument("--embeddings", action="append", required=True, type=parse_named, metavar="NAME=FILE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-init", type=int, nargs="+", default=[1, 10], help="the first value is the main configuration")
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument("--spatial-draws", type=int, default=100)
    parser.add_argument("--pca", type=int, default=128, help="PCA variant (0 = skip)")
    parser.add_argument("--subsamples", type=int, default=20, help="random subsets of the tiles (0 = skip)")
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--projection", choices=["auto", "umap", "tsne", "none"], default="auto")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads of KMeans and BLAS; a few thousand tiles run several times slower on many threads")
    return parser


def analyse(out: Path, scores: pd.DataFrame, selected: dict, positions: np.ndarray, size: int, group_counts: np.ndarray,
            ks: Sequence[int]) -> dict[str, pd.DataFrame]:
    """The analysis tables, written as CSV files to ``out`` and returned."""
    gapped = ka.add_gap(ka.comparable(scores))
    summary = ka.summarize_k(gapped, group_counts)
    tables = {
        "scores_gap": gapped,
        "k_summary": summary,
        "k_choice": ka.choose_k(summary),
        "model_order": pd.concat([ka.order_models(gapped, v).assign(variant=v) for v in gapped["variant"].unique() if v != "baseline"],
                                 ignore_index=True),
        "marginal_gains": ka.marginal_gains(summary),
        "nesting": ka.nesting(selected, positions, size),
        "selection_stability": ka.selection_stability(selected, positions, size),
        "representative_seeds": ka.representative_seeds(scores, ks),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    with threadpool_limits(limits=args.threads):
        return run(parser, args)


def run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    started = time.perf_counter()
    out = args.out
    if (out / "blind_key.json").exists():
        parser.error(f"{out} holds the results of the tile-selection experiment: choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(args.ks))
    seeds = list(range(args.seeds))

    manifest = sel.tile_manifest(pd.read_parquet(args.tiles_parquet))
    if manifest["slide_id"].nunique() != 1:
        parser.error("this experiment handles the tiles of one slide")
    descriptors = pd.read_parquet(args.descriptors).set_index("tile_id").loc[manifest["tile_id"]].reset_index()
    spaces = {}
    for name, path in args.embeddings:
        if name is None:
            parser.error("give embeddings as NAME=FILE")
        spaces[name] = aligned_vectors(path, manifest["tile_id"].tolist())
        logger.info("%s: %d x %d", name, *spaces[name].shape)
    models = list(spaces)
    dimension = min(v.shape[1] for v in spaces.values())
    spaces[NOISE] = preprocess(np.random.default_rng(0).normal(size=(len(manifest), dimension)))

    exp = Experiment(args, manifest, descriptors)
    exp.run_baselines(ks, seeds)
    reduced = {name: preprocess(spaces[name], pca_dim=args.pca) for name in models} if args.pca else {}
    for position, n_init in enumerate(args.n_init):
        suffix = "" if position == 0 else f"-ninit{n_init}"
        for name, vectors in spaces.items():
            exp.run_model(f"l2{suffix}", name, vectors, ks, seeds, n_init=n_init)
        for name, vectors in spaces.items():
            exp.run_model(f"lattice{suffix}", name, vectors, ks, seeds, candidates=exp.lattice, n_init=n_init)
        for name, vectors in reduced.items():
            exp.run_model(f"pca{args.pca}{suffix}", name, vectors, ks, seeds, n_init=n_init)

    keep = int(round(args.fraction * len(manifest)))
    for repetition in range(args.subsamples):
        candidates = np.sort(np.random.default_rng(5000 + repetition).choice(len(manifest), size=keep, replace=False))
        conflicts = exp.conflicts[np.ix_(candidates, candidates)]
        for name, vectors in spaces.items():
            sub = vectors[candidates]
            for k in ks:
                model = fit_kmeans(sub, k, 0)
                local, fallbacks = sel.representatives(sub, model.labels_, model.cluster_centers_, conflicts=conflicts)
                exp.record(f"subsample{round(100 * args.fraction)}", name, k, repetition, "no-overlap", candidates[local], fallbacks)
        logger.info("subsample %d done", repetition)

    scores = pd.DataFrame(exp.scores)
    scores.to_csv(out / "selection_scores.csv", index=False)
    picks = pd.DataFrame(exp.picks)
    picks["tile_id"] = manifest["tile_id"].to_numpy()[picks["tile_index"]]
    picks.to_parquet(out / "selections.parquet", index=False)
    pd.DataFrame(exp.metrics).to_csv(out / "clustering_metrics.csv", index=False)
    seed_stability(exp).to_csv(out / "seed_stability.csv", index=False)
    np.savez_compressed(out / "labels_l2.npz", **{f"{m}|{k}|{s}": labels.astype(np.int8)
                                                  for (v, m, k, s), labels in exp.labels.items() if v == "l2"})
    descriptors.to_parquet(out / "descriptors.parquet", index=False)
    manifest.to_parquet(out / "manifest.parquet", index=False)
    group_counts = exp.scorer.group_counts
    analyse(out, scores, exp.selected, exp.positions, exp.size, group_counts, ks)

    if args.projection != "none":
        frames = []
        for name in models:
            coords, method = project_2d(spaces[name], method=args.projection)
            frames.append(pd.DataFrame({"tile_id": manifest["tile_id"], "model": name, "x": coords[:, 0], "y": coords[:, 1], "method": method}))
            logger.info("%s: %s map done", name, method)
        pd.concat(frames, ignore_index=True).to_parquet(out / "projections.parquet", index=False)

    (out / "run.json").write_text(json.dumps(
        {"args": {k: ([str(x) for x in v] if k == "embeddings" else (str(v) if isinstance(v, Path) else v)) for k, v in vars(args).items()},
         "tiles": len(manifest), "models": models, "ks": ks, "group_counts": group_counts.tolist(),
         "seconds": round(time.perf_counter() - started, 1)}, indent=2), encoding="utf-8")
    logger.info("done in %.0f s: %s", time.perf_counter() - started, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
