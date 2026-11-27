"""Choosing K: the gap to the ceiling, the tie rules, representative seeds and the grid search on a synthetic slide."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mpn_al.evaluation import k_analysis as ka
from mpn_al.evaluation import selection as sel
from mpn_al.evaluation.k_cli import main
from mpn_al.evaluation.tile_descriptors import describe_tile
from test_selection import SCRIPTS, grid, synthetic_tile, table_for


def score_rows(variant, method, k, values, constraint="no-overlap"):
    return [{"variant": variant, "method": method, "k": k, "seed": seed, "constraint": constraint, "coverage_mean": float(v),
             **{c: 0.0 for c in ka.SCORE_COLUMNS if c != "coverage_mean"}} for seed, v in enumerate(values)]


def baselines(ks=(5, 6)):
    rows = []
    for k in ks:
        rows += score_rows("baseline", "random", k, [2.8, 3.0, 3.2]) + score_rows("baseline", "descriptor-kmeans", k, [2.0, 2.0])
    return rows


def test_gap_is_zero_for_random_one_for_the_ceiling_and_counts_beaten_draws():
    scores = pd.DataFrame(baselines((5,)) + score_rows("l2", "m", 5, [2.5, 3.0, 4.0]))
    gapped = ka.add_gap(scores)
    mean = gapped.groupby("method")["gap"].mean()
    assert mean["random"] == pytest.approx(0) and mean["descriptor-kmeans"] == pytest.approx(1)
    model = gapped[gapped["method"] == "m"]
    assert model["gap"].tolist() == pytest.approx([0.5, 0.0, -1.0])
    assert model["beats_random"].tolist() == pytest.approx([1.0, 0.5, 0.0])  # 3.0 beats one draw and ties one
    with pytest.raises(ValueError, match="for K = \\[7\\]"):
        ka.add_gap(pd.DataFrame(baselines((5,)) + score_rows("l2", "m", 7, [2.0])))


def test_comparable_rows_keep_free_selections_of_the_main_variant_only():
    scores = pd.DataFrame(score_rows("l2", "m", 5, [1.0]) + score_rows("l2", "m", 5, [2.0], "free")
                          + score_rows("pca128", "m", 5, [3.0], "free"))
    out = ka.comparable(scores)
    assert sorted(out["variant"]) == ["free", "l2"] and out.set_index("variant").loc["free", "coverage_mean"] == 2.0


def summary_for(gaps, sd=0.05, variant="l2", method="m"):
    return pd.DataFrame([{"variant": variant, "method": method, "k": k, "gap": g, "gap_sd": sd} for k, g in gaps.items()])


def test_the_smallest_k_within_the_spread_is_recommended():
    choice = ka.choose_k(summary_for({5: 0.30, 6: 0.47, 7: 0.40, 8: 0.50, 9: 0.44, 10: 0.20})).iloc[0]
    assert choice["best_k"] == 8 and choice["tied_k"] == "6 8" and choice["recommended_k"] == 6 and not choice["all_tied"]
    flat = ka.choose_k(summary_for({5: 0.40, 6: 0.41, 7: 0.42}, sd=0.1)).iloc[0]
    assert flat["all_tied"] and flat["recommended_k"] == 5
    exact = ka.choose_k(summary_for({5: 0.4, 6: 0.5}, sd=0.0)).iloc[0]  # no spread at all: only the best is tied with itself
    assert exact["tied_k"] == "6"


def test_models_are_ordered_over_all_k_and_ties_go_to_the_cheaper_one():
    rows = []
    for method, centre in (("midnight-12k", 0.50), ("lunit-dino-s16", 0.49), ("phikon-v2", 0.1), ("noise-control", 0.9)):
        for k in (5, 6):
            rows += [{"variant": "l2", "method": method, "k": k, "seed": s, "gap": centre + 0.05 * (s - 1)} for s in range(3)]
    order = ka.order_models(pd.DataFrame(rows))
    assert order["method"].tolist() == ["lunit-dino-s16", "midnight-12k", "phikon-v2"]
    assert order.loc[0, "tied_with"] == "midnight-12k"


def test_representative_seed_is_the_lower_median():
    assert ka.median_item(pd.Series({0: 3.0, 1: 1.0, 2: 2.0})) == 2
    assert ka.median_item(pd.Series({0: 4.0, 1: 1.0, 2: 2.0, 3: 3.0})) == 2  # lower of the two middle values
    scores = pd.DataFrame(score_rows("l2", "m", 5, [3, 1, 2]) + score_rows("l2", "m", 6, [1, 5, 2]))
    seeds = ka.representative_seeds(scores, [5, 6]).set_index("k")["seed"]
    assert seeds[5] == 2 and seeds[6] == 2  # coverage 3, 1, 2 and 1, 5, 2
    assert seeds[0] == 2  # means over K are 2, 3, 2: sorted (2, seed 0), (2, seed 2), (3, seed 1), the middle one is seed 2


def test_smallest_divergence_and_marginal_gains():
    assert ka.min_jsd([1, 1], 2) == pytest.approx(0)
    assert ka.min_jsd([10, 10, 10, 10], 2) > ka.min_jsd([10, 10, 10, 10], 3) > ka.min_jsd([10, 10, 10, 10], 4) == pytest.approx(0)
    summary = summary_for({5: 0.3, 6: 0.4}).assign(coverage_mean=[3.0, 2.5])
    gains = ka.marginal_gains(summary).iloc[0]
    assert gains["coverage_drop"] == pytest.approx(0.5) and gains["gap_change"] == pytest.approx(0.1)


def test_nesting_and_stability_count_overlapping_tiles():
    positions = np.array([[0, 0], [50, 0], [400, 0], [800, 0]], float)
    selected = {("l2", "m", 2, 0, "no-overlap"): np.array([0, 2]), ("l2", "m", 3, 0, "no-overlap"): np.array([1, 2, 3]),
                ("l2", "m", 2, 1, "no-overlap"): np.array([1, 3])}
    nest = ka.nesting(selected, positions, 100).iloc[0]
    assert nest["identical"] == 0.5 and nest["identical_or_overlapping"] == 1.0
    stable = ka.selection_stability(selected, positions, 100).iloc[0]
    assert stable["identical"] == 0 and stable["identical_or_overlapping"] == 0.5


def test_the_grid_search_on_a_synthetic_slide(tmp_path):
    rng = np.random.default_rng(0)
    positions = grid(10, 10, size=256, stride=128) + grid(4, 4, size=256, stride=128, x0=4096)
    table = table_for(positions, size=256)
    table["mpp"] = 0.44
    table.to_parquet(tmp_path / "tiles.parquet")
    kinds = ["fat", "cells", "bone", "red"]
    truth = [int((r.x // 640 + r.y // 640) % 4) for r in table.itertuples()]
    descriptors = pd.DataFrame([{"tile_id": t, **describe_tile(synthetic_tile(kinds[k], rng), mpp=0.44)} for t, k in zip(table["tile_id"], truth)])
    descriptors.to_parquet(tmp_path / "descriptors.parquet")
    arguments = ["--tiles-parquet", str(tmp_path / "tiles.parquet"), "--descriptors", str(tmp_path / "descriptors.parquet")]
    for name, vectors in (("dinov2-s14", rng.normal(size=(len(truth), 16))), ("phikon-v2", np.eye(16)[truth] * 3 + 0.2 * rng.normal(size=(len(truth), 16)))):
        path = tmp_path / f"{name}.parquet"
        pd.DataFrame({"tile_id": table["tile_id"], "base_model": name, "pooling": "cls", "window_mode": "grid",
                      "embedding": [v.astype(np.float32) for v in vectors]}).to_parquet(path)
        arguments += ["--embeddings", f"{name}={path}"]
    sys.path.insert(0, str(SCRIPTS))
    import run_k_selection

    assert run_k_selection.main is main
    out = tmp_path / "out"
    options = ["--ks", "4", "5", "--seeds", "4", "--n-init", "1", "3", "--random-draws", "60", "--spatial-draws", "5", "--pca", "8",
               "--subsamples", "3", "--projection", "tsne"]
    assert main([*arguments, "--out", str(out), *options]) == 0

    scores = pd.read_csv(out / "selection_scores.csv")
    assert set(scores["variant"]) == {"l2", "lattice", "pca8", "l2-ninit3", "lattice-ninit3", "pca8-ninit3", "subsample80", "baseline"}
    main_rows = scores[(scores["variant"] == "l2") & (scores["constraint"] == "no-overlap")]
    assert len(main_rows) == 3 * 2 * 4  # models with the noise control, K, seeds
    assert set(scores.query("variant == 'pca8'")["method"]) == {"dinov2-s14", "phikon-v2"}
    choice = pd.read_csv(out / "k_choice.csv")
    assert set(choice["variant"]) >= {"l2", "free", "subsample80"} and set(choice["recommended_k"]) <= {4, 5}
    order = pd.read_csv(out / "model_order.csv")
    assert order.query("variant == 'l2'").iloc[0]["method"] == "phikon-v2"  # the space that follows the tissue
    summary = pd.read_csv(out / "k_summary.csv").set_index(["variant", "method", "k"])
    assert summary.loc[("baseline", "random", 4), "gap"] == pytest.approx(0, abs=1e-9)
    assert (summary["jsd_excess"] >= -1e-9).all()
    seeds = pd.read_csv(out / "representative_seeds.csv")
    assert set(seeds["k"]) == {0, 4, 5} and {"random", "phikon-v2"} <= set(seeds["method"])
    labels = np.load(out / "labels_l2.npz")
    assert labels["phikon-v2|4|0"].shape == (len(truth),)
    maps = pd.read_parquet(out / "projections.parquet")
    assert set(maps["model"]) == {"dinov2-s14", "phikon-v2"} and len(maps) == 2 * len(truth)
    for name in ("marginal_gains.csv", "nesting.csv", "selection_stability.csv", "seed_stability.csv", "clustering_metrics.csv", "run.json"):
        assert (out / name).exists(), name

    (tmp_path / "old").mkdir()  # the earlier experiment's folder is protected
    (tmp_path / "old" / "blind_key.json").write_text("{}")
    with pytest.raises(SystemExit):
        main([*arguments, "--out", str(tmp_path / "old"), *options])
