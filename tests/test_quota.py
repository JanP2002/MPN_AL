"""Several tiles per cluster: quotas, the ways of picking, the frozen rule and the whole experiment on a synthetic slide."""

import json
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from mpn_al.evaluation import quota_analysis as qa
from mpn_al.evaluation import quota_plates, quota_report
from mpn_al.evaluation import quota_selection as qs
from mpn_al.evaluation import selection as sel
from mpn_al.evaluation.quota_cli import main
from mpn_al.evaluation.selection_cli import aligned_vectors
from mpn_al.evaluation.tile_descriptors import describe_tile
from test_selection import SCRIPTS, grid, synthetic_tile, table_for


def test_quotas_sum_to_the_total():
    sizes = [700, 600, 500, 400, 300, 168]
    assert qs.quotas(sizes, 50, "equal").tolist() == [9, 9, 8, 8, 8, 8]
    assert qs.quotas([1] * 7, 50, "equal").tolist() == [8, 7, 7, 7, 7, 7, 7]
    proportional = qs.quotas(sizes, 50, "proportional")
    assert proportional.sum() == 50 and (np.diff(proportional) <= 0).all()
    assert np.abs(proportional - 50 * np.array(sizes) / sum(sizes)).max() < 1
    tiny = qs.quotas([1000, 5, 5, 5], 10, "proportional")
    assert tiny.tolist() == [7, 1, 1, 1]  # every cluster keeps one place
    with pytest.raises(ValueError):
        qs.quotas([1, 1, 1], 2, "equal")
    with pytest.raises(ValueError, match="unknown allocation"):
        qs.quotas([1, 1], 4, "fair")


def two_clusters():
    """Cluster 0: a dense core, a ring further out and one outlier; cluster 1: a small blob. No tiles overlap."""
    rng = np.random.default_rng(0)
    core = np.array([1.0, 0, 0]) + 0.01 * rng.normal(size=(30, 3))
    ring = np.array([1.0, 0.5, 0]) + 0.01 * rng.normal(size=(10, 3))
    outlier = np.array([[0.2, -1.0, 0]])
    other = np.array([0.0, 0, 1.0]) + 0.01 * rng.normal(size=(9, 3))
    vectors = np.concatenate([core, ring, outlier, other])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array([0] * 41 + [1] * 9)
    centres = np.stack([vectors[labels == c].mean(axis=0) for c in (0, 1)])
    return vectors, labels, centres, np.eye(len(vectors), dtype=bool)


def test_the_methods_differ_in_what_they_take_after_the_first_tile():
    vectors, labels, centres, free = two_clusters()
    unit_centre = centres[0] / np.linalg.norm(centres[0])
    central = int(np.argmax(vectors[:41] @ unit_centre))
    near = qs.select(vectors, labels, centres, 6, allocation="equal", method="m3", conflicts=free)
    assert near.tiles[:3].tolist() == np.argsort(-(vectors[:41] @ unit_centre), kind="stable")[:3].tolist()
    assert near.cluster.tolist() == [0, 0, 0, 1, 1, 1] and near.turn.tolist() == [0, 1, 2, 0, 1, 2]
    diverse = qs.select(vectors, labels, centres, 6, allocation="equal", method="m1", conflicts=free)
    assert diverse.tiles[0] == central and diverse.tiles[1] == 40  # the outlier is the most different from the centre tile
    assert 30 <= diverse.tiles[2] < 40  # then something from the ring, far from both
    trimmed = qs.select(vectors, labels, centres, 6, allocation="equal", method="m1p", conflicts=free)
    assert trimmed.tiles[0] == central and 40 not in trimmed.tiles  # the outlier is among the 5% farthest
    a = qs.select(vectors, labels, centres, 6, allocation="equal", method="m2", conflicts=free, rng=np.random.default_rng(1))
    b = qs.select(vectors, labels, centres, 6, allocation="equal", method="m2", conflicts=free, rng=np.random.default_rng(2))
    assert set(a.tiles) != set(b.tiles) and (labels[a.tiles] == np.repeat([0, 1], 3)).all()
    with pytest.raises(ValueError, match="generator"):
        qs.select(vectors, labels, centres, 6, allocation="equal", method="m2", conflicts=free)


def test_selections_hold_the_total_without_overlap_even_when_a_cluster_runs_out():
    positions = np.array(grid(12, 12), float)  # half-overlapping tiles of side 100
    conflicts = sel.overlap_matrix(positions, 100)
    rng = np.random.default_rng(0)
    labels = np.zeros(144, dtype=int)
    labels[:3] = 1  # three neighbouring tiles form a cluster that can give only one or two of them
    vectors = rng.normal(size=(144, 5))
    centres = np.stack([vectors[labels == c].mean(axis=0) for c in (0, 1)])
    for method in qs.METHODS:
        picks = qs.select(vectors, labels, centres, 20, allocation="equal", method=method, conflicts=conflicts, rng=rng)
        assert len(picks) == 20 and len(set(picks.tiles.tolist())) == 20
        assert conflicts[np.ix_(picks.tiles, picks.tiles)].sum() == 20
        assert picks.fallbacks >= 7  # the small cluster was promised 10
        assert (labels[picks.tiles] == np.array([0, 1])[picks.cluster]).all()
    everything = qs.select(vectors, labels, centres, 100, allocation="equal", method="m3", conflicts=conflicts)
    assert len(everything) < 100 and conflicts[np.ix_(everything.tiles, everything.tiles)].sum() == len(everything)
    single = qs.one_cluster(vectors, 20, conflicts)
    assert len(single) == 20 and (single.cluster == 0).all()


def row(model, k, allocation, method, coverage, sd=0.01, low=3.0, rare=2.0, jsd=0.02, test="grid", part="-"):
    return {"test": test, "part": part, "model": model, "k": k, "allocation": allocation, "method": method, "coverage_mean": coverage,
            "coverage_mean_sd": sd, "low_tissue": low, "rare1_categories": rare, "jsd_groups": jsd}


def test_the_rule_filters_ties_and_breaks_them():
    table = pd.DataFrame([
        row("random", 0, "-", "random", 1.70, low=5.0),
        row("midnight-12k", 5, "equal", "m1", 1.40, low=9.0),  # best coverage, but too many nearly empty tiles
        row("midnight-12k", 6, "equal", "m1", 1.500, rare=3.0),
        row("phikon-v2", 6, "equal", "m1", 1.505, rare=4.0),  # tied with the best, more rare structures
        row("lunit-dino-s16", 7, "equal", "m3", 1.506, rare=4.0, jsd=0.01),  # tied, as rare, closer proportions
        row("dinov2-s14", 8, "equal", "m2", 1.60),
        row("noise-control", 8, "equal", "m2", 1.00),
        row("lunit-dino-s16", 50, "one", "one", 1.00),  # a baseline never competes
    ])
    winners, limit = qa.apply_rule(table, 3)
    assert limit == 5.0
    assert winners["model"].tolist() == ["lunit-dino-s16", "phikon-v2", "midnight-12k"]
    assert winners.iloc[0]["tied"] == 3 and winners.iloc[0]["eligible"] == 4 and winners.iloc[0]["all"] == 5
    with pytest.raises(ValueError, match="low-tissue"):
        qa.apply_rule(table.assign(low_tissue=[0.0] + [1.0] * 7))


def test_a_winner_is_confirmed_only_if_every_test_passes():
    config = ("midnight-12k", 5, "equal", "m1")
    winners = pd.DataFrame([{"place": 1, **dict(zip(qa.CONFIG, config))}])
    sets = {config: 1.4, ("lunit-dino-s16", 8, "one", "one"): 1.5, ("midnight-12k", 8, "one", "one"): 1.3, ("random", 0, "-", "random"): 1.7}
    rows, removal = [], []
    for test, part in (("confirm_seeds", "-"), ("fragment", "A"), ("fragment", "B"), ("subsample", "-"), ("variant", "PCA")):
        rows += [row(*c, coverage=v, test=test, part=part) for c, v in sets.items()]
    removal += [{**dict(zip(qa.CONFIG, c)), "removed": g, "coverage_mean": v} for g in qa.FEATURE_GROUPS for c, v in sets.items()]
    table = pd.DataFrame(rows)
    confirmation = qa.confirm(winners, table, pd.DataFrame(removal), total=8)
    assert len(confirmation) == 3 * (1 + 2 + 6 + 1 + 1)
    verdict = qa.verdicts(confirmation).iloc[0]
    assert verdict["confirmed"]  # losing to the same model with one tile per cluster does not matter for the verdict
    assert verdict[f"failed: {qa.REF_SAME}"] != ""
    worse = table.copy()
    worse.loc[(worse["part"] == "B") & (worse["model"] == "midnight-12k") & (worse["k"] == 5), "coverage_mean"] = 1.55
    verdict = qa.verdicts(qa.confirm(winners, worse, pd.DataFrame(removal), total=8)).iloc[0]
    assert not verdict["confirmed"] and verdict[f"failed: {qa.REF_RECOMMENDED}"] == "fragmenty tkanki"
    close = table.copy()  # a margin smaller than the spread fails the test with new seeds
    close.loc[close["test"] == "confirm_seeds", "coverage_mean_sd"] = 0.5
    assert "nowe seedy" in qa.verdicts(qa.confirm(winners, close, pd.DataFrame(removal), total=8)).iloc[0][f"failed: {qa.REF_RANDOM}"]


def test_median_selection_is_not_hand_picked():
    scores = pd.DataFrame({"coverage_mean": [4.0, 1.0, 3.0, 2.0], "seed": [0, 1, 2, 3], "draw": 0})
    assert qa.median_selection(scores)["seed"] == 3


def synthetic_study(tmp_path, columns=14):
    rng = np.random.default_rng(0)
    positions = grid(columns, columns, size=256, stride=128) + grid(8, 8, size=256, stride=128, x0=8192)
    table = table_for(positions, size=256)
    table["mpp"] = 0.44
    table.to_parquet(tmp_path / "tiles.parquet")
    kinds = ["fat", "cells", "bone", "red"]
    truth = [int((r.x // 640 + r.y // 640) % 4) for r in table.itertuples()]
    images = {t: synthetic_tile(kinds[k], rng) for t, k in zip(table["tile_id"], truth)}
    pd.DataFrame([{"tile_id": t, **describe_tile(image, mpp=0.44)} for t, image in images.items()]).to_parquet(tmp_path / "d.parquet")
    arguments = ["--tiles-parquet", str(tmp_path / "tiles.parquet"), "--descriptors", str(tmp_path / "d.parquet")]
    files = {}
    for name, vectors in (("lunit-dino-s16", np.eye(16)[truth] + rng.normal(size=(len(truth), 16))),
                          ("phikon-v2", np.eye(16)[truth] * 3 + 0.3 * rng.normal(size=(len(truth), 16)))):
        files[name] = tmp_path / f"{name}.parquet"
        pd.DataFrame({"tile_id": table["tile_id"], "base_model": name, "pooling": "cls", "window_mode": "grid",
                      "embedding": [v.astype(np.float32) for v in vectors]}).to_parquet(files[name])
        arguments += ["--embeddings", f"{name}={files[name]}"]
    options = ["--ks", "2", "3", "--total", "8", "--seeds", "3", "--draws", "2", "--random-draws", "30", "--spatial-draws", "4",
               "--variant-seeds", "2", "--subsamples", "2", "--pca", "8", "--variant-n-init", "2", "--projection", "tsne"]
    return arguments, options, images, files


def test_the_whole_experiment_and_resuming_it(tmp_path):
    arguments, options, _, _ = synthetic_study(tmp_path)
    sys.path.insert(0, str(SCRIPTS))
    import run_quota_selection

    assert run_quota_selection.main is main
    out = tmp_path / "out"
    assert main([*arguments, "--out", str(out), *options]) == 0
    scores = pd.read_csv(out / "selection_scores.csv")
    grid_rows = scores[scores["test"] == "grid"]
    # 3 spaces x 2 K x 3 seeds x 2 allocations x (3 methods + 2 random draws)
    assert len(grid_rows[grid_rows["method"].isin(qs.METHODS)]) == 3 * 2 * 3 * 2 * 5
    assert (scores["n_selected"] == 8).all() and (scores["overlapping_pairs"] == 0).all()
    assert set(scores["test"]) == {"grid", "confirm_seeds", "fragment", "subsample", "variant"}
    assert set(scores.loc[scores["test"] == "variant", "part"]) == {"PCA 8", "n_init = 2", f"{pd.read_parquet(out / 'manifest.parquet')['lattice'].sum()} kafli bez nakładania"}
    assert set(scores.loc[scores["test"] == "confirm_seeds", "seed"]) >= {3, 4, 5}
    winners = pd.read_csv(out / "winners.csv")
    assert len(winners) == 3 and winners["method"].isin(qs.METHODS).all() and (winners["model"] != "noise-control").all()
    confirmation = pd.read_csv(out / "confirmation.csv")
    assert set(confirmation["test"]) == set(qa.TESTS) and set(confirmation["place"]) == {1, 2, 3}
    for name in ("summary.csv", "verdicts.csv", "factor_effects.csv", "stability.csv", "tissue_groups.csv", "feature_coverage.csv",
                 "group_descriptions.csv", "representative.csv", "feature_removal.csv", "clustering_metrics.csv", "labels.npz",
                 "selections.parquet", "projections.parquet"):
        assert (out / name).exists(), name
    labels = np.load(out / "labels.npz")
    assert {"phikon-v2|2|0", "phikon-v2|8|0"} <= set(labels.files)
    representative = pd.read_csv(out / "representative.csv").set_index("set")
    assert {"zwycięzca siatki", qa.REF_RECOMMENDED, qa.REF_RANDOM} <= set(representative.index)

    # the saved stages are reused: removing the tables and running again reproduces them
    first = scores.copy()
    (out / "selection_scores.csv").unlink()
    np.savez(tmp_path / "earlier.npz", **{k: labels[k] for k in labels.files if k.endswith("|2|0")})
    assert main([*arguments, "--out", str(out), *options, "--check-labels", str(tmp_path / "earlier.npz")]) == 0
    pd.testing.assert_frame_equal(pd.read_csv(out / "selection_scores.csv"), first)
    check = json.loads((out / "run.json").read_text())["labels_check"]
    assert check["compared"] == 3 and check["identical"] == 3
    with pytest.raises(ValueError, match="other settings"):
        main([*arguments, "--out", str(out), *options[:-2], "--seeds", "4", "--projection", "tsne"])
    with pytest.raises(SystemExit):  # the recommendation must be among the models
        main([*arguments[:4], *arguments[6:], "--out", str(tmp_path / "x"), *options])


def test_both_reports_from_a_synthetic_run(tmp_path):
    arguments, options, images, files = synthetic_study(tmp_path)
    results = tmp_path / "analysis"
    assert main([*arguments, "--out", str(results), *options]) == 0
    notes = tmp_path / "w.md"
    notes.write_text("## streszczenie\nStreszczenie testowe.\n\n## czynniki\nNajważniejsza jest metoda.\n", encoding="utf-8")
    first = quota_report.build(results, images.__getitem__, tmp_path / "r1", notes=notes, pdf=False)
    html = first.read_text(encoding="utf-8")
    for expected in ("Streszczenie testowe.", "Najważniejsza jest metoda.", "Po co to robimy", "Dane i ich przygotowanie", "Słownik miar",
                     "Dlaczego", "T1.", "T4.", "T5.", "W1.", "W7.", "W10.", "9 okien", "Czego te wyniki nie dowodzą"):
        assert expected in html, expected
    assert not re.search(r">nan\b|\bnan ±|\binf\b", re.sub(r"data:image[^']+", "", html))

    ids = pd.read_parquet(results / "manifest.parquet")["tile_id"].tolist()
    vectors = {name: aligned_vectors(path, ids) for name, path in files.items()}
    second = quota_plates.build(results, images.__getitem__, vectors, tmp_path / "r2", pdf=False)
    html = second.read_text(encoding="utf-8")
    for expected in ("Najlepszy z nowych wariantów", "Dotychczasowa metoda", "Liczebność klastrów", "Gdzie na preparacie", "Czym różnią się metody"):
        assert expected in html, expected
    names = {p.name for p in (tmp_path / "r2" / "plansze").iterdir()}
    assert {"A_zwyciezca_umap.png", "A_zwyciezca_plansza_1.png", "A_zwyciezca_mapa_wsi.png", "A_zwyciezca_klastry.png",
            "B_dotychczasowa_umap.png", "D_losowa_plansza.png"} <= names
    assert any(n.startswith("C_metody_klaster_") for n in names)
    key = json.loads((tmp_path / "r2" / "blind_key_50.json").read_text(encoding="utf-8"))
    assert sorted(key.values()) == ["dotychczasowa", "losowa", "zwycięzca"]
    with zipfile.ZipFile(tmp_path / "r2" / "pakiet_dla_patomorfologa_PQR.zip") as archive:
        assert sorted(archive.namelist()) == ["formularz_oceny.html", "zestaw_P.png", "zestaw_Q.png", "zestaw_R.png"]
        form = re.sub(r"data:image[^']+", "", archive.read("formularz_oceny.html").decode("utf-8"))
    for secret in ("lunit", "phikon", "losow", "dotychczas", "zwyci", "blind_key"):
        assert secret not in form, secret
