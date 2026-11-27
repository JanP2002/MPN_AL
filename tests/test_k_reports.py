"""The two reports of the K selection experiment, and the geometry behind the comparison plates."""

import json
import re
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mpn_al.evaluation import k_report, plates_report
from mpn_al.evaluation.k_cli import main
from mpn_al.evaluation.report_html import num, read_notes, table
from mpn_al.evaluation.selection_cli import aligned_vectors
from mpn_al.evaluation.selection_plates import cluster_order, match_pairs, progression_columns
from mpn_al.evaluation.tile_descriptors import describe_tile
from test_selection import grid, synthetic_tile, table_for


def test_pairs_are_matched_identical_then_overlapping_then_by_features():
    positions = np.array([[0, 0], [50, 0], [400, 0], [800, 0], [1200, 0], [1600, 0]], float)
    D = np.array([[0.0], [0.0], [5.0], [5.1], [9.0], [2.0]])
    pairs = match_pairs([0, 2, 4, 5], [5, 1, 3, 2], positions, 100, D, threshold=0.5)
    assert pairs[:2] == [(2, 2, 0), (5, 5, 0)]
    assert pairs[2] == (0, 1, 1)  # 0 and 1 share area
    assert pairs[3] == (4, 3, 3)  # what is left is far apart in the descriptors
    near = match_pairs([2], [3], positions, 100, D, threshold=0.5)
    assert near == [(2, 3, 2)]


def test_a_tile_that_stays_keeps_its_column():
    positions = np.array([[0, 0], [50, 0], [400, 0], [800, 0], [1200, 0]], float)
    rows = progression_columns([[2, 0], [3, 0, 2], [1, 4, 2, 3]], positions, 100)
    assert rows[0] == {0: (2, 0, 0), 1: (0, 1, 0)}
    assert rows[1][0] == (2, 2, 2) and rows[1][1] == (0, 1, 2) and rows[1][2] == (3, 0, 0)
    assert rows[2][0] == (2, 2, 2) and rows[2][2] == (3, 3, 2) and rows[2][1] == (1, 0, 1)  # 1 overlaps the 0 of the row above
    assert rows[2][3] == (4, 1, 0)
    assert cluster_order(np.array([2, 0, 0, 1, 1, 1])).tolist() == [1, 0, 2]


def test_numbers_tables_and_notes(tmp_path):
    assert num(1.5, 2) == "1,50" and num(-0.5, 1) == "−0,5" and num(-0.0001, 2) == "0,00" and num(float("nan")) == "–"
    out = table(["a", "b"], [["x", "1,5 ± 0,1"], ["y", "–"]], mark_cells=[(0, 1)])
    assert "<th class=n>b</th>" in out and "class='n mark'" in out
    path = tmp_path / "notes.md"
    path.write_text("## answer\nPierwsze **ważne** zdanie\ni dalej.\n\n- punkt <1>\n  ciąg dalszy\n- drugi\n## other\ntekst\n", encoding="utf-8")
    notes = read_notes(path)
    assert notes["answer"] == "<p>Pierwsze <b>ważne</b> zdanie i dalej.</p><ul><li>punkt &lt;1&gt; ciąg dalszy</li><li>drugi</li></ul>"
    assert notes["other"] == "<p>tekst</p>" and read_notes(tmp_path / "none.md") == {}


def test_both_reports_from_a_synthetic_grid_search(tmp_path):
    rng = np.random.default_rng(0)
    positions = grid(10, 10, size=256, stride=128) + grid(4, 4, size=256, stride=128, x0=4096)
    table_ = table_for(positions, size=256)
    table_["mpp"] = 0.44
    table_.to_parquet(tmp_path / "tiles.parquet")
    kinds = ["fat", "cells", "bone", "red"]
    truth = [int((r.x // 640 + r.y // 640) % 4) for r in table_.itertuples()]
    images = {t: synthetic_tile(kinds[k], rng) for t, k in zip(table_["tile_id"], truth)}
    pd.DataFrame([{"tile_id": t, **describe_tile(image, mpp=0.44)} for t, image in images.items()]).to_parquet(tmp_path / "d.parquet")
    arguments = ["--tiles-parquet", str(tmp_path / "tiles.parquet"), "--descriptors", str(tmp_path / "d.parquet")]
    files = {}
    for name, vectors in (("dinov2-s14", np.eye(16)[truth] + rng.normal(size=(len(truth), 16))),
                          ("phikon-v2", np.eye(16)[truth] * 3 + 0.2 * rng.normal(size=(len(truth), 16)))):
        files[name] = tmp_path / f"{name}.parquet"
        pd.DataFrame({"tile_id": table_["tile_id"], "base_model": name, "pooling": "cls", "window_mode": "grid",
                      "embedding": [v.astype(np.float32) for v in vectors]}).to_parquet(files[name])
        arguments += ["--embeddings", f"{name}={files[name]}"]
    results = tmp_path / "analysis"
    assert main([*arguments, "--out", str(results), "--ks", "4", "5", "6", "--seeds", "9", "--n-init", "1", "2", "--random-draws", "50",
                 "--spatial-draws", "5", "--pca", "8", "--subsamples", "3", "--projection", "tsne"]) == 0

    notes = tmp_path / "k.md"
    notes.write_text("## answer\nOdpowiedź testowa.\n", encoding="utf-8")
    first = k_report.build(results, tmp_path / "raport_k", notes=notes, pdf=False)
    html = first.read_text(encoding="utf-8")
    for expected in ("Odpowiedź testowa.", "T1.", "T8.", "T7a.", "W1.", "W6.", "podpróbki 80% kafli"):
        assert expected in html, expected
    assert not re.search(r">nan|nan ±", html)
    assert "PCA 8" in html and "n_init = 2" in html and "podpróbek po 80%" in html
    assert {"w1_curves.png", "w3_distributions.png", "w7_tradeoff.png"} <= {p.name for p in (tmp_path / "raport_k" / "figures").iterdir()}

    vectors = {name: aligned_vectors(path, sorted(images)) for name, path in files.items()}
    second = plates_report.build(results, images.__getitem__, vectors, tmp_path / "wiz", pdf=False)
    html = second.read_text(encoding="utf-8")
    for expected in ("V1.", "V2.", "P1.", "P2.", "P3.", "P4.", "P5.", "P6.", "P7.", "P8."):
        assert expected in html, expected
    names = {p.name for p in (tmp_path / "wiz" / "plansze").iterdir()}
    assert {"p1_zestawienie_K4.png", "p4_pary_K6.png", "p6_slajd_K5.png", "p7_mapa2d_K5.png", "p8_profil.png", "p6_czestosc.png"} <= names
    assert any(n.startswith("p3_klastry_phikon-v2_K4") for n in names) and "p5_seedy.png" in names

    key_file = next((tmp_path / "wiz").glob("blind_key_K*.json"))
    assert sorted(json.loads(key_file.read_text()).values()) == ["dinov2-s14", "phikon-v2", "random"]
    pack = next((tmp_path / "wiz").glob("pakiet_*.zip"))
    with zipfile.ZipFile(pack) as archive:
        assert sorted(archive.namelist()) == ["formularz_oceny.html", "zestaw_X.png", "zestaw_Y.png", "zestaw_Z.png"]
        form = archive.read("formularz_oceny.html").decode("utf-8")
    text_only = "".join(part.split('\'', 1)[-1] if i else part for i, part in enumerate(form.split("data:image")))
    for secret in ("dinov2", "phikon", "random", "losow", "blind_key"):
        assert secret not in text_only, secret

    # the representative seed is what the plates show
    seeds = pd.read_csv(results / "representative_seeds.csv").set_index(["method", "k"])["seed"]
    scores = pd.read_csv(results / "scores_gap.csv").query("variant == 'l2' and method == 'phikon-v2' and k == 5")
    ordered = scores.sort_values(["coverage_mean", "seed"])["seed"].tolist()
    assert seeds[("phikon-v2", 5)] == ordered[(len(ordered) - 1) // 2]
    assert Image.open(tmp_path / "wiz" / "plansze" / "p1_zestawienie_K4.png").width == 250 + 4 * (384 + 12) + 12


def test_missing_images_are_not_ignored(tmp_path):
    with pytest.raises(ValueError, match="cannot read the image"):
        (tmp_path / "broken.png").write_bytes(b"not an image")
        from mpn_al.evaluation.report_html import picture

        picture(tmp_path / "broken.png", "x")
