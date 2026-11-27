"""Report 2 of the K selection experiment (Polish): plates comparing what the two best models select, and a blinded pack."""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np

from .k_report import Results
from .report_html import num, page, percent, picture, read_notes, table, to_pdf
from .selection_plates import CATEGORIES, RARE_NAMES, Study
from .tile_descriptors import RARE_FEATURES

logger = logging.getLogger("mpn_al.plates")


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    path.write_bytes(buffer.tobytes())  # imwrite fails silently on non-ASCII Windows paths


def _name(method: str) -> str:
    return "losowa" if method == "random" else method


def blind_k(results: Results, model: str) -> int:
    """The K of the blinded comparison: the recommended K of the first model, or the largest K when K cannot be told apart."""
    choice = results.choice[(results.choice["variant"] == "l2") & (results.choice["method"] == model)].iloc[0]
    return int(max(results.ks)) if choice["all_tied"] else int(choice["recommended_k"])


def blinded_pack(study: Study, out: Path, k: int, *, seed: int = 11, pdf: bool = True) -> dict[str, str]:
    """Three unnamed sets of ``k`` tiles and a form; the key stays outside the folder that is sent."""
    folder = out / "dla_patomorfologa"
    folder.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(seed).permutation(study.sets)
    key = {letter: str(method) for letter, method in zip("XYZ", order)}
    for letter, method in key.items():
        _save_png(folder / f"zestaw_{letter}.png", study.blind_set(study.pick(method, k), f"Zestaw {letter}"))
    (out / f"blind_key_K{k}.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    questions = ("Różnorodność zestawu (1 = mała, 5 = duża)", "Jak dobrze oddaje to, co jest w preparacie (1–5)",
                 "Numery kafli nieprzydatnych do adnotacji", "Numery kafli prawie identycznych ze sobą", "Czego w zestawie brakuje",
                 "Co jest w nadmiarze", "Czy taka liczba kafli w jednej rundzie jest wygodna do adnotacji?")
    rows = "".join(f"<tr><td>{q}</td><td></td><td></td><td></td></tr>" for q in questions)
    sets = "".join(picture(folder / f"zestaw_{x}.png", f"Zestaw {x}", css="page") for x in key)
    body = (f"<h1>Ocena trzech zestawów po {k} kafli (X, Y, Z)</h1>"
            "<p>Trzy zestawy kafli z tego samego preparatu, wybrane trzema różnymi sposobami; sposoby ujawnimy po ocenie. Kafle są uporządkowane od "
            "najmniej do najbardziej komórkowych (pomiar automatyczny). Powiększenia: pliki <code>zestaw_X.png</code>, <code>zestaw_Y.png</code>, "
            "<code>zestaw_Z.png</code>. Obrazy preparatu pacjenta: do użytku wewnętrznego.</p>"
            f"<table><thead><tr><th>pytanie</th><th>zestaw X</th><th>zestaw Y</th><th>zestaw Z</th></tr></thead><tbody>{rows}</tbody></table>"
            "<p><b>Kolejność zestawów od najlepszego do adnotacji:</b> ______ , ______ , ______</p><p><b>Uzasadnienie i uwagi:</b></p>"
            "<p style='height:100px;border:1px solid #d7d9dd;border-radius:6px'></p><p>Imię i nazwisko: ______________________ &nbsp;&nbsp; Data: ______________</p>"
            + sets)
    form = folder / "formularz_oceny.html"
    form.write_text(page("Ocena zestawów X, Y, Z", body, extra_style="td{height:42px} td:first-child{width:31%}"), encoding="utf-8")
    if pdf:
        to_pdf(form, folder / "formularz_oceny.pdf")
    with zipfile.ZipFile(out / "pakiet_dla_patomorfologa_X_Y_Z.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(folder.iterdir()):
            archive.write(file, file.name)
    return key


def build(results: str | Path, image_of: Callable[[str], np.ndarray], vectors: Mapping[str, np.ndarray], out: str | Path, *,
          notes: str | Path | None = None, pdf: bool = True) -> Path:
    r = Results(results)
    out = Path(out)
    plates = out / "plansze"
    plates.mkdir(parents=True, exist_ok=True)
    study = Study(r, image_of, vectors)
    a, b = study.models
    text = read_notes(notes)
    note = lambda key: f"<div class='note'>{text[key]}</div>" if key in text else ""

    def plate(name: str, rgb: np.ndarray, caption: str, css: str = "") -> str:
        _save_png(plates / f"{name}.png", rgb)
        logger.info("plate %s", name)
        return picture(rgb, caption, width=2600, quality=86, css=css)

    choice = r.choice[r.choice["variant"] == "l2"].set_index("method")
    card_ks = {m: sorted({min(study.ks), max(study.ks), int(choice.loc[m, "recommended_k"])}) for m in study.models}
    scores = r.scores[(r.scores["variant"].isin(["l2", "baseline"])) & (r.scores["constraint"] == "no-overlap")]
    scores = scores.set_index(["method", "k", "seed"])
    pair_counts = study.pair_counts()
    contents = study.contents().set_index(["method", "k"])

    pair_rows = []
    for k in study.ks:
        part = pair_counts[pair_counts["k"] == k].set_index("category")
        pair_rows.append([f"K = {k}"] + [f"{int(part.loc[c, 'shown'])} / {num(part.loc[c, 'mean_over_seeds'], 1)}" for c in CATEGORIES])
    content_tables = ""
    heading = ("<h3>V2. Co zawierają zestawy: w ilu seedach (losowaniach) jest choć jeden kafel danego typu</h3>"
               "<p class='mute'>Typ: 5% kafli slajdu o największej wartości cechy. Średnia po wszystkich seedach modeli i losowaniach.</p>")
    for method in study.sets:
        rows = [[f"K = {k}"] + [percent(contents.loc[(method, k), f"has_{f}"]) for f in RARE_FEATURES]
                + [percent(contents.loc[(method, k), "white"]), num(contents.loc[(method, k), "nuclear_density"] / 1000, 1)] for k in study.ks]
        content_tables += (f"<div class='keep'>{'' if content_tables else heading}<h3>{_name(method)}</h3>"
                           + table(["K"] + [f"ma: {RARE_NAMES[f]}" for f in RARE_FEATURES] + ["średnia biel", "gęstość jąder, tys./mm²"], rows) + "</div>")
    shown_rows = []
    for k in study.ks:
        for method in study.sets:
            s = scores.loc[(method, k, study.seed(method, k))]
            shown_rows.append([f"K = {k}", _name(method), study.seed(method, k), num(s["coverage_mean"], 3), num(s["gap"], 2),
                               int(s["rare_categories"]), num(s["nearest_neighbour_mm"], 2)])

    mosaic, origin = study.mosaic()
    logger.info("slide mosaic ready")
    chapters = ""
    for k in study.ks:
        map_path = study.space_maps(k, plates / f"p7_mapa2d_K{k}.png")
        chapters += (f"<h2 class='page'>K = {k}</h2>{note(f'K{k}')}"
                     + plate(f"p1_zestawienie_K{k}", study.overview(k), f"P1. K = {k}. Wiersze: {a}, {b}, losowa. Zielona ramka: kafel wybrany "
                             "przez oba modele; pomarańczowa: nakłada się z kaflem drugiego modelu. Numer = klaster od największego "
                             "(w losowej: kolejność wg gęstości jąder).")
                     + plate(f"p4_pary_K{k}", study.pairs(k), f"P4. K = {k}. Te same wybory dopasowane w pary.")
                     + plate(f"p6_slajd_K{k}", study.slide_numbers(mosaic, origin, k), f"P6. K = {k}. Położenie wybranych kafli na slajdzie "
                             "(numery jak na planszy P1).", css="page")
                     + picture(map_path, f"P7. K = {k}. Mapy 2-D przestrzeni obu modeli (odległości między wyspami są umowne).", lossless=True))

    across = "".join(plate(f"p2_przyrost_{m}", study.progression(m), f"P2. {_name(m)}: wybór dla K = {study.ks[0]}–{study.ks[-1]} z jednego seeda. "
                           "KMeans dla kolejnych K nie jest hierarchiczny, więc kafle mogą znikać.") for m in study.sets)
    profile_path = study.profiles(plates / "p8_profil.png")
    across += picture(profile_path, "P8. Profil cech wybranych kafli (średnia po wszystkich seedach; 0 = średnia slajdu).", lossless=True)
    across += plate("p6_czestosc", study.slide_frequency(mosaic, origin), "P6b. Które miejsca slajdu są wybierane najczęściej.", css="page")
    recommended = {m: int(choice.loc[m, "recommended_k"]) for m in study.models}
    variability = plate("p5_seedy", study.seed_variability_both(recommended),
                        "P5. Osiem kolejnych seedów i kafle wybierane najczęściej, przy zalecanym K: "
                        + ", ".join(f"{m} (K = {k})" for m, k in recommended.items()) + ".")
    cards = "".join(plate(f"p3_klastry_{m}_K{k}", study.cluster_cards(m, k), f"P3. {m}, K = {k}: karty klastrów.", css="page" if i else "")
                    for i, (m, k) in enumerate((m, k) for m in study.models for k in card_ks[m]))

    k_blind = blind_k(r, a)
    key = blinded_pack(study, out, k_blind, pdf=pdf)
    logger.info("blinded pack for K = %d: %s", k_blind, key)

    body = f"""
<h1>Co wybierają dwa najlepsze modele? Porównanie wizualne dla K = {study.ks[0]}–{study.ks[-1]}</h1>
<p class="mute">Slajd 24070_25 (jeden pacjent). Modele według reguły protokołu: <b>{a}</b> i <b>{b}</b>, obok selekcja losowa.
Obrazy preparatu pacjenta: do użytku wewnętrznego.</p>
<div class="warn"><b>Przed rozesłaniem:</b> raport pokazuje nazwy modeli. Osoba oceniająca zaślepione zestawy (X, Y, Z z tego katalogu, a także
A, B, C z raportu o K = 50) powinna je ocenić <b>zanim</b> zobaczy ten raport.</div>
<h2>1. Najważniejsze różnice</h2>
<div class="box">{text.get('summary', '<p>Porównanie w tabelach i na planszach poniżej.</p>')}</div>
<div class="keep"><h3>V1. Pary kafli obu modeli: na planszy (seed medianowy) / średnio po wszystkich seedach</h3>
{table(["K"] + list(CATEGORIES), pair_rows)}
<p class="mute">„Podobne cechy”: odległość w cechach z pikseli mniejsza niż dla 95% losowych par kafli.</p></div>
{content_tables}
{note('contents')}
<div class="keep"><h3>V3. Liczby dla wyborów pokazanych na planszach</h3>
{table(["K", "metoda", "seed", "pokrycie", "G", "rzadkie kategorie z 6", "do najbliższego wybranego, mm"], shown_rows)}</div>
<h2>Jak czytać plansze</h2>
<ul>
<li>Na planszach jest <b>seed o medianowym pokryciu</b> (reguła zapisana przed wynikami), a nie najładniejszy ani pierwszy.</li>
<li>Numer kafla to miejsce jego klastra od największego. Ten sam numer na planszy P1, mapie slajdu P6 i mapie 2-D P7; numery dwóch modeli
nie odpowiadają sobie (odpowiedniki pokazuje P4).</li>
<li>Pod kaflem liczby z cech z pikseli (przybliżenia): udział bieli, gęstość jąder (tys./mm²), udział zbitej tkanki eozynofilowej (kość).</li>
<li>Pliki w pełnej rozdzielczości do powiększania: katalog <code>plansze/</code>.</li>
</ul>
{chapters}
<h2 class="page">Porównanie między K</h2>
{note('across')}
{across}
<h2 class="page">Zależność od seeda</h2>
{note('seeds')}
{variability}
<h2 class="page">Karty klastrów</h2>
{note('cards')}
{cards}
<h2>Zaślepiona ocena i ograniczenia</h2>
<p>Pakiet <code>pakiet_dla_patomorfologa_X_Y_Z.zip</code> zawiera trzy zestawy po {k_blind} kafli bez nazw i formularz. Klucz leży poza pakietem
(<code>blind_key_K{k_blind}.json</code>).</p>
<ul><li>Jeden preparat; obserwacje z oglądania plansz to wrażenia z jednego seeda, nie pomiary.</li>
<li>Autor obserwacji nie jest patomorfologiem; ocena morfologiczna należy do specjalisty.</li></ul>
"""
    target = out / "raport_wizualizacje.html"
    target.write_text(page("Porównanie wizualne wyborów, K = 5–10", body), encoding="utf-8")
    if pdf:
        to_pdf(target, out / "raport_wizualizacje.pdf", timeout=600)
    return target
