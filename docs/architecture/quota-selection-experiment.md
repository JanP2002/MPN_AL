# Eksperyment: 50 kafli z 5–10 klastrów (po kilka kafli z klastra)

Protokół zapisany 27.11.2026 **przed policzeniem tej siatki**. Analiza jest eksploracyjna: na tym samym slajdzie znane były już wyniki wyboru
„jeden kafel z klastra” ([tile-selection-experiment.md](tile-selection-experiment.md)). Zastępuje pytanie z
[k-selection-experiment.md](k-selection-experiment.md), gdzie K oznaczało liczbę kafli w rundzie; **tutaj K to liczba klastrów, a wybieranych kafli jest
zawsze 50**. Dane (slajd 24070_25, jeden pacjent, 2668 kafli 1024 px), modele, wejście (9 okien 507 px → 224 px przy 0,5 µm/px, średnia, L2) i 13 cech
z pikseli są takie same jak w eksperymencie selekcji.

## Pytania

1. Czy wybór 50 kafli po kilka z K = 5–10 klastrów jest bardziej wartościowy niż dotychczasowa metoda (50 klastrów po jednym kaflu) i niż losowanie?
2. Która liczba klastrów, który przydział miejsc, która metoda wyboru w klastrze i który model dają najlepszy zestaw, i dlaczego?

## Siatka

| wymiar | wartości |
|---|---|
| model | `lunit-dino-s16`, `phikon-v2`, `midnight-12k`, `dinov2-s14`; `noise-control` (wektory losowe) jako poziom przypadku |
| liczba klastrów K | 5, 6, 7, 8, 9, 10 (KMeans, `n_init` = 1) |
| seed KMeans | 0–29 |
| przydział 50 miejsc | **równy** (`50 // K`, reszta do największych klastrów); **proporcjonalny** do wielkości klastra (metoda największych reszt, minimum 1) |
| metoda w klastrze | **M1**, **M1p**, **M2**, **M3** |

- **M1 (centroid + najbardziej różne):** pierwszy kafel najbliższy centroidowi (kosinusowo); każdy następny to kafel klastra, którego najmniejsza
  odległość kosinusowa do już wybranych z tego klastra jest największa.
- **M1p (przycięte):** jak M1, ale bez 5% kafli klastra najmniej podobnych do centroidu (ochrona przed artefaktami).
- **M2 (losowo w klastrze):** losowe kafle klastra, 20 losowań na seed.
- **M3 (najbliższe centroidowi):** kolejne kafle najbliższe centroidowi (proste rozszerzenie dotychczasowej metody).
- **Bez nakładania:** żadne dwa z 50 kafli nie dzielą powierzchni. Klastry (od największego) biorą po jednym kaflu na zmianę. Gdy klaster nie ma
  już wolnego kafla, brakujące miejsca przejmują kolejno klastry od największego (liczone jako `fallbacks`).

## Odniesienia

| nazwa | opis |
|---|---|
| dotychczasowa metoda (50 × 1) | KMeans K = 50, z każdego klastra kafel najbliższy centroidowi; seedy 0–29 (seedy 0–9 muszą odtwarzać opublikowane wybory) |
| bez klastrowania | M1 na całym slajdzie: kafel najbliższy środkowi, potem 49 najbardziej różnych (jeden wybór na model) |
| losowa | 1000 losowań 50 kafli bez nakładania |
| przestrzenna | 100 wyborów równomiernie rozłożonych na slajdzie |
| pułap | KMeans K = 50 na cechach z pikseli (optymalizuje cechy oceny; punkt skali, nie konkurent) |

## Miary (zawsze dla 50 kafli, poza przestrzenią modeli)

| miara | definicja | lepiej |
|---|---|---|
| pokrycie | średnia, po wszystkich kaflach slajdu, odległość w cechach do najbliższego wybranego | mniej |
| pokrycie, 95. percentyl | to samo dla 5% najgorzej pokrytych | mniej |
| rzadkie kategorie 5% i 1% | ile z 6 kategorii (kość, erytrocyty, barwnik, gęstość jąder, duże jądra, biel; górne 5% lub 1% kafli) ma przedstawiciela; liczba wybranych kafli z kategorii | więcej |
| rozbieżność JS | rozbieżność Jensena-Shannona rozkładu 8 grup kafli (KMeans na cechach) w wyborze i na slajdzie | mniej |
| różnorodność wybranych | średnia odległość cech między wybranymi | więcej |
| pary prawie identyczne | pary wybranych bliższe w cechach niż 1% losowych par kafli slajdu | mniej |
| kafle mało użyteczne | wybrane kafle z udziałem bieli powyżej 70% | mniej |
| kandydaci na jądra | suma szacowanej liczby jąder w wybranych kaflach | opisowo |
| rozmieszczenie | zajęte pola siatki slajdu, średnia odległość do najbliższego wybranego | opisowo |
| stabilność | udział kafli identycznych lub nakładających się między parami seedów | więcej |
| miary klastrowania | silhouette (kosinusowa), Davies-Bouldin, Calinski-Harabasz | pomocniczo |

## Reguła wyboru (zamrożona)

Dotyczy 192 konfiguracji czterech modeli (model × K × przydział × metoda). Średnie i odchylenia liczone są po **pojedynczych wyborach** (seedach;
dla M2 po wszystkich seedach i losowaniach).

1. Dopuszczone są konfiguracje, w których średnia liczba kafli mało użytecznych nie jest większa niż w selekcji losowej.
2. Wygrywa najniższe średnie pokrycie.
3. Remisują konfiguracje, których średnie pokrycie różni się od najlepszego o mniej niż `sqrt((sd_a² + sd_b²) / 2)`. Wśród remisujących decyduje:
   więcej rzadkich kategorii 1%, potem mniejsza rozbieżność JS, potem tańszy model.
4. Trzy najlepsze (reguła stosowana kolejno po usunięciu zwycięzcy) przechodzą potwierdzenie.

## Potwierdzenie

Porównanie każdego z trzech wariantów z: dotychczasową rekomendacją (`lunit-dino-s16`, 50 × 1), metodą 50 × 1 tego samego modelu i losową.

| test | co się zmienia | zaliczony wobec odniesienia, gdy |
|---|---|---|
| nowe seedy | seedy 30–59 | pokrycie niższe o więcej niż łączne odchylenie |
| fragmenty tkanki | cała procedura osobno na dwóch największych fragmentach (1424 i 1185 kafli), pokrycie liczone na fragmencie | średnie pokrycie niższe w obu |
| usuwanie cech | ocena bez jednej z 6 grup cech | niższe we wszystkich 6 |
| podpróbki | 20 podpróbek po 80% kafli (KMeans seed 0) | średnie pokrycie niższe |
| warianty przestrzeni | PCA 128; `n_init` = 10; tylko 675 kafli bez nakładania (seedy 0–9) | niższe w każdym z trzech |

Wariant jest **potwierdzony**, gdy zalicza wszystkie testy wobec dotychczasowej rekomendacji i wobec losowej. W przeciwnym razie raport mówi to
wprost, a rekomendacją pozostaje to, co przechodzi testy.

## Wizualizacje

Pokazywany jest wybór o **medianowym pokryciu** (dolna mediana po seedach, dla M2 po seedach i losowaniach), nie wybrany ręcznie.

## Ograniczenia

Jeden preparat i pacjent. Miary to przybliżenia z pikseli (kolor, komórkowość, struktury), nie ocena patomorfologa. Przy 192 konfiguracjach
zwycięzca siatki może mieć przewagę przypadkową; testy potwierdzające korzystają z tego samego preparatu, więc nie zastępują innych pacjentów.
Czy lepszy wybór poprawia model w Active Learning, pokaże dopiero uczenie na adnotacjach.

## Wyniki (27.11.2026, job 3588744)

Pliki: `outputs/selection/24070_25_quota/` (`analysis/` tabele, `raport_wyniki/`, `raport_wariant/`). Pokrycie: mniej = lepiej, średnie po
pojedynczych wyborach.

| zestaw 50 kafli | pokrycie | rzadkie 1% (z 6) | mało użyteczne | rozbieżność JS |
|---|---|---|---|---|
| dotychczasowa: `lunit-dino-s16`, 50 × 1 | **1,634 ± 0,018** | 4,1 | 2,8 | 0,041 |
| `midnight-12k`, 50 × 1 | 1,633 ± 0,019 | 3,0 | 4,4 | 0,023 |
| zwycięzca reguły: `lunit-dino-s16`, 9 klastrów, równy, M2 | 1,709 ± 0,031 | 3,4 | 3,4 | 0,053 |
| losowa | 1,721 ± 0,038 | 2,5 | 5,6 | 0,026 |
| pułap (KMeans na cechach) | 1,454 | 4,0 | 2,0 | 0,042 |

- **Żadna ze 192 konfiguracji (5–10 klastrów) nie ma pokrycia niższego niż dotychczasowa metoda**; niższe niż średnia losowa ma około 30%.
  Najniższe surowe pokrycie nowych: 1,685 (`midnight-12k`, 10 klastrów, proporcjonalnie, M3), przy 0,5 rzadkiej kategorii 1% z 6.
- Średnio po modelach: metoda M2 1,713, M3 1,742, M1p 1,751, M1 1,835; K = 5 → 10: 1,794 → 1,737; przydział równy 1,769, proporcjonalny
  1,751. M1 bierze najwięcej rzadkich struktur i skrajności; M3 niemal kopie (24–42 pary prawie identyczne) i gubi rzadkie.
- **Żaden z trzech najlepszych nie został potwierdzony**: przegrywają z dotychczasową metodą we wszystkich testach, wobec losowej nie
  przechodzą testu z nowymi seedami (przewaga mniejsza niż rozrzut).
- W grupach kafli: dotychczasowa metoda ma lepsze pokrycie od zwycięzcy we wszystkich 8 grupach i przenosi miejsca z pospolitych grup do
  rzadkich (kość: 8,6 kafla wobec 4,0 w losowaniu).
- **Rekomendacja według reguł: bez zmian** (`lunit-dino-s16`, 50 × 1). Liczb klastrów między 10 a 50 z kilkoma kaflami nie badano.

## Uruchomienie i weryfikacja

Kod: `src/evaluation/quota_selection.py` (przydział, metody, miary), `quota_cli.py` (siatka, potwierdzenia, zapis etapów),
`quota_analysis.py` (reguła i analizy), `quota_results.py`, `quota_report.py`, `quota_plates.py`; skrypty `scripts/run_quota_selection.py`,
`scripts/build_quota_reports.py`; testy `tests/test_quota.py`.

```bash
R='$SCRATCH/mpn_al_runs/selection_24070_25'; D='$SCRATCH/mpn_al_data/slide_24070_25'
bash scripts/plgrid/remote.sh submit --job-name=mpn-quota --time=03:00:00 scripts/run_quota_selection.py \
    --tiles-parquet "$D/tiles.parquet" --descriptors "$R/analysis/descriptors.parquet" \
    --embeddings "dinov2-s14=$R/dinov2-s14/embeddings.parquet" --embeddings "lunit-dino-s16=$R/lunit-dino-s16/embeddings.parquet" \
    --embeddings "phikon-v2=$R/phikon-v2/embeddings.parquet" --embeddings "midnight-12k=$R/midnight-12k/embeddings.parquet" --out "$R/quota"
bash scripts/plgrid/remote.sh fetch "$R/quota" outputs/selection/24070_25_quota/analysis
python scripts/build_quota_reports.py --results outputs/selection/24070_25_quota/analysis --out outputs/selection/24070_25_quota \
    --images <prep>/slides/24070_25/tiles --embeddings "lunit-dino-s16=<embeddings.parquet>" \
    --notes-results outputs/selection/24070_25_quota/notatki_wyniki.md --notes-variant outputs/selection/24070_25_quota/notatki_wariant.md
```

Job trwał 29 minut. Ten sam eksperyment policzony na laptopie (`analysis_local/`) dał identyczne klastrowania (1050 z 1050) i identyczne
wybory w 51 267 z 51 285 przypadków; różnice dotyczą wyłącznie 18 wyborów „50 × 1” w testach na fragmentach i wariantach przestrzeni, a
zwycięzcy i werdykty są takie same. Dotychczasowa metoda odtwarza opublikowane wybory dla K = 50 (60 z 60: modele, kontrola i pułap,
seedy 0–9), a klastrowania K = 5–10 są zgodne z jobem 3588517 (900 z 900). Komentarze w raportach (`notatki_*.md`) pisane są ręcznie dla
tych wyników; po nowym przebiegu trzeba je zweryfikować.

