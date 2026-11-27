# Indeks wyników: gdzie są wykresy i tabele

Stan na 27.11.2026. Ścieżki względem katalogu repozytorium. Folder `outputs/` (ok. 12,6 GB) **nie jest w Git** (leży tylko na tym komputerze, w folderze
synchronizowanym przez OneDrive). Pliki w `outputs/selection/` zawierają obrazy preparatu pacjenta.

## Najnowszy eksperyment: 50 kafli z 5–10 klastrów (drugi zbiór, 27.11.2026)

Opis i wyniki: [quota-selection-experiment.md](quota-selection-experiment.md). Katalog: `outputs/selection/24070_25_quota/` (zawiera obrazy
preparatu pacjenta).

| plik | zawartość |
|---|---|
| `outputs/selection/24070_25_quota/raport_wyniki/raport_eksperyment_i_wyniki.pdf` (i `.html`) | raport 1: pytanie, dane, miary, wyniki, dlaczego |
| `outputs/selection/24070_25_quota/raport_wariant/raport_wybrany_wariant.pdf` (i `.html`) | raport 2: UMAP, plansze, mapy WSI, liczebność klastrów |
| `outputs/selection/24070_25_quota/raport_wariant/plansze/` | plansze i mapy w pełnej rozdzielczości |
| `outputs/selection/24070_25_quota/raport_wariant/pakiet_dla_patomorfologa_PQR.zip` | zaślepione zestawy (wysłać przed raportami; klucz `blind_key_50.json`) |
| `outputs/selection/24070_25_quota/analysis/` | `summary.csv` (wszystkie konfiguracje), `winners.csv`, `confirmation.csv`, `verdicts.csv`, `factor_effects.csv`, `stability.csv`, `selections.parquet` |

## Który plik z podsumowaniem dotyczy którego zbioru

| zbiór | co to | sekcje niżej |
|---|---|---|
| **Drugi zbiór** (`prep_33de9068`, slajd 24070_25) | 2668 kafli 1024 px z jednego WSI, jeden pacjent; cel: wybór modelu do selekcji kafli | 1 |
| **BaMBo** | 204 obrazy, 47 pacjentów, etykiety zastępcze (pseudo-jądra); cel: sprawdzenie potoku | 2–5 |
| żaden (dane syntetyczne) | testy klastra i potoku | 6 |

### Podsumowania dla drugiego zbioru (slajd 24070_25)
| plik | rodzaj |
|---|---|
| `outputs/selection/24070_25/raport_zespol/raport_selekcja_kafli.pdf` (i `.html`) | raport dla zespołu |
| `docs/architecture/tile-selection-experiment.md` | protokół i wyniki |
| `outputs/selection/24070_25/report.md` | automatyczny raport z analizy |
| `outputs/selection/24070_25/ranking.csv`, `summary.csv` | tabele zbiorcze |
| `outputs/selection/24070_25/figures/index.html` | przegląd wykresów i plansz |

### Podsumowania dla BaMBo
| plik | czego dotyczy |
|---|---|
| `docs/architecture/experiment-plan.md` | siatka segmentacji, powtórki, figury, wada i poprawka pseudo-etykiet |
| `docs/architecture/raport-wyniki-i-testy.md` | siatka 48 przebiegów, powtórka i testy lokalne (stan na 23.11) |
| `docs/architecture/model-input-report.md` | porównanie wizualne i wpływ kontekstu (liczby z BaMBo; opis wejścia modeli jest ogólny) |
| `docs/architecture/embedding-comparison.md` | klastrowanie embeddingów dwóch modeli |
| `outputs/plgrid/exp_bambo_final/report.md` | automatyczny raport siatki |
| `outputs/plgrid/exp_bambo_rerun/bambo_proxy/report.md`, `outputs/plgrid/exp_bambo_viz/bambo_proxy/report.md` | automatyczne raporty powtórek |
| `outputs/figures/bambo_validation/index.html`, `outputs/figures/bambo_test/index.html` | przegląd figur segmentacji |
| `outputs/clustering/bambo2/report.md`, `outputs/clustering/bambo2/figures/index.html` | automatyczny raport i figury klastrowania |

Częściowo BaMBo: `ssl-base-models.md` (próby na laptopie na BaMBo, pomiary na A100 na danych syntetycznych) i `local-run-report.md` (czasy; z BaMBo tylko
niektóre odniesienia). `plgrid.md` i `data-layout.md` nie zawierają wyników na obrazach. **Wyników z obu zbiorów nie należy porównywać:** inne zadanie, dane i miary.

## 1. Selekcja kafli, slajd 24070_25 (drugi zbiór, 26.11.2026)

Opis: [tile-selection-experiment.md](tile-selection-experiment.md). Katalog: `outputs/selection/24070_25/`.

### Raport dla zespołu: `raport_zespol/`
| plik | zawartość |
|---|---|
| `raport_selekcja_kafli.pdf`, `.html` | wnioski, tabele, wykresy, plansze, mapa slajdu, ograniczenia |
| `pakiet_dla_patomorfologa.zip`, `dla_patomorfologa/` | zaślepione zestawy A, B, C i formularz oceny (wysłać **przed** raportem) |

### Wykresy: `figures/` (przegląd: `index.html`)
| plik | co pokazuje |
|---|---|
| `curves_selection.png` | pokrycie, 95. percentyl, rzadkie struktury, reprezentatywność, zasięg na slajdzie wobec K, z punktami odniesienia |
| `curves_clustering.png` | silhouette, Davies-Bouldin, Calinski-Harabasz, ARI między seedami wobec K |
| `heatmap_coverage_mean.png`, `heatmap_silhouette.png`, `heatmap_ari_seed.png` | model × K |
| `matrix_ari.png`, `matrix_knn_overlap.png` | zgodność modeli między sobą (K = 80) |
| `map_<model>.png` (4 modele i `noise-control`) | UMAP: klastry z wybranymi kaflami, biel, gęstość jąder, położenie |
| `board_lunit-dino-s16_K50.png`, `board_midnight-12k_K50.png`, `board_random_K50.png` | po 50 wybranych kafli |
| `slide_map_K50.png` | położenie trzech zestawów na slajdzie |
| `blind_A.png`, `blind_B.png`, `blind_C.png` | te same zestawy bez nazw; klucz w `../blind_key.json` (nie pokazywać oceniającemu) |

### Tabele
| plik | zawartość (jeden wiersz = ) |
|---|---|
| `ranking.csv` | model: pokrycie według zamrożonej reguły, rozrzut, remisy |
| `summary.csv` | wariant × metoda × K: średnie i odchylenia wszystkich miar selekcji, percentyl wobec losowej |
| `selection_scores.csv` | pojedyncza selekcja (seed lub losowanie) |
| `selections.parquet` | wybrany kafel (`tile_id`, kolejność); bez selekcji losowych |
| `clustering_metrics.csv` | model × K × seed: silhouette, DB, CH, inercja |
| `seed_stability.csv` | zgodność klastrowań i wybranych kafli między seedami |
| `models_agreement.csv`, `neighbour_quality.csv` | porównania modeli i jakość sąsiedztw |
| `board_comparison.csv` | pokrywanie się zestawów z plansz |
| `robustness_descriptors.csv` | ranking po usunięciu cech (analiza po wynikach) |
| `descriptors.parquet`, `manifest.parquet` | 13 cech i metadane 2668 kafli |
| `report.md`, `run.json` | automatyczny raport i parametry przebiegu |

## 2. [BaMBo] Segmentacja: siatka 48 przebiegów (pseudo-etykiety)

Opis: [experiment-plan.md](experiment-plan.md), zestawienie: [raport-wyniki-i-testy.md](raport-wyniki-i-testy.md).
**Uwaga:** 28 z 204 obrazów miało zepsute pseudo-etykiety; wszystkie trzy przebiegi poniżej je zawierają. Powtórkę z poprawionymi etykietami anulowano przed startem.

| katalog | co to | pliki |
|---|---|---|
| `outputs/plgrid/exp_bambo_final/` | pierwszy przebieg (21.11) | `runs.csv`, `curves.csv`, `comparison.csv`, `report.md` |
| `outputs/plgrid/exp_bambo_rerun/bambo_proxy/` (4,2 GB) | powtórka, PQ identyczne | te same tabele i katalogi 48 przebiegów |
| `outputs/plgrid/exp_bambo_viz/bambo_proxy/` (7 GB) | przebieg z zapisem predykcji | jak wyżej |

W katalogu przebiegu `<model>__native__f<odsetek>__s<seed>/`: `result.json` (metryki), `tile_counts.jsonl` (zliczenia per obraz), `head/` (`head.pt`, `history.json`),
`provenance.json`, a w `exp_bambo_viz` także `viz/` (predykcje). `runs.csv`: wiersz na przebieg; `curves.csv`: średnie po seedach (krzywe uczenia);
`comparison.csv`: testy wobec `dinov2-s14`.

### Figury segmentacji: `outputs/figures/bambo_validation/` i `bambo_test/`
Po 9 obrazów, na każdy `_gallery`, `_zoom`, `_curve`, `_seeds`, `_disagreement` (PNG), `viz_scores.csv` (PQ każdego przebiegu na każdym obrazie) i `index.html`.

## 3. [BaMBo] Porównanie wizualne modeli i wpływ kontekstu (BaMBo)

Opis: [model-input-report.md](model-input-report.md).

| katalog | zawartość |
|---|---|
| `outputs/plgrid/visual_bambo/` | `<obraz>_compare.png`, `_context.png`; `agreement.csv`, `context_effect.csv`, `context_summary.csv`, `geometry.csv` |
| `outputs/plgrid/visual_bambo_zoom/` | to samo, dodatkowo `<obraz>_zoom.png` i `reference_pq.csv` |

## 4. [BaMBo] Klastrowanie embeddingów (tylko dwa modele: `phikon-v2`, `lunit-dino-s16`)

Opis: [embedding-comparison.md](embedding-comparison.md). Katalog: `outputs/clustering/bambo2/`.

- Tabele: `metrics.csv`, `best_k.csv`, `clusters.parquet`, `compare_models.csv`, `compare_spaces.csv`, `report.md`.
- Wykresy w `figures/` (`index.html`): `curves_<wariant>.png`, `heatmap_*.png`, `matrix_*.png`, `map_<model>.png`, `composition_<model>.png`, `gallery_<model>.png`.
- Embeddingi wejściowe: `outputs/plgrid/bambo/embeddings.parquet` (phikon-v2), `outputs/local_bambo/lunit-dino-s16/embeddings.parquet`.

## 5. [BaMBo] Pseudo-etykiety: diagnoza i poprawka

W `outputs/`: `bambo_pseudolabel_check.csv` (przed), `bambo_pseudolabel_check_v2.csv` (po), `bambo_otsu_study.csv`, `bambo_floor_study.csv`, `bambo_wb_study.csv`,
nowe maski `bambo_masks_wb/masks/` (204), podział bez 11 obrazów `bambo_split_v2.csv`, lista `bambo_excluded_v2.txt`.

## 6. Testy klastra i czasy

`outputs/plgrid/smoke-<job>/` (`smoke_report.md`, `.json`, `pytest.txt`): czasy i pamięć na A100. Czasy na laptopie: [local-run-report.md](local-run-report.md).

## Nie są wynikami eksperymentów
`outputs/demo_array/`, `outputs/local_single/`, `outputs/plgrid/arraytest/`, `exp_demo/`, `exp_after/`: dane i przebiegi testowe potoku. `outputs/bambo_png/`: obrazy wejściowe.
