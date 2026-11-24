# Eksperyment: porównanie modeli bazowych SSL do segmentacji komórek

Protokół i stan wdrożenia. Kontekst: [ssl-base-models.md](ssl-base-models.md), uruchamianie na klastrze:
[plgrid.md](plgrid.md).

## Pytanie i hipotezy (zapisane przed wynikami)

Który zamrożony enkoder SSL daje najlepszą segmentację komórek i reprezentacje przy małej liczbie anotacji i jaki
jest koszt. H1: modele pretrenowane na histologii biją ogólny DINOv2. H2: `h0-mini` jest konkurencyjny wobec
dużych modeli. H3: dla megakariocytów 0,5 µm/px wystarcza, dla małych komórek potrzeba 0,25.

## Konstrukcja

- **Jedna komórka siatki** = model × rozdzielczość głowicy × odsetek kafli treningowych × seed
  (`configs/experiments/*.yaml`, `scripts/run_experiment.py`). Indeks zadania tablicy Slurm wskazuje komórkę.
- **Stałe:** architektura i hiperparametry głowicy, kafle, podział po pacjentach, augmentacje.
- **Podział:** plik z kolumnami `tile_id, patient_id, split` (CSV albo `tiles.parquet`). Pacjent w dwóch zbiorach
  to błąd. Podzbiory kafli są zagnieżdżone dla danego seeda.
- **Test:** `--final` odblokowuje zbiór testowy, jeden raz po zamrożeniu konfiguracji. Bez flagi nie jest
  nawet liczony.
- **Metryki** (`mpn_al.evaluation`): PQ/DQ/SQ, mPQ i PQ per klasa, przedział 95% przez bootstrap **po
  pacjentach**, nakład korekty (obiekty do dodania, usunięcia, poprawy, piksele do poprawy), czas na kafel.
  Przy mniej niż 3 pacjentach przedział to NaN, bo nie ma uczciwego.
- **Wyniki:** `<out>/<grid>/<run_id>/result.json`, `tile_counts.jsonl` (zliczenia per kafel, do ponownych
  analiz), `provenance.json`, `head/`.

## Kolejność

| Krok | Stan |
|---|---|
| 0. `remote.sh smoke` na Atenie, czasy i pamięć | **wykonane 20.11.2026** (A100, wyniki w ssl-base-models.md, sekcja 10); runner siatki sprawdzony w tablicy 8 zadań na danych demo (plgrid.md) |
| 1. Decyzje: tło/ignore, obrys komórki, klasy, `mpp`, podział po pacjentach | do ustalenia z zespołem i lekarzem |
| 2. Metryki, runner, siatki, testy | **gotowe**, 111 testów (lokalnie, losowy enkoder) |
| 3. Pilotaż `pilot.yaml`: czas i dobór hiperparametrów na walidacji | po kroku 0 i 1 |
| 4. Zamrożenie konfiguracji, pełna siatka `encoders.yaml` | po pilotażu |
| 5. Jedno otwarcie testu (`--final`), baseline'y zewnętrzne, ablacje | po kroku 4 |

## Uruchomienie pilotażu

```bash
python scripts/run_experiment.py --grid configs/experiments/pilot.yaml --list        # ile komórek
bash scripts/plgrid/remote.sh put ./tiles '$SCRATCH/mpn_al_data/tiles'               # kafle, maski, split.csv
bash scripts/plgrid/remote.sh submit --array=0-17 scripts/run_experiment.py \
    --grid configs/experiments/pilot.yaml --images '$SCRATCH/mpn_al_data/tiles' \
    --masks '$SCRATCH/mpn_al_data/masks' --split-file '$SCRATCH/mpn_al_data/split.csv' \
    --mpp 0.25 --out '$SCRATCH/mpn_al_runs/experiments' --skip-existing
```

## Zbieranie wyników

```bash
python scripts/summarize_experiment.py <katalog> --split validation --reference dinov2-s14
```

Zapisuje `runs.csv` (jedna komórka = wiersz), `curves.csv` (średnia i odchylenie po seedach dla każdego modelu,
rozdzielczości i odsetka kafli, czyli krzywe uczenia), `comparison.csv` (każdy model wobec odniesienia, test
Wilcoxona sparowany po (rozdzielczość, odsetek, seed) z poprawką Holma) i `report.md`. Przy kilku parach
najmniejsze możliwe p jest duże (6 par: 0,031), więc brak istotności znaczy „za mało przebiegów”.

## Decyzja: cache cech enkodera

Nie zrobiony celowo. Trening stosuje przerzuty i zmiany koloru przed enkoderem, więc cache cech oznaczałby inną,
słabszą augmentację i zmienił wyniki. Wrócimy do tego po pomiarze czasu treningu na GPU (`smoke`): jeśli enkoder
dominuje koszt, można trenować z cache bez augmentacji fotometrycznej i porównać obie wersje na pilotażu.

## Wynik eksperymentu zastępczego na BaMBo (21.11.2026)

**Co to jest:** `configs/experiments/bambo_proxy.yaml`, 204 obrazy BaMBo, 47 pacjentów (podział losowy po
pacjentach: 30 trening / 8 walidacja / 9 test, ziarno 0), etykiety to **maski jąder z baseline'u barwnikowego**,
nie anotacje lekarza. 4 modele otwarte × 4 odsetki obrazów (10–100%) × 3 seedy = 48 przebiegów na A100 (job
3567632, 4–5 min na przebieg), zbiór testowy policzony raz (`--final`), konfiguracja nie była stroiona (pilotaż
mierzył tylko czas). Mierzy: jak dobrze głowica na danym zamrożonym enkoderze odtwarza obrysy jąder
znalezione barwnikowo. **Nie mówi nic o megakariocytach, o zgodności z patologiem ani o komórkach z
cytoplazmą.** BaMBo nie zapisuje rozdzielczości (założone 0,5 µm/px).

PQ na zbiorze testowym (średnia z 3 seedów; odchylenie między seedami 0,003–0,024):

| model | 10% (13 obr.) | 25% | 50% | 100% (130 obr.) |
|---|---|---|---|---|
| `dinov2-s14` (ogólny) | 0,510 | 0,499 | 0,497 | 0,490 |
| `lunit-dino-s16` | 0,522 | 0,543 | 0,552 | 0,547 |
| `phikon-v2` | 0,517 | 0,524 | 0,538 | 0,539 |
| `midnight-12k` (ViT-g) | 0,499 | 0,518 | 0,525 | 0,521 |

Test sparowany (Wilcoxon po odsetku i seedzie, 12 par, poprawka Holma) względem `dinov2-s14`: `lunit-dino-s16`
+0,042 (p_holm 0,003), `phikon-v2` +0,031 (0,003), `midnight-12k` +0,017 (0,034).

**Jak to czytać:**
- Trzy modele pretrenowane na histologii lub dużych danych są nieznacznie lepsze od ogólnego DINOv2 (różnice
  0,02–0,04 PQ), co jest zgodne z H1, ale tylko w tym zastępczym zadaniu. Pary nie są niezależne: ten sam
  zbiór testowy (9 pacjentów) i zagnieżdżone podzbiory, więc p-wartości są **zbyt optymistyczne**. Prawdziwa
  liczba niezależnych jednostek to 9 pacjentów testowych.
- Lunit (ViT-S, 22 mln parametrów) jest tu nie gorszy niż ViT-g (1,1 mld) i ViT-L. Różnic między tymi trzema
  nie testowano. Duży model nie daje przewagi w tym zadaniu, a koszt treningu głowicy jest podobny (125–158 s),
  bo dominuje dekoder.
- Krzywe uczenia są niemal płaskie od 25% obrazów (dla `dinov2-s14` nawet lekko spadają). Zadaniem jest odtworzenie
  deterministycznej reguły barwnej, którą głowica uczy się z kilkudziesięciu obrazów, więc krzywe nie mówią nic o
  tym, ile anotacji lekarza potrzeba.
- Walidacja w pilotażu dała PQ ok. 0,60, test 0,50–0,55; różnica może wynikać z innego składu pacjentów
  (test to 9 pacjentów, w tym obrazy innej skali), nie badano jej.
- Nie sprawdzono: modeli bramkowanych (H-optimus, UNI2-h, Virchow2, H0-mini), rozdzielczości (H3), prawdziwych
  anotacji, innego podziału po pacjentach (jeden podział, ziarno 0).

Wnioski dla dalszych prac: potok od obrazu po `report.md` działa w skali 48 przebiegów i daje powtarzalne
wyniki (małe odchylenia między seedami). Ranking enkoderów trzeba powtórzyć na prawdziwych anotacjach komórek
przed jakąkolwiek decyzją.

### Powtórzenie eksperymentu zastępczego (22–23.11.2026)

Ta sama siatka (`configs/experiments/bambo_proxy.yaml`, bez zmian, bez `--skip-existing`, `--final`) uruchomiona ponownie na Atenie jako tablica 3573579 (indeksy 0–47) do nowego katalogu `exp_bambo_rerun`; wyniki lokalnie w `outputs/plgrid/exp_bambo_rerun/bambo_proxy` (runs.csv, curves.csv, comparison.csv, report.md). Wszystkie 48 komórek ma `result.json`, żadne zadanie nie padło ani nie było zgłaszane ponownie.

Porównanie z przebiegiem 3567632 (`outputs/plgrid/exp_bambo_final`), PQ na teście, parami po `run_id`: maksymalna bezwzględna różnica 0,0000 (wszystkie 48 wartości identyczne), więc żadna różnica nie przekracza progu 0,02. Średnie PQ per model (ułamek 0,10–1,00) i ułamek też się zgadzają w całości; porównanie z dinov2-s14 (comparison.csv) jest takie samo (lunit-dino-s16 +0,0418, phikon-v2 +0,0305, midnight-12k +0,0166, Holm p odpowiednio 0,0029, 0,0029, 0,0342). Różnią się tylko czasy (`train_seconds`, średnio o ok. 7 s), co potwierdza, że były to świeże wykonania, a nie ponowne wczytanie starych wyników.

Wniosek: pipeline (seedy, podział, trening głowy, ewaluacja) jest w tej konfiguracji w pełni deterministyczny, niedeterminizm GPU nie wpływa na PQ. Wartości z sekcji powyżej pozostają bez zmian.

## Zapis predykcji do wizualizacji (etapy 1–2, 23.11.2026)

Plan wizualizacji wyników każdego modelu na obrazach: najpierw przebieg zapisuje predykcje, potem figury rysuje się offline
(bez GPU i bez enkodera). Zrobione są etapy zapisu (1–2) i skrypt składający figury (3, `scripts/visualize_experiment.py`); nie uruchomiono go jeszcze na wynikach z klastra.

- **Konfiguracja:** sekcja `visualize: {per_split: N, seed: S}` w pliku siatki (brak sekcji = nic nie jest zapisywane).
  Nadpisanie z wiersza poleceń: `--viz-per-split N`, wyłączenie: `--no-viz`.
- **Wybór kafli zależy tylko od danych**, nigdy od wyniku modelu (`src/experiment/viz.py`), więc wszystkie przebiegi siatki pokazują te same kafle:
  `dense` (najwięcej komórek referencyjnych na megapiksel), `atypical` (rozmiar inny niż najczęstszy w zbiorze), `random` (ziarno z konfiguracji).
  Najpierw po jednym kaflu na pacjenta, dopiero potem powtórzenia. Grupa jest zapisana w `viz/index.csv`.
- **Bramka testu:** zbiór testowy jest wybierany i zapisywany tylko wtedy, gdy jest oceniany (`--final`), tak jak jego metryki.
- **Zawartość `<przebieg>/viz/<zbiór>/<tile_id>.npz`:** mapa instancji, prawdopodobieństwo (uint8), HV (int8), PCA tokenów w rozdzielczości siatki
  tokenów (uint8), klasy instancji oraz referencja i maska `ignore`, względem których liczono wynik. Test sprawdza, że TP, FP i FN policzone z tego pliku
  są równe zliczeniom w `tile_counts.jsonl`, więc figura nie może się rozjechać z tabelą.
- **Koszt:** kilkanaście kafli na przebieg, kilkaset kB na kafel w formie skompresowanej; PCA liczone z macierzy kowariancji D × D (ok. 1 s dla ViT-g).
  `bambo_proxy.yaml` ma teraz `per_split: 9`; ta sekcja nie wpływa na żadną metrykę.

### Figury z zapisanych predykcji (`scripts/visualize_experiment.py`)

```bash
python scripts/visualize_experiment.py <katalog_eksperymentu> --images <obrazy> --out outputs/figures/<nazwa>     --split validation --fraction 1.0 --seed 0 --zoom 256
```

Na kafel (z grupy `dense`, `atypical` lub `random`, patrz wyżej) powstają: `_gallery.png` (wiersz na model: kontury | błędy | prawdopodobieństwo | PCA tokenów),
`_zoom.png` (najgęstsze okno powiększone bez interpolacji), `_curve.png` (kolumny to odsetki kafli treningowych, ten sam seed), `_seeds.png` (kontury
wszystkich seedów i miejsca niezgodności), `_disagreement.png` (ile modeli uznaje piksel za komórkę), a do tego `viz_scores.csv` (każdy przebieg × kafel) i `index.html`.

- **Błędy** są dopasowane tak jak w metryce PQ (IoU > 0,5): trafienie (TP), fałszywy alarm (FP), pominięcie (FN), w kolorach Okabe-Ito dla osób z zaburzeniami widzenia barw.
- **Spójność z tabelą:** `viz_scores.csv` ma kolumnę `matches_counts`; skrypt ostrzega, jeśli zapisana predykcja nie odtwarza `tile_counts.jsonl` przebiegu.
- **Zbiór testowy** istnieje tylko dla przebiegów z `--final`; bez niego skrypt kończy się błędem, zamiast cokolwiek rysować.
- Skrypt nie potrzebuje GPU ani enkodera; wystarczą zapisane `viz/` i katalog z obrazami (obraz jest skalowany do rozdzielczości roboczej predykcji).
- Teksty na figurach są po angielsku i bez polskich znaków (OpenCV); strona `index.html` jest po angielsku.
- **Uwaga do `_seeds.png` i `_curve.png`:** pojawiają się tylko wtedy, gdy siatka ma więcej niż jeden seed lub odsetek.

## Czego jeszcze brakuje

Baseline'y CellViT++ i HistoPLUS, sonda liniowa na embeddingach, test na prawdziwych anotacjach. Dopóki nie ma prawdziwych anotacji, żadna liczba z tej siatki nie jest wynikiem dla megakariocytów.
