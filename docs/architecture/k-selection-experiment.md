# Eksperyment: dobór K (liczby kafli w rundzie) dla K = 5–10

Protokół zapisany 27.11.2026 **przed policzeniem K = 6–9 i seedów 10–29**. Analiza jest jednak **eksploracyjna**: wyniki dla K = 5 i K = 10
(10 seedów) były już znane z [tile-selection-experiment.md](tile-selection-experiment.md) (tam m.in. `midnight-12k` miał lepsze pokrycie niż
`lunit-dino-s16` przy małym K). Dane, modele, wejście, przestrzeń embeddingów, selekcja i 13 cech z pikseli są takie same jak w tamtym
protokole (slajd 24070_25, jeden pacjent, 2668 kafli).

> **Stan (27.11.2026): zastąpiony** przez [quota-selection-experiment.md](quota-selection-experiment.md) (zawsze 50 kafli, K to liczba
> klastrów). Obliczenia z tego protokołu zakończył job 3588517 (pierwsza próba, 3586858, przekroczyła limit czasu); wyniki leżą w
> `outputs/selection/24070_25_smallK/analysis/`, ale raportów z nich nie zbudowano i nie opisano.

## Pytania

1. Które K z {5, 6, 7, 8, 9, 10} daje najlepszy wybór kafli względem tego, co przy tym samym K daje losowanie?
2. Czy odpowiedź zależy od modelu, sposobu przygotowania embeddingów albo próbki kafli?
3. Czym różnią się zestawy kafli wybrane przez dwa najlepsze modele?

## Dlaczego nie samo pokrycie

Pokrycie (średnia odległość w cechach od każdego kafla do najbliższego wybranego, mniej = lepiej) poprawia się z każdym dodanym kaflem, więc
wybrałoby zawsze największe K. Miarą porównywaną między różnymi K jest **część drogi od losowej do pułapu**:

`G(model, K, seed) = (R(K) − pokrycie(model, K, seed)) / (R(K) − P(K))`

`R(K)`: średnie pokrycie 1000 selekcji losowych o K kaflach; `P(K)`: średnie pokrycie KMeans na cechach z pikseli (30 seedów). G = 0 oznacza
poziom losowania, G = 1 poziom pułapu. Pułap optymalizuje cechy oceny, więc jest punktem skali, nie konkurentem.

## Siatka

| wymiar | wartości | rola |
|---|---|---|
| K | 5, 6, 7, 8, 9, 10 | badane |
| model | `lunit-dino-s16`, `phikon-v2`, `midnight-12k`, `dinov2-s14`, `noise-control` | badane |
| seed KMeans | 0–29 | powtórzenia |
| przestrzeń | L2 w pełnym wymiarze (**główna**); 675 kafli bez nakładania; PCA 128 (bez kontroli szumem) | odporność |
| `n_init` | 1 (**główne**), 10 | odporność |
| nakładanie w wyborze | zakazane (**główne**), dozwolone | odporność |
| próbka kafli | wszystkie (**główne**); 20 losowych podpróbek po 80% (KMeans seed 0, pokrycie liczone na wszystkich kaflach) | odporność |
| odniesienia | losowa 1000, przestrzenna 100, KMeans na cechach 30 seedów | skala |

`R(K)` i `P(K)` są w każdym wariancie te same (wszystkie kafle, bez nakładania), żeby warianty były porównywalne.

## Reguły (zamrożone)

1. **Wybór K, osobno dla każdego modelu, w konfiguracji głównej:** najlepsze K ma najwyższe średnie G po seedach. K, którego średnie G jest niższe
   o mniej niż łączne odchylenie po seedach `sqrt((sd_a² + sd_b²) / 2)`, remisuje z najlepszym. **Zalecane jest najmniejsze K spośród
   remisujących** (mniej adnotacji). Jeśli remisują wszystkie sześć, raport stwierdza, że dane nie rozróżniają K i o K decyduje budżet adnotacji.
2. **Dwa najlepsze modele:** najwyższe G uśrednione po K = 5–10 (dla każdego seeda średnia po K, potem średnia po seedach). Remis według tej samej
   miary odchylenia wygrywa tańszy model. Kontrola szumem nie bierze udziału.
3. **Miary pomocnicze nie zmieniają wyniku reguł:** rzadkie struktury, 95. percentyl pokrycia, rozbieżność Jensena-Shannona ponad najmniejszą
   osiągalną dla danego K (przy K < 8 nie da się trafić wszystkich 8 grup), rozmieszczenie na slajdzie, stabilność między seedami (ARI, wspólne
   kafle), silhouette, Davies-Bouldin, Calinski-Harabasz, inercja, zysk K → K+1, część wspólna wyborów dla K i K+1.
4. **Warianty odporności** są opisywane tą samą regułą, ale nie służą do wybierania najlepszego ustawienia osobno dla modelu.
5. **Plansze** pokazują seed o medianowym pokryciu (dolna mediana po posortowaniu 30 seedów) dla danego modelu i K, a dla losowej losowanie o
   medianowym pokryciu. Zestawienie „przyrost K” używa jednego seeda dla wszystkich K: o medianowym pokryciu uśrednionym po K.
6. Numer kafla na planszach to pozycja jego klastra od największego; ten sam numer na planszy, mapie slajdu i mapie 2-D.

## Ograniczenia

Jeden preparat i jeden pacjent; rozrzut po seedach i podpróbkach nie mówi nic o zmienności między pacjentami. Miary to przybliżenia z pikseli.
Przy małym K miary rzadkich struktur i proporcji są mało czułe. Wybór po K kaflach w jednej rundzie nie odpowiada na pytanie o kilka kolejnych
rund Active Learning. Plansze zawierają obrazy preparatu pacjenta (do użytku wewnętrznego) i nazwy modeli: osoba oceniająca zaślepione zestawy
nie powinna ich widzieć przed oceną.

## Uruchomienie

Kod: `src/evaluation/k_analysis.py` (G i reguły), `k_cli.py` (Grid Search), `k_report.py` (raport 1), `selection_plates.py` i `plates_report.py`
(raport 2), `report_html.py`; skrypty `scripts/run_k_selection.py` i `scripts/build_k_reports.py`; testy `tests/test_k_analysis.py`,
`tests/test_k_reports.py`. Obrazy kafli nie są potrzebne do obliczeń, tylko do plansz.

Obliczenia na Atenie (dane i embeddingi z eksperymentu selekcji):

```bash
R='$SCRATCH/mpn_al_runs/selection_24070_25'; D='$SCRATCH/mpn_al_data/slide_24070_25'
bash scripts/plgrid/remote.sh sync
bash scripts/plgrid/remote.sh submit --job-name=mpn-smallk --time=02:00:00 scripts/run_k_selection.py \
    --tiles-parquet "$D/tiles.parquet" --descriptors "$R/analysis/descriptors.parquet" \
    --embeddings "dinov2-s14=$R/dinov2-s14/embeddings.parquet" --embeddings "lunit-dino-s16=$R/lunit-dino-s16/embeddings.parquet" \
    --embeddings "phikon-v2=$R/phikon-v2/embeddings.parquet" --embeddings "midnight-12k=$R/midnight-12k/embeddings.parquet" \
    --out "$R/smallK"
bash scripts/plgrid/remote.sh fetch "$R/smallK" outputs/selection/24070_25_smallK/analysis
```

Raporty na komputerze z obrazami (Edge lub Chrome do PDF):

```bash
O=outputs/selection/24070_25_smallK
python scripts/build_k_reports.py --results $O/analysis --out $O --images <prep>/slides/24070_25/tiles \
    --embeddings "<model>=$O/embeddings/<model>/embeddings.parquet" ... --notes-k $O/notes_k.md --notes-visual $O/notes_wizualne.md
```

Przed uruchomieniem sprawdzono, że pobrane embeddingi odtwarzają na laptopie dokładnie te same selekcje dla K = 5 i 10 (seedy 0–9, 4 modele),
co opublikowane w eksperymencie selekcji.
