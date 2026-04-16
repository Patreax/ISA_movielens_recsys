# Mini-projekt 2 — Steam Recommendation System
## Kompletný popis systému

---

## 1. Čo sme robili a prečo

Cieľom mini-projektu 2 je postaviť **personalizovaný odporúčací systém** na datasete Steam recenzií pomocou **neurónovej siete** (Neural Matrix Factorization — NeuMF).

Projekt nadväzuje na mini-projekt 1 (klasická kolaboratívna filtrácia na MovieLens), ale musí použiť:
- **iný dataset** → Steam Review and Bundle Dataset
- **iný prístup** → neurónová sieť namiesto klasickej maticovej faktorizácie

---

## 2. Dataset

### Zdroj
HuggingFace: `recommender-system/steam-review-and-bundle-dataset`  
Australskí Steam používatelia (geograficky obmedzený vzorek).

### Päť súborov

| Súbor | Obsah | Počet záznamov |
|---|---|---|
| `australian_user_reviews.json.gz` | Recenzie používateľov + binárny signál `recommend` | ~59 000 |
| `australian_users_items.json.gz` | Herná knižnica každého používateľa + čas hrania (minúty) | ~5 100 000 |
| `steam_games.json.gz` | Metadáta hier: žánre, tagy, cena, vydavateľ, dátum vydania | ~32 000 |
| `bundle_data.json.gz` | Balíčky hier (615 balíčkov) | 615 |
| `steam_reviews.json.gz` | Skrapovanné recenzie s hodinami hrania a dátumom | ~7 800 000 |

### Čo je primárny interakčný signál
Stĺpec `recommend` (True/False) z `australian_user_reviews`. Ak používateľ odporučil hru → pozitívna interakcia. Negatívy (hry, ktoré hrál ale nerecenzoval) sampľujeme náhodne pri trénovaní.

---

## 3. Architektúra systému (pipeline)

```
Raw JSON.gz súbory
        │
        ▼
Parquet cache  (data/interim/steam/*.parquet)
        │
        ▼
EDA (sekcie A–J v notebooku)
        │
        ▼
Data Preprocessing (filtrovanie, encoding, split)
        │
        ▼
NeuMF trénovanie (GMF + MLP vetvy)
        │
        ▼
Evaluácia & Ablation study
```

---

## 4. Caching a načítavanie dát

### Prečo cache?
Parsovanie 7.8M JSON riadkov trvá niekoľko minút. Po prvom spustení sa výsledok uloží do Parquet formátu (stĺpcový binárny formát — rýchle čítanie, 5–10× menšia veľkosť).

### Logika (project2/dataset.py)
```
Existuje parquet a je novší ako .json.gz?
    ÁNO → načítaj z parquet (sekundy)
    NIE → parsuj JSON.gz, ulož do parquet, vráť DataFrame
```

### Technický detail — list stĺpce
Stĺpce `genres`, `tags`, `specs` v `steam_games` sú pôvodne Python listy. PyArrow ich pri zápise do parquet uloží ako **list-type stĺpce**. Pri spätnom načítaní PyArrow 23+ ich vracia ako `numpy.ndarray` objekty. Preto pri každej operácii, kde pandas potrebuje hashovať hodnoty (`nunique()`, `duplicated()`, `value_counts()`), musíme tieto stĺpce ošetriť špeciálne alebo konvertovať `ndarray → list`.

---

## 5. Exploratory Data Analysis (EDA)

EDA pokrýva 10 sekcií (A–J). Cieľ: pochopiť dáta pred modelovaním a identifikovať problémy.

### A. Dataset Overview
- Tvar každého datasetu (počet riadkov, stĺpcov)
- Granularita: čo predstavuje jeden riadok
- Detekcia duplikátov a chýbajúcich kľúčov
- Klasifikácia stĺpcov (ID, numerické, kategorické, boolean, list/array, text)

### B. Data Quality Assessment
- Chýbajúce hodnoty (null%) pre každý stĺpec → `df_games` má najviac nullov (metascore, sentiment)
- Heatmapa chýbajúcich hodnôt
- Duplicátna analýza (s ochranou pred list stĺpcami)
- Pokrytie `item_id` naprieč súbormi: ~30% hier z recenzií **nemá** metadata v `df_games` → cold-start problém pre side features

### C. Data Preprocessing
Každý krok je zdokumentovaný v preprocessing summary tabuľke:

| Stĺpec | Problém | Riešenie |
|---|---|---|
| `posted` | Textový dátum "Posted November 5, 2011." | Parsovanie regex + `pd.to_datetime` → `posted_date` |
| `helpful` | Hlasy v stringu "15 people found this helpful" | Extrakcia čísla → `helpful_votes` |
| `playtime_forever` | Silne pravostranná distribúcia (skewness > 10) | `log1p` transformácia + cap na 99.9 percentil |
| `genres`/`tags`/`specs` | Uložené ako str repr listu `"['Action', ...]"` | `ast.literal_eval` → Python list |
| `price` | Zmiešané typy: `"4.99"`, `"Free To Play"`, `None` | Parsovanie → `price_numeric`; `is_free` flag |
| `review` | Prázdne reťazce | `review_len`, `has_text` flag |

### D. Univariate Analysis
**Aktivita používateľov:**
- Distribúcia počtu recenzií na používateľa — silná power-law: väčšina napísala 1–3, malé percento stovky
- Distribúcia počtu hier na používateľa — podobný vzor
- Distribúcia herného času — log-normálna distribúcia (typické pre herné správanie)

**Metadáta hier:**
- Top žánre: Indie, Action, Casual dominujú počtom
- Distribúcia cien: väčšina hier stojí 0–20 USD, mediána ~10 USD
- Vydávanie hier: výrazný nárast od 2012, peak okolo 2016

### E. Bivariate Analysis
**Kľúčové zistenia:**
1. **Čas hrania vs odporúčanie**: Odporúčané hry sú hrané ~3× dlhšie (v log škále). Silný signál.
2. **Dĺžka recenzie vs odporúčanie**: Negatívne recenzie sú mierne dlhšie (violin plot). Slabý signál.
3. **Žáner vs miera odporúčania**: RPG a Strategy majú vyššiu mieru ako Free-to-Play. Žánre nesú prediktívnu informáciu.
4. **Temporálny trend**: Objem recenzií rástol 2013–2015. Miera odporúčania zostala stabilná ~88–92%.

### F. Multivariate Analysis
- Pearsonova korelačná matica user-level features
  - `n_games` a `total_playtime` sú silne korelované (r≈0.7) → redundantné
  - `pos_rate` (miera odporúčania) nie je korelovaná s objemom hier → nezávislý signál
- Sparsita interakčnej matice: **<0.01%** density → extrémne riedka matica

### G. Outlier & Anomaly Analysis
- IQR metóda na `playtime_forever`: ~8% outlierov (horná strana), max 50 000+ hodín
  - **Záver**: Ide o power userov (platné správanie), nie chyby. Pre lineárne modely capnúť, pre embedding modely ponechať.
- Dĺžka recenzií: recenzie >10 000 znakov existujú (platné, len nie typické)

### H. Target Analysis (Cieľová premenná)
- **88% recenzií je pozitívnych** (`recommend=True`)
- Silná pozitívna zaujatosť: ľudia recenzujú prevažne hry, ktoré odporúčajú
- Na používateľskej úrovni: 45% používateľov **nikdy** neodporučilo (pos_rate=1.0) — títo nenesú diferenciálny signál
- Implikácia: Klasická accuracy by bola ľahko 88% aj s "odporuč všetko" baseline → metriky ranking sú dôležitejšie

### I. Feature Usefulness Review
Point-biserial korelácia s `recommend`:

| Feature | r | Záver |
|---|---|---|
| `log(playtime+1)` | ~0.25 | Stredný signál — viac hrania → pravdepodobnejšie odporúčanie |
| `review_len` | ~-0.05 | Slabý signál — dlhšie recenzie mierne negatívnejšie |
| `helpful_votes` | ~0.01 | Žiadny signál — drop |

### J. Business / Domain Insight Summary
1. Silný pozitívny bias → používaj ranking metriky (NDCG, Hit@K)
2. Sparsita matice → klasické metódy zlyhávajú, potrebujeme embeddingy
3. Čas hrania je najsilnejší explicitný signál
4. Žánre nesú obsahovú informáciu → vhodné ako side features
5. Geograficky obmedzený dataset → výsledky nemusia generalizovať globálne

---

## 6. Data Preparation for Modeling

### Filtrovanie (project2/modeling/data_prep.py)
Pracujeme len s **pozitívnymi interakciami** (`recommend=True`).

**Iteratívne core filtrovanie:**
```
Opakuj až do konvergencie (max 10 iterácií):
    1. Ponech len používateľov s ≥5 recenziami
    2. Ponech len hry s ≥10 recenziami
```
Prečo iteratívne? Po vyfiltrovaní vzácnych hier môžu niektorí používatelia klesnúť pod limit a naopak.

**Po filtrovaní:**
- 7 552 interakcií
- 1 254 používateľov
- 211 hier
- 22 žánrov

### ID Encoding
Embeddingy v PyTorch (`nn.Embedding`) potrebujú **súvislé celé čísla od 0**.

```python
user_to_idx = {"76561197970982479": 0, "765611979...": 1, ...}
item_to_idx = {"10": 0, "730": 1, ...}
```

Pôvodné Steam ID (64-bit číslo) → lokálny index. Spätné mapovanie `idx_to_item` sa používa pri generovaní odporúčaní.

### Genre Multi-Hot Matrix
Shape: `(n_items=211, n_genres=22)`, dtype `float32`

Pre každú hru: vektor 22 hodnôt, kde `1.0` = hra má daný žáner, `0.0` = nemá.

```
           Action  Casual  Indie  RPG  ...
hra_0  →  [1.0,    0.0,    1.0,  0.0, ...]
hra_1  →  [0.0,    1.0,    1.0,  0.0, ...]
```

Táto matica sa predáva do MLP vetvy NeuMF ako **side features** (statické — nie sú trénované, ale prechádzajú cez learnable projekčnú vrstvu).

### Leave-One-Out Split (temporálny)
Štandardný protokol z NCF papiera (He et al. 2017):

```
Pre každého používateľa, zoraď interakcie podľa dátumu:
    posledná interakcia        → TEST set
    predposledná interakcia    → VALIDATION set
    zvyšok                     → TRAIN set
```

**Prečo temporálne?** Aby sme simulovali reálny scenár — model nevidí "budúce" interakcie pri trénovaní. Priestorový (náhodný) split by bol dátový leak.

**Výsledky:**
- Train: 5 044 interakcií
- Val: 1 254 (jedna na používateľa)
- Test: 1 254 (jedna na používateľa)

### Negative Sampling
Implicit feedback problém: nevidíme skutočné negatívne preferencie, len chýbajúce interakcie.

**Postup:**
```
Pre každú pozitívnu (user, item) dvojicu:
    Náhodne vyber 4 hry, ktoré používateľ NEhral
    Označ ich label=0 (predpokladaná negatívna preferencia)
```

**Ratio 4:1** je štandardný z literatúry. Väčší pomer znižuje recall.

**Kľúčové:** Negativy sa **re-sampleujú každú epochu** s iným seed-om. Fixed negativy by spôsobili overfitting — model by sa naučil presne tieto negatívne páry.

---

## 7. Modely

### 7.1 Popularity Baseline (nulová personalizácia)

Skóruje každú hru podľa frekvencie v trénovacej sade:

```
score(item) = počet interakcií s itemom / celkový počet interakcií
```

Každý používateľ dostane rovnaké odporúčania — top najpopulárnejšie hry, ktoré ešte nehral. Slúži ako dolná hranica — ak model nevybehne nad tento baseline, niečo je zle.

---

### 7.2 GMF-only (Generalized Matrix Factorization)

**Čo robí:** Diferenciálna verzia klasickej maticovej faktorizácie (podobné mini-projektu 1).

**Architektúra:**
```
user_id → Embedding(n_users, 32) → u ∈ ℝ³²
item_id → Embedding(n_items, 32) → v ∈ ℝ³²
                   ↓
        element-wise product: gmf = u ⊙ v ∈ ℝ³²
                   ↓
         Linear(32 → 1) → sigmoid → predikcia [0,1]
```

**Čo sa naučia embeddingy?**
Embedding je lookup tabuľka — pre každého používateľa a hru existuje vektor náhodne inicializovaných čísel. Počas trénovania backpropagácia aktualizuje len tie riadky tabuľky, ktoré boli použité v danom batchi. Po trénovaní platí: **používatelia s podobným vkusom majú podobné embedding vektory** (pretože dostávali podobné gradienty od podobných hier).

Element-wise product `u ⊙ v` je matematicky ekvivalentný skalárneho súčinu v klasickom MF, ale tu má každá dimenzia vlastnú váhu cez finálnu lineárnu vrstvu — preto "Generalized".

**Obmedzenie:** Len lineárne interakcie — nedokáže zachytiť nelineárne vzory (napr. "rád strategy A AJ RPG B, ale nie kombináciu oboch").

---

### 7.3 MLP-only (Multi-Layer Perceptron)

**Čo robí:** Nelineárna alternatíva — embedding stĺpce sa konkatenujú a prechádzajú cez FC vrstvy.

**Architektúra:**
```
user_id → Embedding(n_users, 64) → u ∈ ℝ⁶⁴
item_id → Embedding(n_items, 64) → v ∈ ℝ⁶⁴

[voliteľné] genres ∈ ℝ²²
          → Linear(22 → 32) + ReLU → g ∈ ℝ³²

concat([u, v]) ∈ ℝ¹²⁸  (alebo concat([u, v, g]) ∈ ℝ¹⁶⁰ s genres)
    → Linear(128→128) + ReLU
    → Linear(128→64)  + ReLU
    → Linear(64→32)   + ReLU
    → Linear(32→1)
    → sigmoid → predikcia [0,1]
```

**Väčšie embeddingy (64 vs 32 v GMF):** MLP potrebuje viac priestoru na zakódovanie používateľov/hier, pretože musí reprezentovať nelineárne vzory. GMF vystačí s menšími embedding vektormi, lebo jeho vzorce sú jednoduché (element-wise product).

**Side features (žánre):** Žánrový vektor prechádza projekčnou vrstvou `Linear(22→32)` pred konkatenáciou. Pomáha pri **cold-start** — aj nová hra s nulovými interakciami má žánrové tagy.

---

### 7.4 NeuMF (Neural Matrix Factorization) — Hlavný model

**Referencia:** He et al. (2017). *Neural Collaborative Filtering.* WWW 2017.

**Kľúčová myšlienka:** Kombinuj výhody GMF (lineárne vzory) aj MLP (nelineárne vzory) — každý typ vzoru zachyť separátnou vetvou, výstupy skombinovania cez Fusion vrstvu.

#### Kompletná architektúra

```
                         INPUT
              user_id              item_id
                 │                    │
    ┌────────────┼──────────┐    ┌────┼────────────┐
    │ GMF vetva  │          │    │    │  MLP vetva  │
    │            │          │    │    │             │
    │  Emb(32)◄──┘          │    │    └──►Emb(64)  │
    │  user_gmf             │    │       item_mlp  │
    │                       │    │                 │
    │  Emb(32)◄─────────────┼────┼──►Emb(64)      │
    │  item_gmf             │    │   user_mlp      │
    │       │               │    │       │         │
    │  u⊙v ∈ ℝ³²           │    │  genres(22)     │
    │       │               │    │    ↓            │
    │       │               │    │  Linear(22→32)  │
    │       │               │    │    ↓ ReLU       │
    │       │               │    │  g ∈ ℝ³²        │
    │       │               │    │       │         │
    │       │               │    │  concat(u,v,g)  │
    │       │               │    │  ∈ ℝ¹⁶⁰        │
    │       │               │    │    ↓ FC+ReLU   │
    │       │               │    │  ∈ ℝ¹²⁸        │
    │       │               │    │    ↓ FC+ReLU   │
    │       │               │    │  ∈ ℝ⁶⁴         │
    │       │               │    │    ↓ FC+ReLU   │
    │  gmf_out ∈ ℝ³²        │    │  mlp_out ∈ ℝ³² │
    └───────┼───────────────┘    └───────┼─────────┘
            │                           │
            └────────┬──────────────────┘
                     │
              concat ∈ ℝ⁶⁴
                     │
              Linear(64→1)
                     │
                  sigmoid
                     │
             predikcia [0,1]
```

#### Prečo oddelené embedding tabuľky pre GMF a MLP?

GMF obmedzuje interakcie na lineárne (element-wise product). Ak by MLP zdieľalo rovnaké embeddingy, gradienty z MLP (ktoré sa učia nelineárne vzory) by "rušili" jednoduché gradients z GMF a naopak. Separátne tabuľky umožňujú každej vetve nezávisle sa optimalizovať pre svoju úlohu.

#### Celkový počet parametrov (NeuMF + genres)
- GMF user emb: 1254 × 32 = 40 128
- GMF item emb: 211 × 32 = 6 752
- MLP user emb: 1254 × 64 = 80 256
- MLP item emb: 211 × 64 = 13 504
- Genre proj: 22 × 32 + 32 = 736
- MLP vrstvy: (160→128) + (128→64) + (64→32) ≈ 29 312
- Fusion: 64 → 1 = 65
- **Celkom: ~172 000 parametrov** — veľmi malý model

---

## 8. Trénovanie

### Loss funkcia: Binary Cross-Entropy (BCE)

```
BCE = -[y · log(ŷ) + (1-y) · log(1-ŷ)]
```

Pre pozitívnu interakciu (y=1) chceme ŷ → 1.  
Pre negatívnu (y=0) chceme ŷ → 0.

Prečo BCE a nie MSE? Predikujeme pravdepodobnosť (0/1 výstup cez sigmoid), nie spojitú hodnotu. BCE je prirodzená loss pre binárnu klasifikáciu.

### Optimizer: Adam (lr=1e-3)
Adam adaptuje learning rate pre každý parameter zvlášť. Vhodný pre embeddingy — väčšina parametrov sa neaktualizuje v každom batchi (len tie embedding riadky, ktoré boli v batchi).

### Trénovacia slučka

```
Pre každú epochu (max 20):
    1. Re-sample negativy (seed = epoch číslo)
       → iné negatívy každú epochu = lepšia generalizácia
    2. Shuffle dát
    3. Pre každý batch (512 príkladov):
          a. Forward pass → predikcie
          b. BCE loss
          c. Backprop → gradienty
          d. Adam.step() → aktualizuj parametre
    4. Evaluuj na val sete → NDCG@10
    5. Ak NDCG@10 lepší ako doteraz:
          → ulož checkpoint (best_state_dict)
          → reset patience counter
       Inak:
          → patience_counter += 1
          → ak counter ≥ 4: zastav (early stopping)
6. Načítaj najlepší checkpoint
```

### Early stopping
Zastaví trénovanie keď validačné NDCG@10 prestane rásť. Zabraňuje overfittingu — model sa naučí trénovacie dáta naspamäť, ale zhoršuje sa na nevídených dátach.

---

## 9. Evaluácia

### Protokol: Leave-One-Out
Pre každého testovacieho používateľa:
1. Vezmi jeho jedinú test interakciu (hru, ktorú "chceme odporučiť")
2. Zostav kandidátske hry = všetky hry, ktoré tento používateľ **nevidel** počas trénovania
3. Každú kandidátsku hru ohodnoť modelom → forward pass
4. Zoraď hry podľa skóre zostupne → ranking
5. Skontroluj, kde sa nachádza test hra v rankingu

### Metriky pri K = 5, 10, 20

**Hit@K** — "Je test hra v top-K?"
```
Hit@K = 1  ak je test hra v top K výsledkoch
Hit@K = 0  inak
```
Agregát cez všetkých používateľov = priemerná hit rate.

**NDCG@K** — Normalized Discounted Cumulative Gain
```
NDCG@K = 1 / log₂(rank + 1)  ak je test hra v top K
NDCG@K = 0                    inak
```
Hra na 1. mieste = skóre 1.0. Na 2. mieste = 0.63. Na 10. mieste = 0.29.  
NDCG penalizuje nízky rank viac ako Hit@K — **kvalita rankingu je dôležitejšia ako len prítomnosť v top-K**.

**Precision@K**
```
Precision@K = počet relevantných v top-K / K
```
Pre leave-one-out (jeden test item na používateľa) = Hit@K / K.

### Prečo tieto metriky a nie accuracy?
- Dataset má 88% pozitívnych recenzií → accuracy je zavádzajúca
- Cieľ odporúčacieho systému nie je "predikovať rating" ale "zoradiť hry dobre"
- NDCG je štandardná metrika v RecSys literatúre (NCF paper, BPR, atď.)

---

## 10. Výsledky — Ablation Study

Porovnávame 5 variantov na test sete (leave-one-out):

| Model | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 | Hit@20 | NDCG@20 |
|---|---|---|---|---|---|---|
| Popularity Baseline | 0.190 | 0.123 | 0.284 | 0.154 | 0.354 | 0.172 |
| GMF-only | 0.187 | 0.122 | 0.285 | 0.154 | 0.349 | 0.170 |
| MLP-only (no genres) | 0.195 | 0.131 | 0.280 | 0.158 | 0.353 | 0.177 |
| NeuMF (no genres) | 0.192 | 0.130 | 0.282 | 0.159 | 0.355 | 0.178 |
| **NeuMF + genres** | **0.191** | **0.129** | **0.286** | **0.160** | **0.354** | **0.177** |

### Interpretácia výsledkov

**Prečo sú rozdiely malé?**  
Dataset má len 211 hier po filtrovaní. Pri 211 kandidátoch je náhoda silný "konkurent" — aj Popularity baseline sa "trafí" do top-10 s ~28% pravdepodobnosťou. S väčším počtom kandidátov (napr. 10 000+) by rozdiely medzi modelmi boli výraznejšie.

**GMF-only ≈ Popularity baseline**  
Lineárna CF nedokáže zachytiť vzory nad popularitu pri tak malom datasete. Interakcie sú prevažne lineárne — každá hra je buď populárna alebo nie.

**MLP a NeuMF prekonávajú GMF**  
Nelineárne vzory (kombinácie žánrov, používateľské preferencie) mierne pomáhajú. NeuMF + genres má najlepší NDCG@10 (0.160 vs 0.154 baseline) — žánrové side features pridávajú marginálnu hodnotu.

**Záver:** Pri takomto malom počte položiek (211) sú všetky modely relatívne blízko sebe. Výsledok je realistický a pri väčšom datasete by boli rozdiely jasnejšie.

---

## 11. Ukážka odporúčaní (NeuMF + genres)

Pre 5 náhodných používateľov:

| Používateľ | Top-5 odporúčaní |
|---|---|
| user_1197 | Counter-Strike: GO, Garry's Mod, Left 4 Dead 2, Dota 2, PAYDAY 2 |
| user_633 | Team Fortress 2, CS:GO, Garry's Mod, Left 4 Dead 2, Dota 2 |
| user_222 | Team Fortress 2, CS:GO, Garry's Mod, Left 4 Dead 2, Dota 2 |
| user_708 | Team Fortress 2, CS:GO, Garry's Mod, Dota 2, PAYDAY 2 |
| user_948 | CS:GO, Team Fortress 2, Garry's Mod, Dota 2, Left 4 Dead 2 |

Odporúčania dávajú zmysel — najpopulárnejšie Valve hry s veľkými komunitami. Pri malom datasete (211 hier) model prirodzene konverguje k populárnym titulom.

---

## 12. Štruktúra kódu

```
project2/
├── config.py          — cesty (PROJ_ROOT, DATA_DIR, STEAM_DATA_DIR)
├── dataset.py         — načítanie a caching JSON.gz → parquet
├── plots.py           — EDA vizualizačné funkcie (reusable)
└── modeling/
    ├── data_prep.py   — filtrovanie, encoding, split, negative sampling, Dataset
    ├── model.py       — PopularityBaseline, GMFOnly, MLPOnly, NeuMF
    ├── train.py       — trénovacia slučka s early stopping
    └── evaluate.py    — Hit@K, NDCG@K, Precision@K (leave-one-out)

notebooks/
└── 02_steam_recsys.ipynb   — hlavný notebook (47 buniek)

data/interim/steam/
├── australian_user_reviews.parquet
├── australian_users_items.parquet
├── steam_games.parquet
├── bundle_data.parquet
└── steam_reviews.parquet
```

---

## 13. Kľúčové rozhodnutia a ich zdôvodnenie

| Rozhodnutie | Hodnota | Prečo |
|---|---|---|
| Interakčný signál | Len `recommend=True` | Binary feedback; negatívy sú neobsevované páry |
| Filtrovanie | ≥5 recenzií/user, ≥10/item | Rovnováha medzi pokrytím a hustotou matice |
| Negative ratio | 4:1 | Štandard v NCF literatúre; vyšší pomer znižuje recall |
| GMF embedding dim | 32 | He et al. 2017 odporúčanie |
| MLP embedding dim | 64 | Väčšie embeddingy pre nelineárny priestor |
| MLP vrstvy | 128→64→32 | Postupné zmenšovanie = bežná prax |
| Early stopping | patience=4, val NDCG@10 | NDCG je primárna ranking metrika |
| Genre projection | 22→32 dim | Mapuje sparse multi-hot na hustý priestor |
| Split | Leave-one-out temporálny | Zabraňuje data leakage; štandardný protokol |

---

## 14. Referencie

1. **He et al. (2017).** *Neural Collaborative Filtering.* WWW 2017. https://arxiv.org/abs/1703.04730
   - Primárna referencia pre NeuMF architektúru
2. **d2l.ai Chapter 21.6** — NeuMF implementačná referencia: https://d2l.ai/chapter_recommender-systems/neumf.html
3. **Kang & McAuley (2018).** *Self-attentive sequential recommendation.* ICDM 2018. — citácia datasetu
4. **Pathak, Gupta & McAuley (2017).** *Generating and personalizing bundle recommendations on Steam.* SIGIR 2017. — citácia bundle datasetu
