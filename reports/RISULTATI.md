# Risultati chiave

Estratto automatico dagli output della pipeline (`src/step10_summary.py`).
Numeri da citare nel paper e nelle slide; il ragionamento completo sta in
[REPORT.md](REPORT.md), il paper completo.

## Il risultato centrale

> Il prezzo della piena trasparenza e' **1.58 punti di macro-F1**.
> Random Forest (300 alberi, ~118.000 foglie): **0.7617**.
> Albero singolo di profondita' 4 (14 foglie): **0.7459**.

## 1. Confronto fra modelli (5-fold CV sul train pool)

| modello                  |   macro_F1 |     SE |
|:-------------------------|-----------:|-------:|
| Random Forest            |     0.744  | 0.0036 |
| Gradient Boosting (hist) |     0.7373 | 0.0044 |
| Decision Tree            |     0.7355 | 0.0051 |
| SVC (kernel RBF)         |     0.7229 | 0.0046 |
| Logistic Regression (L2) |     0.7115 | 0.0053 |
| Linear SVC               |     0.7074 | 0.005  |

## 2. Prestazioni finali (test set, toccato una sola volta)

| modello                  |   macro_F1_cv |   macro_F1_test |   accuracy_test |
|:-------------------------|--------------:|----------------:|----------------:|
| Filone A — Random Forest |        0.744  |          0.7617 |          0.7688 |
| Filone B — Decision Tree |        0.7272 |          0.7459 |          0.7508 |

## 3. Parsimonia: quanto costa ridurre le feature (L1)

|     C |   n_feature |   macro_F1_cv |
|------:|------------:|--------------:|
| 0.002 |          11 |        0.7085 |
| 0.005 |          12 |        0.7066 |
| 0.01  |          17 |        0.7082 |
| 0.03  |          27 |        0.7089 |
| 0.1   |          43 |        0.7106 |
| 0.3   |          53 |        0.7102 |
| 1     |          57 |        0.7111 |

## 4. Fairness per fascia d'eta' (out-of-fold, n=10.000)

| gruppo   |    n |   base_rate_Poor |   sel_rate_Poor |   TPR_Poor |   FPR_Poor |
|:---------|-----:|-----------------:|----------------:|-----------:|-----------:|
| <25      | 2807 |            0.386 |           0.433 |      0.843 |      0.175 |
| 25-40    | 4403 |            0.355 |           0.377 |      0.806 |      0.14  |
| 40-60    | 2790 |            0.244 |           0.21  |      0.662 |      0.064 |

Il tasso di falsi 'Poor' e' **2.7 volte** piu' alto per la fascia piu' giovane rispetto alla piu' anziana.

### Rimuovere l'eta' non serve

|                       |   con Age |   senza Age |   variazione |
|:----------------------|----------:|------------:|-------------:|
| DP gap (Poor)         |    0.2232 |      0.2249 |       0.0018 |
| EO gap TPR (Poor)     |    0.1816 |      0.1753 |      -0.0063 |
| EO gap FPR (Poor)     |    0.1103 |      0.1159 |       0.0056 |
| DP gap (Standard)     |    0.0235 |      0.0239 |       0.0004 |
| EO gap TPR (Standard) |    0.0304 |      0.0322 |       0.0019 |
| EO gap FPR (Standard) |    0.0206 |      0.0246 |       0.004  |
| DP gap (Good)         |    0.2077 |      0.2095 |       0.0018 |
| EO gap TPR (Good)     |    0.0639 |      0.0616 |      -0.0023 |
| EO gap FPR (Good)     |    0.1279 |      0.1303 |       0.0024 |
| gap macro-F1          |    0.017  |      0.0134 |      -0.0037 |

## 5. Perche' l'accuratezza si ferma al 77%

**Non e' la numerosita'** — curva di apprendimento piatta:

|   n_train |   cv_macro_F1 |
|----------:|--------------:|
|      1200 |        0.7274 |
|      2333 |        0.7356 |
|      3466 |        0.7353 |
|      4600 |        0.7366 |
|      5733 |        0.7407 |
|      6866 |        0.7416 |
|      8000 |        0.744  |

**Non e' rigore metodologico pagato in performance** — l'aggregazione per cliente batte sia lo split per riga sia la versione con leakage:

|                                       |   accuracy |   macro_F1 |
|:--------------------------------------|-----------:|-----------:|
| riga (casuale) — CON leakage          |     0.707  |     0.7001 |
| riga (per cliente) — SENZA leakage    |     0.6825 |     0.6786 |
| riga + Customer_ID fra le feature     |     0.7083 |     0.7011 |
| cliente (aggregato) — questo progetto |     0.7688 |     0.7617 |

**E' il rumore del label**: il giudizio originale, sullo stesso cliente e con feature quasi identiche, cambia nel 15,6% dei mesi. Solo il 41,7% dei clienti ha un giudizio costante sugli 8 mesi.

## Figure

Selezione per le slide in [FIGURE.md](FIGURE.md) e `reports/figures_paper/`.