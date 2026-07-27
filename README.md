# Credit Score Classification

Progetto di Machine Learning su `parisrohan/credit-score-classification` (Kaggle):
classificazione multi-classe del merito creditizio (`Poor` / `Standard` / `Good`)
a partire da 8 mesi di storia per cliente.

**Metrica primaria: macro-F1**, fissata prima di qualunque fit.

## Risultato in una riga

Passare da un Random Forest (300 alberi, profondità 12) a **un singolo albero
di profondità 4** — leggibile su una pagina, spiegabile a un cliente senza
strumenti aggiuntivi — costa **1,58 punti di macro-F1** (0,7617 → 0,7459).

## Documenti

| | |
|---|---|
| [Domain research](reports/00_domain_research.md) | Perché macro-F1, perché l'interpretabilità è vincolante (EU AI Act, GDPR art. 22, ECOA) |
| [Data audit](reports/01_data_audit.md) | Audit sul CSV reale, sentinelle, valori implausibili, scoperte che hanno modificato il piano |
| [**Report finale**](reports/02_report_finale.md) | Metodo, risultati, SHAP, fairness, model card |

## Pipeline

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/step01_build_dataset.py   # cleaning, aggregazione, split 80/20
./.venv/bin/python src/step02_eda.py             # EDA (solo sul train pool)
./.venv/bin/python src/step03_baselines.py       # Step 0: gap lineare / non lineare
./.venv/bin/python src/step04_filone_a.py        # Filone A: tuning + regola 1-SE  (~14 min)
./.venv/bin/python src/step05_filone_b.py        # Filone B: L1, albero, statsmodels
./.venv/bin/python src/step06_final_test.py      # test set, toccato una sola volta
./.venv/bin/python src/step07_shap.py            # SHAP globale + 3 casi locali
./.venv/bin/python src/step08_fairness.py        # fairness su età + unawareness test
```

I file `src/step*.py` sono script a celle `# %%`: eseguibili da terminale o
cella-per-cella in VS Code. Moduli condivisi: `config.py` (costanti e seed),
`data_prep.py` (cleaning + aggregazione), `features.py` (preprocessing),
`models.py` (costruttori dei modelli lineari).

## Scelte metodologiche

- **Aggregazione a una riga per cliente** (mediana / moda) prima di qualunque
  split: elimina il leakage fra righe dello stesso cliente e agisce da filtro di
  rumore (dopo l'aggregazione i missing residui sono 0,00%).
- **Split 80/20 stratificato**, tuning esclusivamente in Stratified 5-fold CV sul
  train pool; **test set toccato una sola volta**, alla fine.
- **Class weighting** nella loss, niente SMOTE (spiegare con SHAP un modello
  addestrato su esempi sintetici sarebbe contraddittorio con l'obiettivo di
  governance).
- **Regole di scelta dichiarate prima di guardare i risultati**: one-standard-error
  rule per il Filone A, `C` più parsimonioso entro 1 SE per la L1 del Filone B.
- Valori implausibili → missing per **vincoli di dominio**, mai con capping IQR
  automatico.

## Output

- `artifacts/` — dataset a livello cliente, split, modelli serializzati, predizioni OOF
- `reports/` — tabelle CSV dei risultati, summary statsmodels, i tre documenti
- `reports/figures/` — 18 figure (EDA, albero, matrici di confusione, SHAP, fairness)
