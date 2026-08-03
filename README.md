# Credit Score Classification

Progetto di Machine Learning su `parisrohan/credit-score-classification` (Kaggle):
classificazione multi-classe del merito creditizio (`Poor` / `Standard` / `Good`)
a partire da 8 mesi di storia per cliente.

**Metrica primaria: macro-F1**, fissata prima di qualunque fit.

## Risultato in una riga

Passare da un Random Forest (300 alberi, profondità 12, ~118.000 foglie) a **un
singolo albero di profondità 4 con 14 foglie** — leggibile su una pagina,
spiegabile a un cliente senza strumenti aggiuntivi — costa **1,58 punti di
macro-F1** (0,7617 → 0,7459).

## Documenti

| | |
|---|---|
| [**REPORT.md**](reports/REPORT.md) | **Il paper completo.** Dominio e normativa, audit dei dati, teoria ML (loss/obiettivo/complessità), metodo, tutti i risultati con figure, SHAP, fairness, diagnostica, model card. Documento di riferimento unico. |
| [RISULTATI.md](reports/RISULTATI.md) | I soli numeri da citare — generato automaticamente, punto di partenza per le slide |
| [FIGURE.md](reports/FIGURE.md) | Le 10 figure selezionate per la presentazione, con didascalie |

## Come si esegue

Il dataset non è versionato: scaricare `train.csv` da Kaggle e metterlo in
`Data/train.csv`.

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/run_all.py            # pipeline completa (~20 min)
```

`run_all.py` accetta `--from 06` per ripartire da uno step o `--only 03 04` per
eseguirne alcuni. I singoli script restano eseguibili da soli e sono scritti a
celle `# %%`: si possono lanciare cella-per-cella in VS Code.

## Struttura

**Moduli condivisi** — nessuna logica di analisi, solo definizioni riusate:

| File | Responsabilità |
|---|---|
| `config.py` | Path, seed, metrica primaria, bin d'età, flag delle estensioni |
| `data_prep.py` | Cleaning riga per riga + aggregazione per cliente |
| `features.py` | Spazio delle feature, preprocessing (imputazione, scaling, one-hot) |
| `models.py` | Catalogo dei candidati: costruttori, griglie di tuning, ordine di complessità |
| `evaluation.py` | Caricamento degli split, tuning in CV, regola 1-SE |

**Step della pipeline** — uno per domanda, nell'ordine della roadmap:

| Step | Domanda a cui risponde |
|---|---|
| `step01_build_dataset` | Come si passa da 100.000 righe sporche a 12.500 clienti puliti? |
| `step02_eda` | Cosa dicono i dati prima di modellare? (solo train pool) |
| `step03_baselines` | Quanto vale la non linearità? (Step 0, modelli a default) |
| `step04_filone_a` | Qual è il modello più semplice che basta? (tuning + regola 1-SE) |
| `step05_filone_b` | Quanto lontano si arriva restando pienamente interpretabili? |
| `step06_final_test` | Quanto vale davvero, sul test toccato una sola volta? |
| `step07_shap` | Su cosa si basa il modello, globalmente e sul singolo cliente? |
| `step08_fairness` | Il modello tratta le fasce d'età allo stesso modo? |
| `step09_diagnostics` | Perché l'accuratezza si ferma al 76,9%? |
| `step10_summary` | Cosa va nel paper? (estrae il sottoinsieme presentabile) |

## Scelte metodologiche

- **Aggregazione a una riga per cliente** (mediana / moda) prima di qualunque
  split: elimina il leakage fra righe dello stesso cliente e agisce da filtro di
  rumore (dopo l'aggregazione i missing residui sono 0,00%).
- **Split 80/20 stratificato**, tuning esclusivamente in Stratified 5-fold CV sul
  train pool; **test set toccato una sola volta**, alla fine.
- **Class weighting** nella loss, niente SMOTE (spiegare con SHAP un modello
  addestrato su esempi sintetici sarebbe contraddittorio con l'obiettivo di
  governance). Verificato in `step09` su predizioni out-of-fold: guadagna 0,47
  punti di macro-F1 a parità di accuratezza.
- **Regole di scelta dichiarate prima di guardare i risultati** e implementate
  una volta sola in `evaluation.py`: one-standard-error rule per il Filone A,
  `C` più parsimonioso entro 1 SE per la L1 del Filone B.
- Valori implausibili → missing per **vincoli di dominio**, mai con capping IQR
  automatico.

## Output

- `artifacts/` — dataset a livello cliente, split, modelli serializzati, predizioni OOF
- `reports/` — tabelle CSV dei risultati, summary statsmodels, i cinque documenti
- `reports/figures/` — 19 figure: tutto, inclusi i controlli intermedi
- `reports/figures_paper/` — le 10 figure selezionate per la presentazione, con didascalie in `FIGURE.md`

## Riproducibilità

`RANDOM_STATE = 42` ovunque. Dipendenze fissate in `requirements.txt`. L'unica
sorgente di variabilità residua è il parallelismo di `n_jobs=-1`, che non
influenza i risultati riportati.
