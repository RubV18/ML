# Handoff — Progetto ML: Credit Score Classification (Esame Universitario ML)

## Contesto e obiettivo

Progetto per esame universitario di Machine Learning. Il professore richiede: scegliere un dataset reale online, capire il dominio, risolvere il problema richiesto usando **metriche proprie del dominio** (non applicare algoritmi a caso e confrontarli). Ha anche insegnato fairness in un altro corso, quindi apprezza l'aggiunta di un layer di governance/fairness.

Obiettivo secondario (non prioritario rispetto all'esame): rendere il progetto spendibile come case study su LinkedIn, aggiungendo XAI (SHAP) e fairness check senza compromettere la solidità del lavoro ML core.

**Priorità esplicita:** la parte ML/esame viene prima. La parte XAI/fairness è un'aggiunta con budget di tempo limitato (indicativamente non oltre ~20% del tempo totale), da non far sconfinare a scapito della difendibilità del lavoro core.

## Dataset

- **Fonte:** `parisrohan/credit-score-classification` (Kaggle)
- **Dimensioni originali:** ~100.000 righe, 28 colonne, ~12.500 clienti unici (una riga per cliente per mese, ~8 mesi ciascuno)
- **Target:** `Credit_Score` — multi-classe: `Good` / `Standard` / `Poor` (distribuzione sbilanciata, circa 18k/53k/29k sulle righe originali non aggregate)
- **IMPORTANTE:** `test.csv` fornito da Kaggle **non contiene la colonna target** — è per submission al leaderboard Kaggle, non usabile come validation/test locale. Tutti gli split (train/CV/test) devono venire esclusivamente da `train.csv`.
- Il dataset ha probabilmente colonne `Name`/`SSN` fittizie — da verificare che non contengano PII reali prima di qualunque pubblicazione (LinkedIn, repo pubblica).

## Decisioni consolidate

### Struttura dati e aggregazione
- **Drop immediato:** `ID`, `Name`, `SSN` (nessun valore predittivo)
- `Customer_ID`: mai usato come feature, serve solo per aggregazione e per lo split (evitare data leakage tra righe dello stesso cliente in train/test diversi)
- `Month`: usato solo per l'aggregazione, poi droppato
- **Aggregazione obbligatoria prima di modellare:** una riga per `Customer_ID` (moda per categoriali, mediana per numerici) → porta il dataset a ~12.500 righe. Questo evita sia il data leakage (stesso cliente in train e test) sia rispetta il vincolo dimensionale del progetto.

### Data cleaning
- Colonne numeriche sporche da stringa (`Annual_Income`, `Num_of_Loan`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Outstanding_Debt`, `Amount_invested_monthly`, `Monthly_Balance`): strip caratteri non numerici → `pd.to_numeric(errors='coerce')`
- Valori implausibili (età negativa, tassi d'interesse assurdi, ecc.) → trattati come **missing**, non cappati con IQR cieco (motivazione di dominio, non statistica automatica)
- `Credit_History_Age`: parse da testo tipo "22 Years 1 Months" a mesi totali
- Placeholder di missing mascherati come stringhe sentinella (es. `Credit_Mix = "_"`, `Payment_Behaviour` con stringhe tipo `"!@9#%8"`) → trattati esplicitamente come missing, non come categorie valide
- **Strategia missing values:** decisione per colonna in base al volume — pochi missing → imputazione (moda/mediana); molti missing → valutare drop della colonna o categoria esplicita "Unknown" (non forzare imputazione se il volume è alto)
- **Flag `is_missing_X`:** da aggiungere per le colonne dove l'assenza potrebbe essere informativa (missingness non-random / MNAR), non per tutte indiscriminatamente
- **Coerenza inter-colonna:** `Annual_Income` dovrebbe essere costante per lo stesso cliente tra i mesi pre-aggregazione — variazioni sono rumore da documentare come limite del dataset nel report finale

### Feature engineering
- **`Type_of_Loan`** (stringa con lista di prestiti concatenata): **multi-hot encoding** sui tipi principali (es. `has_payday_loan`, `has_mortgage`, `has_auto_loan`...) raggruppando i tipi rari in "Other". Non usare il semplice conteggio come sostituto (si perde il segnale sul *tipo* di debito, che nel credit risk conta più del numero — es. Payday Loan è un predittore di rischio più forte di Auto Loan). Il conteggio totale può essere tenuto come feature aggiuntiva, non sostitutiva.
- **`Occupation`** (15+ categorie): **one-hot con bucket "Other"** per le categorie rare. **Niente target encoding** — motivo: se si userà occupation anche per fairness check, il target encoding inietterebbe nella feature la stessa disparità storica che si vuole misurare, rendendo l'analisi circolare. (Nota: in questo progetto occupation NON viene usata per il fairness check — vedi sotto — ma la scelta di encoding resta valida a prescindere.)

### Split e validazione
- **80% train pool / 20% test finale**, split stratificato sul target, eseguito **dopo** l'aggregazione per cliente
- Dentro l'80% train pool: **Stratified 5-fold CV** per tutto il confronto tra modelli di riferimento (step 0) e per il tuning di iperparametri (Filone A e Filone B)
- **Test set finale toccato una sola volta**, alla fine, dopo che i modelli finali di Filone A e B sono già stati scelti tramite CV — mai usato per scegliere tra modelli o iperparametri
- Se serve calibrazione delle probabilità o tuning della soglia di decisione: usare le **predizioni out-of-fold della CV**, non un quarto blocco fisso separato (il dataset, ~12.500 righe dopo aggregazione, non è abbastanza grande da permettersi di frammentarlo ulteriormente)

### Gestione sbilanciamento classi
- **Class weighting** (pesi inversi alla frequenza nella loss) — **NO resampling/SMOTE**
- Motivo: SMOTE genera esempi sintetici interpolati; spiegare con SHAP un modello allenato in parte su dati non reali è in conflitto concettuale con l'obiettivo di governance/spiegabilità del progetto

### Metrica
- **Macro-F1** come metrica primaria, fissata **prima** di allenare qualunque modello (non scelta a posteriori in base a chi performa meglio)

### Filone A — modello più semplice sufficiente
1. **Step 0 (obbligatorio prima di scegliere):** fit di 2-3 modelli di riferimento (Logistic Regression, Random Forest, Gradient Boosting) con setting di default, confronto macro-F1 in CV
2. Se il gap di performance tra lineare e non lineare è piccolo (indicativamente <3-5 punti di macro-F1) → il problema è "lineare-sufficiente" → **Logistic Regression multinomiale con penalità L2**
3. Se il gap è grande → scegliere il modello più semplice tra quelli che raggiungono la performance necessaria (non necessariamente il più complesso in assoluto)
4. **Attenzione terminologica:** per un problema di classificazione, i termini corretti sono "Logistic Regression con L2" e "Logistic Regression con L1" — **non** "Ridge"/"Lasso", che sono termini specifici della regressione lineare continua

### Filone B — massimizza XAI
1. **Logistic Regression multinomiale con penalità L1** (sparsifica i coefficienti, feature selection automatica)
2. **Confronto diretto** con un **albero decisionale singolo, profondità limitata (3-4 livelli)** — è nativamente interpretabile senza bisogno di SHAP come layer aggiuntivo
3. **statsmodels come step diagnostico**, eseguito **dopo** la selezione L1 (sul sottoinsieme ridotto di feature sopravvissute), non su tutte le feature grezze — motivo: con feature grezze correlate (es. Annual_Income, Monthly_Inhand_Salary, Outstanding_Debt) i p-value da inferenza classica sarebbero inaffidabili per multicollinearità. Controllare VIF prima di fidarsi dei p-value anche sul sottoinsieme ridotto.
4. **Punto da verificare, non forzare:** se il modello scelto nel Filone A risulta già sufficientemente interpretabile (es. Logistic L2 semplice), non inventare un secondo modello distinto solo per rispettare la struttura del progetto — segnalarlo esplicitamente nel report come risultato genuino, non nascondere la convergenza

### Spiegabilità (XAI)
- **SHAP globale** (summary plot, dependence plot) — narrativa aggregata per il report ("il modello si basa principalmente su X, Y, Z")
- **+ 2-3 esempi locali** (waterfall/force plot): un vero positivo, una misclassificazione, un caso borderline — dimostrazione pratica del caso d'uso reale ("perché a questo cliente è stato negato/assegnato questo credit score"), rilevante sia per la difendibilità in esame sia per la spendibilità su LinkedIn

### Fairness
- **Solo età (binned)** come proxy di gruppo protetto (es. <25, 25-40, 40-60, 60+)
- **Occupation esplicitamente esclusa** dal fairness check (per evitare sia il conflitto di circolarità con l'encoding, sia il problema di sotto-gruppi troppo piccoli dopo l'aggregazione per cliente)
- Metriche di fairness da applicare: demographic parity e/o equalized odds tra fasce d'età (da definire in dettaglio quando si arriva a questa fase)

### Governance / model card
- Sezione finale del notebook/report che documenta esplicitamente: rumore residuo nei dati (vedi coerenza inter-colonna sopra), limiti del dataset, limiti del fairness check (solo età, campione non enorme), trade-off performance/interpretabilità quantificato tra Filone A e Filone B

## Ambiente

- **VS Code locale** (presumibilmente notebook Jupyter o file `.py` con celle `# %%`)
- Confidenza dichiarata dall'utente con Python/pandas: buona, già usato su progetti — non serve spiegare le basi di sintassi pandas
- Nessuna scadenza fissa — si può procedere con rigore metodologico senza tagliare scorciatoie per il tempo

## Roadmap step-by-step (ordine da seguire)

1. Domain research breve (credit scoring reale, framework normativi high-risk) — per motivare metriche e sezione di governance
2. **Data audit reale** sul CSV scaricato: `.info()`, `.describe()`, % missing per colonna, valori unici per categoriali — questo produce i valori esatti (soglie di missing, quali categorie di Occupation raggruppare in "Other") che finora sono stati decisi solo in linea teorica
3. Pipeline di cleaning + aggregazione per cliente (script riproducibile)
4. EDA (distribuzione classi, correlazioni, distribuzione età/occupation per classe — utile poi per il fairness check)
5. Split 80/20 stratificato (dopo aggregazione)
6. Step 0: fit modelli di riferimento in Stratified 5-fold CV su train pool
7. Decisione Filone A basata sui risultati di step 0
8. Tuning Filone A in CV
9. Tuning Filone B in CV (L1 + albero + statsmodels diagnostico)
10. Valutazione finale una tantum sul test set per entrambi i modelli finali (A e B)
11. SHAP globale + locale sul modello/i finale/i
12. Fairness check su età
13. Model card / sezione limiti

## Nota per l'agente che prende in carico il progetto

Tutte le decisioni sopra sono già state prese e concordate con l'utente attraverso un processo di domande guidate — non riaprire queste decisioni proponendo alternative, a meno che l'audit reale dei dati (punto 2 della roadmap) riveli un problema concreto che le renda inapplicabili (es. una colonna attesa non esiste, una percentuale di missing molto diversa da quella stimata). In quel caso, segnalarlo esplicitamente e proporre una revisione mirata solo del punto in conflitto, non dell'intero piano.
