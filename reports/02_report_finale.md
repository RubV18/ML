# Credit Score Classification — report finale

Dataset: `parisrohan/credit-score-classification` (Kaggle).
Metrica primaria: **macro-F1**, fissata prima di qualunque fit.
Codice: `src/step01…step08`. Documenti collegati:
[domain research](00_domain_research.md), [data audit](01_data_audit.md).

---

## 1. Il problema, in termini di dominio

Classificazione multi-classe del merito creditizio (`Poor` / `Standard` /
`Good`) a partire da 8 mesi di storia per cliente. Il dataset copre tutte e
cinque le famiglie informative di uno score reale (storia pagamenti, debito e
utilizzo, anzianità creditizia, mix, nuove richieste): è credit scoring vero,
non un dataset tabellare qualunque.

**Il target non è un default osservato**, ma un giudizio già prodotto da un
processo di scoring a monte. Il modello impara quindi a *replicare una
valutazione esistente*, non a predire un'insolvenza futura. Ne discende tutto il
resto del report: le feature più predittive sono in parte esse stesse output di
quel processo, e i suoi eventuali bias vengono appresi e riprodotti.

### Perché macro-F1

Le tre classi hanno costi d'errore diversi e di segno opposto (un falso `Good`
costa una perdita al creditore; un falso `Poor` nega credito a chi lo merita) e
non quantificabili senza una matrice di costi dal business. Il macro-F1 pesa le
tre classi allo stesso modo indipendentemente dalla numerosità. L'accuracy
sarebbe fuorviante: `Standard` copre il 48,9% dei clienti.

Sbilanciamento (rapporto max/min = 2,74) trattato con **class weighting** nella
loss, **non** con SMOTE: spiegare con SHAP un modello addestrato in parte su
esempi sintetici interpolati sarebbe in contraddizione con l'obiettivo di
governance del progetto.

---

## 2. Dati: da 100.000 righe a 12.500 clienti

Pipeline in `src/data_prep.py`. Dettagli e numeri in [data audit](01_data_audit.md).

1. Drop `ID`, `Name`, `SSN`.
2. Sentinelle testuali → `NaN` **prima** della conversione numerica (`__10000__`
   diventerebbe altrimenti un valore plausibile).
3. Colonne numeriche sporche → strip dei caratteri spuri → `to_numeric`.
4. `Credit_History_Age` da `"22 Years and 1 Months"` a mesi totali.
5. Valori implausibili → missing, per **vincoli di dominio** (nessun capping IQR
   cieco). Inclusa una regola inter-colonna: rata mensile > reddito mensile
   (debt-service ratio > 1) è implausibile.
6. **Aggregazione a una riga per `Customer_ID`**: mediana per i numerici, moda
   per i categoriali.

L'aggregazione fa due cose insieme: elimina alla radice il leakage fra split
(un cliente non può stare in train e test) e agisce da **filtro di rumore** —
dopo di essa i missing residui sono **0,00% su tutte le colonne**.

### Le tre scoperte dell'audit che hanno modificato il piano

| Scoperta | Effetto sul piano |
|---|---|
| Il target **varia entro cliente** (solo 41,7% dei clienti ha un label costante; 5,2% ha una parità perfetta fra due mode) | Aggiunta regola: moda + **tie-break prudenziale** sulla classe peggiore. Il label ha rumore intrinseco → tetto alla performance |
| **Zero missing** dopo l'aggregazione; la frazione di mesi mancanti non discrimina il target (15,6 / 15,1 / 14,7% per Good/Poor/Standard) | I flag `is_missing_X` previsti **non** vengono aggiunti: sarebbero costanti a zero e la missingness è MCAR per costruzione |
| Le 15 occupazioni sono **uniformemente distribuite** (~750 clienti ciascuna) | Il bucket "Other" previsto è superfluo: one-hot diretto |

Inoltre: 688 clienti (5,5%) hanno età 14–17 stabile su tutti i mesi — un vincolo
`Age ≥ 18` li avrebbe cancellati; corretto a `Age ∈ [14,100]`. Età massima
osservata 56 → **la fascia 60+ del fairness check è vuota**.

### Feature engineering

- **`Type_of_Loan` → multi-hot** su 9 tipi + conteggio dei tipi distinti come
  feature *aggiuntiva*, non sostitutiva (nel credit risk il tipo di debito conta
  quanto il numero). `NaN` = nessun prestito (verificato contro
  `Num_of_Loan == 0`) → multi-hot a zero, nessuna imputazione.
- **`Occupation` → one-hot**, niente target encoding (inietterebbe nella feature
  la stessa disparità storica che si vuole misurare).
- **Due estensioni motivate dall'audit** (disattivabili in `config.py`):
  - *dispersione* — l'audit mostra che gran parte delle colonne è costante entro
    cliente; restano davvero time-varying utilizzo, ritardi e inquiries. Per
    quelle si aggiungono `Credit_Utilization_Ratio_std`,
    `Delay_from_due_date_max`, `Num_Credit_Inquiries_max`: nel credit risk il
    *picco* di delinquenza è un segnale distinto dal suo livello medio.
  - *rapporti di dominio* — `debt_to_income` e `debt_service_ratio`.

**Split:** 80/20 stratificato **dopo** l'aggregazione → 10.000 clienti nel train
pool, 2.500 nel test. Tuning e confronti esclusivamente in **Stratified 5-fold
CV sul pool**; il test set è stato toccato una sola volta, alla fine.

---

## 3. Step 0 — quanto vale la non linearità?

Modelli di riferimento a setting di default, stessa CV, stessa metrica
(`reports/step0_baselines.csv`):

| Modello | macro-F1 (CV) | ± sd |
|---|---:|---:|
| Random Forest | **0,7415** | 0,0081 |
| Gradient Boosting (hist) | 0,7337 | 0,0056 |
| Decision Tree (depth 4) | 0,7266 | 0,0096 |
| Logistic Regression (L2) | 0,7111 | 0,0117 |
| Dummy (stratified) | 0,3362 | 0,0130 |

**Gap lineare / non lineare: 3,04 punti** — esattamente nella zona grigia
(3–5 punti) del criterio fissato a priori. La decisione non poteva quindi
essere presa sui default: è stata rinviata al confronto fra modelli *tutti*
tunati con lo stesso budget.

Già qui un risultato non banale: un **albero a profondità 4 batte la regressione
logistica** di 1,5 punti. La struttura utile del problema non è lineare, ma è
poco profonda.

---

## 4. Filone A — il modello più semplice sufficiente

Regola di scelta dichiarata prima di guardare i risultati (**one-standard-error
rule**): fra i modelli tunati si sceglie il più semplice il cui macro-F1 medio in
CV resti entro 1 errore standard dal migliore.

| Modello | macro-F1 (CV) | SE | iperparametri |
|---|---:|---:|---|
| **Random Forest** | **0,7440** | 0,0036 | `max_depth=12, max_features=0.4, min_samples_leaf=1, n_estimators=300` |
| Gradient Boosting (hist) | 0,7373 | 0,0044 | `lr=0.05, max_leaf_nodes=31, max_iter=200, l2=1.0` |
| Decision Tree | 0,7355 | 0,0051 | `max_depth=6, min_samples_leaf=100` |
| Logistic Regression (L2) | 0,7115 | 0,0053 | `C=10` |

Soglia = 0,7440 − 0,0036 = **0,7404**. Nessun modello più semplice la raggiunge.

> **Filone A = Random Forest.** Il problema **non** è lineare-sufficiente.

Va detto con onestà che con 5 fold su 10.000 clienti l'errore standard è molto
piccolo (0,0036) e la regola 1-SE degenera quasi nel "vince il migliore". Il
confronto sostanziale non va quindi cercato qui, ma fra Filone A e Filone B
(§ 6), dove la differenza di leggibilità è di un altro ordine di grandezza.

---

## 5. Filone B — massimizzare l'interpretabilità

### B1. Logistic Regression con penalità L1

Selezionare `C` sul solo massimo di macro-F1 avrebbe vanificato lo scopo della
L1 in questo filone: a `C=1` sopravvivono **57 feature su 58**. Regola coerente
con l'obiettivo del filone: si prende il `C` **più parsimonioso** che resti entro
1 SE dal migliore.

| C | n. feature attive | macro-F1 (CV) |
|---:|---:|---:|
| **0,002** | **11** | **0,7085** |
| 0,005 | 12 | 0,7066 |
| 0,03 | 27 | 0,7089 |
| 0,1 | 43 | 0,7106 |
| 1,0 | 57 | 0,7111 |

> Da 57 a **11 feature** si perde **0,26 punti** di macro-F1. È il risultato più
> netto sul fronte parsimonia: l'80% delle feature non serve.

Le 11 superstiti, in ordine di |coefficiente| (feature standardizzate):
`Credit_Mix`, `Interest_Rate`, `Num_Credit_Inquiries`, `Num_Credit_Card`,
`Delay_from_due_date`, `Changed_Credit_Limit`, `Num_of_Delayed_Payment`,
`Num_Bank_Accounts`, `Outstanding_Debt`, `Credit_History_Age_Months`.
Corrispondono alle famiglie informative di uno score reale — nessuna delle
feature ingegnerizzate sopravvive alla L1 più aggressiva.

### B2. Albero decisionale a profondità limitata

`max_depth=4, min_samples_leaf=100, criterion=entropy` → **macro-F1 CV 0,7272**,
contro 0,7085 della L1: **+1,9 punti**, e con una struttura che si legge su una
pagina (figura `08_decision_tree.png`).

> **Filone B = albero decisionale (profondità 4).** L'albero domina la L1 sia in
> performance sia in leggibilità: non serve SHAP per spiegarlo.

La prima decisione dell'albero è `Credit_Mix_Good`, la seconda si biforca su
`Outstanding_Debt` (ramo con mix buono) e `Delay_from_due_date` (ramo con mix non
buono) — cioè esattamente la logica di un analista del credito.

### B3. Diagnostica inferenziale (statsmodels)

Eseguita **dopo** la selezione L1, sul solo sottoinsieme ridotto. Prima di
guardare i p-value:

- **Collinearità strutturale** rimossa a priori: `n_loan_types` è per
  costruzione la somma esatta delle dummy `has_*`; lasciarla dentro rende la
  matrice singolare e manda tutti i VIF a infinito.
- Categoria di riferimento per ciascuna variabile categoriale.
- **VIF massimo sul sottoinsieme finale: 2,67.** Nessuna potatura ulteriore
  necessaria → i p-value sono leggibili.

Per contrasto, sul set completo il VIF di `Monthly_Inhand_Salary` è **360** e
quello di `Annual_Income` **352** (ρ di Spearman = 0,994): è esattamente il
motivo per cui l'inferenza classica va fatta *dopo* la selezione, non prima.

MNLogit (baseline `Poor`), pseudo-R² = 0,329, tutti i coefficienti tranne uno
significativi a p < 0,05. Segni coerenti col dominio: `Interest_Rate`,
`Delay_from_due_date`, `Num_Credit_Inquiries` e `Num_Credit_Card` spingono verso
`Poor`; `Credit_History_Age_Months` verso `Standard`/`Good`.

⚠️ Un'anomalia da non nascondere: `Num_of_Delayed_Payment` ha coefficiente
**positivo** verso `Standard` e `Good` (p < 0,001), cioè il segno "sbagliato".
È un effetto di soppressione dovuto alla correlazione con `Delay_from_due_date`
(che ha invece segno corretto e magnitudo maggiore): condizionatamente
all'entità del ritardo, il *numero* di ritardi correla con l'essere un cliente
attivo. Il VIF basso (1,99) non protegge da questo: il VIF misura la varianza
gonfiata, non l'interpretabilità causale del segno.

---

## 6. Valutazione finale sul test set

Test set toccato **una sola volta**, dopo che entrambi i modelli finali erano già
stati scelti in CV.

| | macro-F1 CV | **macro-F1 TEST** | accuracy | F1 Poor | F1 Standard | F1 Good |
|---|---:|---:|---:|---:|---:|---:|
| **Filone A** — Random Forest | 0,7440 | **0,7617** | 0,769 | 0,798 | 0,777 | 0,710 |
| **Filone B** — Decision Tree (d=4) | 0,7272 | **0,7459** | 0,751 | 0,780 | 0,749 | 0,709 |

La stima in CV ha retto: lo scarto CV → test è +0,018 per entrambi, coerente e
nella direzione attesa (il modello finale è rifittato su tutto il pool).

> ### Il risultato centrale del progetto
> **Il prezzo della piena trasparenza è 1,58 punti di macro-F1** (0,7617 →
> 0,7459). Passare da un ensemble di 300 alberi profondi 12 a **un singolo
> albero di profondità 4, leggibile su una pagina e spiegabile a un cliente
> senza strumenti aggiuntivi**, costa l'1,6% in metrica. Sulla classe `Good` la
> differenza è di 0,1 punti: praticamente nulla.

Per un'applicazione classificata **ad alto rischio** dall'EU AI Act, con obbligo
di motivare i rifiuti (ECOA/Reg. B, GDPR art. 22), 1,58 punti sono un prezzo che
va confrontato con il costo di dover spiegare, documentare e difendere un
ensemble. **Questo report raccomanda il modello del Filone B**, e considera il
Filone A come benchmark che quantifica quanto si sta lasciando sul tavolo.

### Dove sbaglia il modello

Matrice di confusione out-of-fold (Filone A, % per riga):

| reale ↓ / predetto → | Poor | Standard | Good |
|---|---:|---:|---:|
| **Poor** | 78,9 | 10,4 | 10,7 |
| **Standard** | 16,3 | **67,0** | 16,8 |
| **Good** | 2,2 | 8,1 | 89,7 |

L'errore si concentra su `Standard`, la classe intermedia — che è anche quella
dove il label mensile è più instabile. Confusione `Poor` ↔ `Good` quasi assente
(2,2%): il modello non sbaglia mai di due categorie.

**Quanto di questo errore è rumore del label?** I clienti classificati
correttamente hanno un accordo medio col proprio label mensile dell'**86,3%**;
quelli classificati male, del **78,6%**. Una parte dell'errore residuo non è
migliorabile: è ambiguità del target, non del modello.

---

## 7. Spiegabilità (SHAP sul modello di Filone A)

### Globale

| Feature | |SHAP| medio |
|---|---:|
| `Credit_Mix_Good` | 0,098 |
| `Outstanding_Debt` | 0,056 |
| `Interest_Rate` | 0,040 |
| `Payment_of_Min_Amount_Yes` | 0,036 |
| `Credit_Mix_Standard` | 0,033 |
| `Delay_from_due_date` | 0,029 |

Il modello si regge su **mix di credito, debito residuo, tasso applicato,
pagamento del minimo e ritardi**: le famiglie canoniche del credit scoring. Le
feature ingegnerizzate (`debt_to_income`, `Delay_from_due_date_max`) compaiono
ma con contributi di un ordine di grandezza inferiore.

⚠️ **Le due feature più importanti sono in parte output del processo che si sta
replicando.** `Credit_Mix` è una valutazione di bureau e `Interest_Rate` è il
prezzo applicato *dopo* aver valutato il merito creditizio. Il loro peso va letto
come coerenza interna col sistema di scoring esistente, non come scoperta causale
sul rischio. In produzione andrebbero verificate contro la disponibilità
effettiva al momento della decisione.

### Locale — i tre casi

| Caso | Cliente | Reale | Predetto | p |
|---|---|---|---|---:|
| A — corretto | `CUS_0x7745` | Poor | Poor | 0,98 |
| B — errore sicuro | `CUS_0x7cd3` | Standard | Poor | 0,98 |
| C — borderline | `CUS_0x613c` | Standard | Poor | 0,49 |

Il confronto **A vs B** è il più istruttivo: due clienti con **spiegazioni quasi
identiche** (stessi cinque driver, `Delay_from_due_date` +0,27 / +0,29 in testa a
entrambi) ricevono label reali diversi. Non è un errore che si corregge con più
capacità del modello: sono due profili di rischio sovrapposti a cui il processo
originale ha assegnato giudizi diversi. È il rumore del label di § 6, visto su un
singolo cliente.

Il caso C mostra il comportamento atteso in zona d'incertezza: margine 0,00 fra
prima e seconda classe, contributi tutti piccoli (< 0,06) e di segno opposto.
Operativamente è il caso che andrebbe instradato a revisione umana — la
"sorveglianza umana" richiesta dall'EU AI Act ha qui un criterio quantitativo.

---

## 8. Fairness — età

Perimetro: **solo età**, unico attributo protetto (o proxy diretto) presente.
`Occupation` esplicitamente esclusa. Metriche su predizioni out-of-fold
(n = 10.000); vista sul test in `reports/fairness_test.csv`, stesse conclusioni.

| Fascia | n | Tasso reale `Poor` | Tasso predetto `Poor` | Tasso reale `Good` | Tasso predetto `Good` |
|---|---:|---:|---:|---:|---:|
| <25 | 2.807 | 38,6% | **43,3%** | 13,0% | **20,5%** |
| 25–40 | 4.403 | 35,5% | 37,7% | 15,9% | 23,8% |
| 40–60 | 2.790 | 24,4% | **21,0%** | 25,6% | **41,2%** |

**Demographic parity gap: 22,3 punti** su `Poor`, 20,8 su `Good`.

Ma il dato importante non è il gap in sé — le classi di base differiscono già
nei dati (i giovani sono realmente più spesso `Poor` in questo dataset). Il dato
importante è la **direzione dello scarto**:

> Il modello **amplifica** la disparità esistente. Per gli under-25 predice
> `Poor` più spesso di quanto non lo siano (43,3% vs 38,6%); per i 40–60 lo
> predice meno spesso (21,0% vs 24,4%). Sulla classe `Good` l'amplificazione è
> ancora più marcata (41,2% predetto vs 25,6% reale per i 40–60).

Equalized odds, classe `Poor`:

| Fascia | TPR | FPR |
|---|---:|---:|
| <25 | 0,843 | **0,175** |
| 25–40 | 0,806 | 0,140 |
| 40–60 | 0,662 | **0,064** |

**Il tasso di falsi `Poor` è 2,7 volte più alto per gli under-25 che per i
40–60** (17,5% vs 6,4%). Questa è una violazione di equalized odds vera e
propria: non è spiegabile con le classi di base, perché è calcolata *dentro* il
gruppo dei clienti che `Poor` non sono. Tradotto in termini operativi: un
under-25 che merita credito ha quasi il triplo di probabilità di vedersi
classificato male rispetto a un over-40 nella stessa condizione.

Il gap di macro-F1 fra gruppi è invece piccolo (0,017): il modello è
*ugualmente accurato* per tutti, ma sbaglia in direzioni sistematicamente
diverse. È un caso da manuale del perché l'accuratezza aggregata non basta come
controllo di equità.

### Test di "fairness through unawareness"

Rimuovere `Age` dalle feature e rifittare:

| | con `Age` | senza `Age` | variazione |
|---|---:|---:|---:|
| DP gap (`Poor`) | 0,223 | 0,225 | +0,002 |
| EO gap TPR (`Poor`) | 0,182 | 0,175 | −0,006 |
| EO gap FPR (`Poor`) | 0,110 | 0,116 | +0,006 |
| macro-F1 OOF | 0,7440 | 0,7419 | −0,002 |

> **Rimuovere l'età non cambia nulla.** Tutte le disparità restano entro 0,006.
> La disparità non passa dalla variabile `Age`: passa dai suoi proxy —
> `Num_Credit_Inquiries` (ρ = 0,26), `Credit_History_Age_Months` (0,24, che con
> l'età è strutturalmente legata), `Interest_Rate` (0,22).

È il risultato che smonta la soluzione istintiva: cancellare l'attributo protetto
costa 0,2 punti di performance e **non produce alcun beneficio di equità**. Un
intervento reale richiederebbe vincoli espliciti in fase di addestramento o una
correzione post-hoc delle soglie per gruppo — con i relativi trade-off legali,
che questo progetto non affronta.

---

## 9. Model card e limiti

### Uso previsto
Progetto didattico. **Non utilizzabile per decisioni creditizie reali.**

### Dati
- 12.500 clienti sintetici, 8 mesi ciascuno, fonte Kaggle. Nessuna PII reale
  (`Name`, `SSN` sintetici, rimossi come primo passo insieme a `ID`).
- Il target è un giudizio pre-esistente, non un default osservato.

### Limiti dei dati
1. **Rumore del label.** Solo il 41,7% dei clienti ha un giudizio costante sugli
   8 mesi; il 5,2% ha una parità perfetta risolta con una regola prudenziale.
   Il divario di accordo fra predizioni corrette (86,3%) e sbagliate (78,6%)
   indica che una parte dell'errore residuo è irriducibile.
2. **Incoerenze inter-colonna.** 965 clienti (7,7%) hanno `Annual_Income`
   variabile fra mesi, quando dovrebbe essere costante. Assorbito dalla mediana,
   ma resta un indice della qualità del dataset.
3. **Corruzione iniettata.** Sentinelle testuali e valori assurdi (età −500 e
   8.698, redditi da 24 M, tassi al 5.797%) su 1–4,4% delle righe per colonna.
4. **Semantica di dominio non sempre rispettata.** Tutti i tipi di prestito
   mostrano un tasso `Poor` simile (42,7–44,8%, contro una base del 33,3%): il
   *tipo* di debito non discrimina, mentre nel credit risk reale un Payday Loan
   è un segnale molto più forte di un Auto Loan. Il multi-hot su `Type_of_Loan`
   è stato mantenuto (era la scelta corretta a priori) ma **empiricamente porta
   poco segnale** in questo dataset.
5. **688 clienti (5,5%) hanno 14–17 anni**, sotto l'età legale per contrarre
   credito nella quasi totalità degli ordinamenti — ulteriore conferma della
   natura sintetica dei dati.

### Limiti del modello
- Feature ad alto peso (`Credit_Mix`, `Interest_Rate`) sono plausibilmente
  output del processo di scoring che si sta replicando: il modello ha una
  componente di circolarità non eliminabile con questi dati.
- Performance stimata su un unico split 80/20 con seed fisso. Nessun intervallo
  di confidenza sul macro-F1 di test (la variabilità in CV era ±0,004–0,012).
- Nessuna calibrazione delle probabilità: `predict_proba` non va letto come
  probabilità di rischio ben calibrata.
- Nessuna validazione temporale: il panel di 8 mesi è stato collassato, quindi
  non c'è alcuna garanzia di tenuta su dati futuri (drift non testabile).

### Limiti del fairness check
- **Un solo attributo protetto** (età), l'unico disponibile. Genere, etnia,
  residenza — centrali nel credito reale — non sono nel dataset e non sono
  controllabili in alcun modo.
- **Fascia 60+ vuota** (età massima 56): l'analisi copre 3 gruppi, e proprio la
  fascia anziana, storicamente esposta a discriminazione creditizia, manca.
- Sottogruppi da 2.790 a 4.403 clienti in OOF: sufficienti per i gap riportati,
  al limite per analisi più fini (es. intersezione età × tipo di prestito).
- Le metriche misurano la **conservazione e amplificazione del bias del processo
  originale**, non l'equità in senso assoluto: il ground truth è esso stesso il
  prodotto di un giudizio potenzialmente distorto.
- Nessun intervento di mitigazione è stato implementato; l'unico testato
  (rimozione dell'attributo) è risultato inefficace.

### Trade-off dichiarato

| | Filone A | Filone B |
|---|---|---|
| Modello | Random Forest (300 alberi, prof. 12) | Albero singolo, prof. 4 |
| macro-F1 test | 0,7617 | 0,7459 |
| Interpretabilità | post-hoc (SHAP) | nativa, una pagina |
| Adverse action notice | derivabile da SHAP, con costo di calcolo e di spiegazione | leggibile direttamente dal percorso |

**Costo della trasparenza: 1,58 punti di macro-F1 (−2,1% relativo).**

### Riproducibilità
`RANDOM_STATE = 42` ovunque; `python src/step01…step08` in ordine. Dipendenze in
`requirements.txt`. L'unica sorgente di variabilità residua è il parallelismo di
`n_jobs=-1`, che non influenza i risultati riportati.
