# Data audit — `Data/train.csv`

Audit eseguito sul CSV reale prima di scrivere la pipeline. Serve a sostituire
con numeri le decisioni prese finora solo in linea teorica, e a segnalare i punti
in cui i dati contraddicono il piano iniziale.

Riprodotto da `src/step01_build_dataset.py` (diagnostiche D1, D1b, D2, D3).

## 1. Struttura

| | |
|---|---|
| Righe | 100.000 |
| Colonne | 28 |
| Clienti unici (`Customer_ID`) | 12.500 |
| Righe per cliente | esattamente 8 (gennaio–agosto), per tutti |
| Target | `Credit_Score` ∈ {Poor, Standard, Good} |
| Distribuzione (righe) | Standard 53.174 / Poor 28.998 / Good 17.828 |
| Distribuzione (clienti, post-aggregazione) | Standard 48,9% / Poor 33,3% / Good 17,8% |

Il panel è perfettamente bilanciato: nessun cliente ha mesi mancanti. Confermata
la scelta di aggregare a una riga per cliente (12.500 righe finali).

## 2. Valori sentinella — verificati, non ipotizzati

Il dataset codifica i missing come stringhe, non come `NaN`:

| Colonna | Sentinella | Righe | % |
|---|---|---:|---:|
| `Credit_Mix` | `_` | 20.195 | 20,2% |
| `Payment_of_Min_Amount` | `NM` | 12.007 | 12,0% |
| `Payment_Behaviour` | `!@9#%8` | 7.600 | 7,6% |
| `Occupation` | `_______` | 7.062 | 7,1% |
| `SSN` | `#F%$D@*&8` | 5.572 | 5,6% |
| `Amount_invested_monthly` | `__10000__` | 4.305 | 4,3% |
| `Changed_Credit_Limit` | `_` | 2.091 | 2,1% |
| `Monthly_Balance` | `__-3333…33__` | 9 | <0,1% |

⚠️ `__10000__` è la sentinella più insidiosa: ripulita dai caratteri spuri
diventa `10000.0`, un valore numerico perfettamente plausibile. Se la si
coercisse prima di rimuoverla, il 99° percentile di `Amount_invested_monthly`
diventerebbe esattamente 10.000 — un artefatto puro. Nella pipeline le
sentinelle sono quindi rimosse **prima** della conversione numerica.

Altre colonne numeriche sono sporcate da un `_` finale (`Age` → `28_`,
`Annual_Income` → `14388.79_`, ecc.): 1.000–7.000 righe ciascuna.

## 3. Valori implausibili

Trattati come **missing**, in base a vincoli di dominio, non a un capping IQR
automatico. Percentuale di righe nullificata da ciascuna regola:

| Regola | Righe nullificate |
|---|---:|
| `Num_of_Loan` ∉ [0, 20] | 4,35% |
| `Total_EMI_per_month` > reddito mensile (DSR > 1) | 3,04% |
| `Age` ∉ [14, 100] | 2,78% |
| `Num_Credit_Card` > 20 | 2,26% |
| `Interest_Rate` ∉ [1, 50] | 2,03% |
| `Num_Credit_Inquiries` > 50 | 1,63% |
| `Num_of_Delayed_Payment` ∉ [0, 60] | 1,38% |
| `Num_Bank_Accounts` ∉ [0, 20] | 1,34% |
| `Annual_Income` > 1.000.000 | 0,96% |

Ordini di grandezza dei valori scartati: `Age` fino a 8.698 e fino a −500,
`Interest_Rate` fino a 5.797%, `Annual_Income` fino a 24,2 milioni,
`Num_Bank_Accounts` fino a 1.798. Sono corruzioni iniettate, non outlier reali.

Nessuna regola nullifica più del 4,4% delle righe: nessun vincolo si è rivelato
troppo aggressivo.

## 4. Scoperte che hanno modificato il piano

### 4.1 Il target NON è costante nel tempo per lo stesso cliente ⚠️

Il piano dava per scontato che aggregare per cliente fosse un'operazione neutra
sul target. Non lo è:

| | Clienti | % |
|---|---:|---:|
| Label identico su tutti gli 8 mesi | 5.208 | 41,7% |
| 2 label distinti | 7.262 | 58,1% |
| 3 label distinti | 30 | 0,2% |
| **Parità perfetta fra due mode** | **655** | **5,2%** |

Accordo medio con la moda mensile: 84,4% (minimo 37,5%).

**Regola aggiunta:** il target del cliente è la **moda** sugli 8 mesi; in caso di
parità si assegna la **classe peggiore** (`Poor` > `Standard` > `Good`), per
principio prudenziale del credit risk — in dubbio, non sottostimare il rischio.
Riguarda 5,2% dei clienti.

Conseguenza sostanziale: **il label ha rumore intrinseco**, e questo pone un
tetto alla performance raggiungibile. Verificato a valle (§ report finale): i
clienti classificati male dal modello hanno un accordo medio col proprio label
del 78,6%, contro 86,3% di quelli classificati bene.

### 4.2 Il dataset contiene clienti minorenni ⚠️

688 clienti (5,5%) hanno età fra 14 e 17 anni, **stabile su tutti e 8 i mesi** —
quindi generata, non rumore. Un primo vincolo `Age ≥ 18` (età legale per
contrarre credito) li avrebbe cancellati tutti. Il vincolo è stato corretto a
`Age ∈ [14, 100]`.

Età massima osservata: 56 → **la fascia 60+ del fairness check è vuota**;
l'analisi opera di fatto su tre gruppi.

### 4.3 Dopo l'aggregazione non restano missing

Per **ogni** colonna e **ogni** cliente esiste almeno un mese con valore valido.
La mediana per cliente assorbe integralmente la corruzione riga-per-riga:

**missing residui dopo aggregazione: 0,00% su tutte le 39 colonne.**

Due conseguenze sul piano iniziale:

- **I flag `is_missing_X` non vengono aggiunti.** Sarebbero costanti a zero.
  La versione sensata a livello cliente — la *frazione* di mesi mancanti — è
  stata testata e non discrimina (D3): frazione media di mesi mancanti per
  `Monthly_Inhand_Salary` = 15,6% / 15,1% / 14,7% rispettivamente per Good /
  Poor / Standard. La missingness è MCAR per costruzione: nessun segnale.
- L'imputer resta nella pipeline solo per robustezza su dati nuovi.

### 4.4 Il bucket "Other" per `Occupation` è superfluo

Le 15 occupazioni reali sono distribuite quasi uniformemente (5.885–6.575 righe
ciascuna, ≈ 750 clienti a categoria). Non esistono categorie rare da
raggruppare: one-hot diretto su 15 categorie, nessun bucket "Other". Il
sentinella `_______` (7,1%) sparisce con la moda per cliente.

### 4.5 `Type_of_Loan`: missing strutturale, non da imputare

`Type_of_Loan` è **costante entro cliente** (0 clienti con più di un valore
distinto) → il multi-hot a livello cliente è privo di ambiguità.

Gli 11,4% di `NaN` corrispondono a `Num_of_Loan == 0` (10.930 righe su 11.408):
è **assenza di prestiti**, non informazione mancante. Diventano multi-hot tutto
a zero, senza imputazione.

Nove tipi di prestito, ciascuno presente in ~38–40 mila righe, più il token
`Not Specified` (39.616) trattato come categoria a sé — è informazione sulla
qualità della disclosure, non un missing.

## 5. Rumore inter-colonna da documentare

`Annual_Income` dovrebbe essere costante entro cliente: **965 clienti (7,7%)**
mostrano valori diversi fra i mesi. Idem `Occupation` (5.550 clienti prima della
rimozione della sentinella) e `Age` (8.307 prima della pulizia). Dopo pulizia e
aggregazione per mediana/moda il problema si risolve, ma resta un limite di
qualità del dataset da dichiarare nella model card.

`Credit_History_Age` è invece perfettamente coerente: per tutti e 12.500 i
clienti la differenza fra età della storia creditizia e indice del mese è
costante (incrementa di 1 mese al mese). Parsata da testo a mesi totali.

## 6. PII

`Name` e `SSN` sono sintetici (gli `SSN` seguono il formato `xxx-xx-xxxx` con
valori generati; 5.572 righe hanno la sentinella `#F%$D@*&8`). Vengono comunque
**rimossi insieme a `ID` come primo passo della pipeline**: nessun valore
predittivo e nessuna ragione di trattarli.
