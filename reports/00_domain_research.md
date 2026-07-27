# Domain research — credit scoring: cosa vincola le scelte di modellazione

Nota breve, serve a motivare metrica, modelli e sezione di governance. Non è
una rassegna di letteratura.

## 1. Cosa fa davvero un credit score

Un credit score sintetizza il rischio che un soggetto non onori un'obbligazione
creditizia. Nel mondo reale è prodotto da bureau (FICO, VantageScore, CRIF in
Italia) a partire da cinque famiglie di informazioni, con pesi noti e stabili:

| Famiglia | Peso tipico FICO | Corrispettivo nel dataset |
|---|---|---|
| Storia dei pagamenti | ~35% | `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Payment_of_Min_Amount` |
| Ammontare del debito / utilizzo | ~30% | `Outstanding_Debt`, `Credit_Utilization_Ratio` |
| Lunghezza della storia creditizia | ~15% | `Credit_History_Age` |
| Mix di credito | ~10% | `Credit_Mix`, `Type_of_Loan` |
| Nuove richieste di credito | ~10% | `Num_Credit_Inquiries` |

Il dataset copre tutte e cinque le famiglie: è quindi un problema di credit
scoring genuino, non un dataset tabellare qualunque. Questo giustifica di
leggere i risultati con il lessico del dominio (delinquenza, utilizzo,
debt-to-income) invece che solo con quello statistico.

## 2. Conseguenze operative sulla scelta della metrica

Le tre classi non hanno lo stesso costo d'errore, ma i costi vanno in direzioni
opposte e non sono quantificabili con i dati a disposizione:

- classificare come **Good** un cliente realmente **Poor** → perdita attesa su
  credito erogato (costo diretto per il creditore);
- classificare come **Poor** un cliente realmente **Good** → mancato ricavo,
  e soprattutto un danno per il richiedente (accesso negato o pricing peggiore).

Senza una matrice di costi fornita dal business, la scelta difendibile è una
metrica che **non privilegia nessuna classe per la sua sola numerosità**: da qui
il **macro-F1**, fissato prima di qualunque fit. L'accuracy sarebbe fuorviante
(la sola classe `Standard` copre ~49% dei clienti: un modello banale che predice
sempre `Standard` avrebbe accuracy 49% e macro-F1 22%).

Per lo stesso motivo lo sbilanciamento si tratta con **class weighting** nella
loss e non con resampling.

## 3. Perché l'interpretabilità qui non è un optional

Il credit scoring è una delle applicazioni più regolate del machine learning:

- **EU AI Act** — la valutazione del merito creditizio delle persone fisiche è
  esplicitamente classificata **ad alto rischio** (Allegato III, punto 5b). Ne
  discendono obblighi di gestione del rischio, data governance, documentazione
  tecnica, trasparenza verso l'utilizzatore e sorveglianza umana.
- **GDPR art. 22 + considerando 71** — diritto a non essere sottoposti a
  decisioni interamente automatizzate con effetti giuridici significativi, e
  diritto a ottenere una spiegazione della logica applicata.
- **ECOA / Regulation B (USA)** — obbligo di *adverse action notice*: al cliente
  rifiutato vanno comunicate le ragioni specifiche e principali del rifiuto. La
  CFPB ha chiarito (circolare 2022-03) che l'uso di modelli complessi non esime
  dall'obbligo.
- **Basilea / EBA** — sui modelli interni si richiede che i driver di rischio
  siano economicamente spiegabili, non solo statisticamente performanti.

Due conseguenze dirette sul progetto:

1. La struttura a due filoni (modello *più semplice sufficiente* vs modello
   *massimamente interpretabile*) non è un esercizio accademico: è la scelta che
   un'istituzione deve realmente motivare, e il **prezzo in performance della
   trasparenza va quantificato**, non assunto.
2. Le spiegazioni locali (SHAP waterfall su un singolo cliente) sono la forma
   tecnica di un obbligo giuridico — la *adverse action notice*, non un
   abbellimento del report.

## 4. Fairness: perché l'età

Nel credito, età e reddito sono le due dimensioni su cui la discriminazione è
storicamente documentata e normativamente vietata. ECOA vieta esplicitamente la
discriminazione per età (fatta salva la capacità di contrarre); l'EU AI Act
richiede, per i sistemi ad alto rischio, esami dei bias sui dataset di training.

L'età è l'unico attributo protetto (o suo proxy diretto) presente nel dataset,
il che rende il perimetro del check una conseguenza dei dati, non una scelta di
comodo. Il limite va dichiarato: nessun controllo è possibile su genere, etnia o
residenza, che nel credito reale sono altrettanto rilevanti.

## 5. Ciò che questo dataset **non** è

Il target `Credit_Score` non è un evento di default osservato: è una
**classificazione già prodotta da un processo di scoring a monte**. Il modello
impara quindi a **replicare un giudizio esistente**, non a predire un'insolvenza
futura. Due implicazioni serie:

- Feature come `Credit_Mix` e `Interest_Rate` sono a loro volta **output** di
  quel processo (il tasso applicato dipende dal merito creditizio già valutato).
  Il loro peso elevato va letto come coerenza interna con il sistema di scoring,
  non come scoperta causale sul rischio.
- Qualunque bias del processo originale viene appreso e riprodotto. È
  esattamente la ragione per cui un fairness check è dovuto, e per cui va
  interpretato come misura di *conservazione del bias*, non di *equità reale*.
