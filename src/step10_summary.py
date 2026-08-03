# %%
"""Step 14 — sintesi per il paper / la presentazione.

Gli step precedenti producono tutto cio' che serve a *difendere* il lavoro:
19 figure e 15 tabelle, molte delle quali sono controlli intermedi. In una
presentazione servono i risultati, non i controlli.

Questo step non ricalcola nulla: legge gli output gia' prodotti e ne estrae il
sottoinsieme che regge una tesi.

Produce:
  reports/RISULTATI.md       — una pagina con i soli numeri da citare
  reports/figures_paper/     — le figure selezionate, rinominate in ordine di
                               esposizione, con didascalie in FIGURE.md
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import config as C

OUT_FIG = C.REPORTS / "figures_paper"
OUT_FIG.mkdir(exist_ok=True)

# %%
# --- Le figure che reggono un'affermazione, in ordine di racconto ---------
# Ogni voce e' (file sorgente, nome in uscita, cosa dimostra).
# Le figure escluse non sono sbagliate: sono controlli che vivono nel report
# lungo e appesantirebbero una presentazione.
FIGURE = [
    ("01_target_distribution.png", "01_problema_tre_classi.png",
     "Il problema: tre classi sbilanciate (49/33/18%). Motiva macro-F1 e class weighting."),
    ("03_mutual_information.png", "02_feature_informative.png",
     "Quali variabili portano segnale: mix di credito, tasso, debito, ritardi — "
     "le famiglie canoniche di uno score reale."),
    ("05_age_and_target.png", "03_eta_e_target.png",
     "Distribuzione dell'eta' e composizione del target per fascia: la base del "
     "fairness check, e la disparita' gia' presente nei dati."),
    ("07b_l1_path.png", "04_parsimonia_L1.png",
     "Percorso di regolarizzazione L1: da 57 a 11 feature si perdono 0,26 punti. "
     "L'80% delle variabili non serve."),
    ("08_decision_tree.png", "05_albero_finale.png",
     "Il modello del Filone B per intero: profondita' 4, 14 foglie. "
     "La prima domanda che pone e' il mix di credito, come farebbe un analista."),
    ("09_confusion_test.png", "06_confusione_test.png",
     "Prestazioni sul test set per entrambi i filoni. L'errore si concentra su "
     "'Standard'; confusione Poor<->Good quasi assente (2,2%)."),
    ("10_shap_global_bar.png", "07_shap_globale.png",
     "Su cosa si basa il Random Forest, per classe. Le due feature principali "
     "sono pero' output del processo di scoring a monte."),
    ("13_shap_local_B_errore.png", "08_shap_caso_errore.png",
     "Un cliente 'Standard' classificato 'Poor' con p=0,98. La spiegazione e' "
     "quasi identica a quella di un vero 'Poor': rumore del label, non del modello."),
    ("14_fairness_parity.png", "09_fairness_eta.png",
     "Tasso reale vs predetto per fascia d'eta': il modello *amplifica* la "
     "disparita' esistente in entrambe le direzioni."),
    ("15_learning_curve.png", "10_curva_apprendimento.png",
     "La curva e' piatta: il limite non e' la numerosita' del campione."),
]

righe = []
for src, dst, caption in FIGURE:
    s = C.FIGURES / src
    if not s.exists():
        print(f"  ATTENZIONE: manca {src} (esegui prima lo step che la produce)")
        continue
    shutil.copy2(s, OUT_FIG / dst)
    righe.append(f"### {dst}\n\n![{dst}](figures_paper/{dst})\n\n{caption}\n")

(C.REPORTS / "FIGURE.md").write_text(
    "# Figure selezionate per la presentazione\n\n"
    "Sottoinsieme di `reports/figures/` (19 figure) che regge un'affermazione "
    "della tesi, in ordine di esposizione.\n\n" + "\n".join(righe))
print(f"{len(righe)} figure copiate in {OUT_FIG.name}/")

# %%
# --- I numeri da citare ---------------------------------------------------
def leggi(nome, **kw):
    p = C.REPORTS / nome
    return pd.read_csv(p, **kw) if p.exists() else None


tuning = leggi("filone_a_tuning.csv", index_col=0)
finale = leggi("final_test_results.csv", index_col=0)
sparsity = leggi("filone_b_l1_sparsity.csv")
fair = leggi("fairness_oof.csv", index_col=0)
unaware = leggi("fairness_unawareness.csv", index_col=0)
lc = leggi("diagnostica_t3_learning_curve.csv")
leak = leggi("diagnostica_t1_leakage.csv", index_col=0)

A, B = finale.index[0], finale.index[1]
trade = (finale.loc[A, "macro_F1_test"] - finale.loc[B, "macro_F1_test"]) * 100

md = [
    "# Risultati chiave",
    "",
    "Estratto automatico dagli output della pipeline (`src/step10_summary.py`).",
    "Numeri da citare nel paper e nelle slide; il ragionamento completo sta in",
    "[REPORT.md](REPORT.md), il paper completo.",
    "",
    "## Il risultato centrale",
    "",
    f"> Il prezzo della piena trasparenza e' **{trade:.2f} punti di macro-F1**.",
    f"> Random Forest (300 alberi, ~118.000 foglie): **{finale.loc[A,'macro_F1_test']:.4f}**.",
    f"> Albero singolo di profondita' 4 (14 foglie): **{finale.loc[B,'macro_F1_test']:.4f}**.",
    "",
    "## 1. Confronto fra modelli (5-fold CV sul train pool)",
    "",
    tuning[["macro_F1", "SE"]].round(4).to_markdown(),
    "",
    "## 2. Prestazioni finali (test set, toccato una sola volta)",
    "",
    finale[["macro_F1_cv", "macro_F1_test", "accuracy_test"]].round(4).to_markdown(),
    "",
    "## 3. Parsimonia: quanto costa ridurre le feature (L1)",
    "",
    sparsity.round(4).to_markdown(index=False),
    "",
    "## 4. Fairness per fascia d'eta' (out-of-fold, n=10.000)",
    "",
    fair[["n", "base_rate_Poor", "sel_rate_Poor", "TPR_Poor", "FPR_Poor"]]
    .round(3).to_markdown(),
    "",
    f"Il tasso di falsi 'Poor' e' **{fair['FPR_Poor'].max()/fair['FPR_Poor'].min():.1f} volte** "
    "piu' alto per la fascia piu' giovane rispetto alla piu' anziana.",
    "",
    "### Rimuovere l'eta' non serve",
    "",
    unaware.round(4).to_markdown(),
    "",
    "## 5. Perche' l'accuratezza si ferma al 77%",
    "",
    "**Non e' la numerosita'** — curva di apprendimento piatta:",
    "",
    lc[["n_train", "cv_macro_F1"]].round(4).to_markdown(index=False),
    "",
    "**Non e' rigore metodologico pagato in performance** — l'aggregazione per "
    "cliente batte sia lo split per riga sia la versione con leakage:",
    "",
    leak.round(4).to_markdown(),
    "",
    "**E' il rumore del label**: il giudizio originale, sullo stesso cliente e "
    "con feature quasi identiche, cambia nel 15,6% dei mesi. Solo il 41,7% dei "
    "clienti ha un giudizio costante sugli 8 mesi.",
    "",
    "## Figure",
    "",
    "Selezione per le slide in [FIGURE.md](FIGURE.md) e `reports/figures_paper/`.",
]

(C.REPORTS / "RISULTATI.md").write_text("\n".join(md))
print(f"scritto {C.REPORTS / 'RISULTATI.md'}")
print(f"\ntrade-off A vs B: {trade:.2f} punti di macro-F1")
