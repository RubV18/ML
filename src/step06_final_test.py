# %%
"""Step 10 (roadmap) — valutazione finale, UNA SOLA VOLTA, sul test set.

A questo punto i modelli di Filone A e Filone B sono gia' stati scelti e tunati
usando esclusivamente la CV sul train pool. Il test set non ha influenzato
nessuna decisione: viene toccato adesso e solo per stimare la performance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config as C

pd.set_option("display.width", 200)

pool = pd.read_csv(C.ARTIFACTS / "train_pool.csv")
test = pd.read_csv(C.ARTIFACTS / "test.csv")
drop = [c for c in ["Customer_ID", C.TARGET, "_target_agreement"] if c in pool.columns]
X_pool, y_pool = pool.drop(columns=drop), pool[C.TARGET]
X_test, y_test = test.drop(columns=drop), test[C.TARGET]

meta_a = joblib.load(C.ARTIFACTS / "model_filone_a_meta.joblib")
meta_b = joblib.load(C.ARTIFACTS / "model_filone_b_meta.joblib")
MODELS = {
    f"Filone A — {meta_a['nome']}": joblib.load(C.ARTIFACTS / "model_filone_a.joblib"),
    f"Filone B — {meta_b['nome']}": joblib.load(C.ARTIFACTS / "model_filone_b.joblib"),
}

# %%
rows = []
fig, axes = plt.subplots(1, len(MODELS), figsize=(4.6 * len(MODELS), 4))
for ax, (name, model) in zip(np.atleast_1d(axes), MODELS.items()):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n{'='*70}\n{name}\nmacro-F1 sul TEST = {f1:.4f}\n{'='*70}")
    print(classification_report(y_test, y_pred, labels=C.CREDIT_SCORE_ORDER, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=C.CREDIT_SCORE_ORDER, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=C.CREDIT_SCORE_ORDER).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{name}\nmacro-F1 = {f1:.3f}", fontsize=9)

    rep = classification_report(y_test, y_pred, labels=C.CREDIT_SCORE_ORDER,
                                output_dict=True)
    rows.append({"modello": name, "macro_F1_test": f1,
                 "accuracy_test": rep["accuracy"],
                 **{f"F1_{k}": rep[k]["f1-score"] for k in C.CREDIT_SCORE_ORDER},
                 **{f"recall_{k}": rep[k]["recall"] for k in C.CREDIT_SCORE_ORDER}})
fig.suptitle("Matrici di confusione sul test set (normalizzate per riga)", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGURES / "09_confusion_test.png"); plt.close(fig)

final = pd.DataFrame(rows).set_index("modello")

# %%
# --- CV vs test: la stima in CV ha retto? ---------------------------------
final["macro_F1_cv"] = [meta_a["cv_macro_f1"], meta_b["cv_macro_f1"]]
final["scarto_cv_test"] = final["macro_F1_test"] - final["macro_F1_cv"]
print("\n\n=== RIEPILOGO FINALE ===")
print(final[["macro_F1_cv", "macro_F1_test", "scarto_cv_test", "accuracy_test"]]
      .round(4).to_string())
print("\nF1 per classe sul test:")
print(final[[f"F1_{k}" for k in C.CREDIT_SCORE_ORDER]].round(3).to_string())

trade = (final["macro_F1_test"].iloc[0] - final["macro_F1_test"].iloc[1]) * 100
print(f"\nTrade-off performance/interpretabilita' (A - B): {trade:+.2f} punti di macro-F1")
final.round(4).to_csv(C.REPORTS / "final_test_results.csv")

# %%
# --- Errori out-of-fold: dove sbaglia il modello di Filone A? -------------
# Usa le predizioni OOF della CV sul pool, non un quarto blocco separato.
cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
model_a = MODELS[list(MODELS)[0]]
oof = cross_val_predict(model_a, X_pool, y_pool, cv=cv, n_jobs=-1)
oof_proba = cross_val_predict(model_a, X_pool, y_pool, cv=cv, n_jobs=-1,
                              method="predict_proba")
np.save(C.ARTIFACTS / "oof_proba.npy", oof_proba)
pd.Series(oof, name="oof_pred").to_csv(C.ARTIFACTS / "oof_pred.csv", index=False)

print("\n\nMatrice di confusione out-of-fold (Filone A, % per riga):")
cm_oof = confusion_matrix(y_pool, oof, labels=C.CREDIT_SCORE_ORDER, normalize="true")
print(pd.DataFrame(cm_oof * 100, index=C.CREDIT_SCORE_ORDER,
                   columns=C.CREDIT_SCORE_ORDER).round(1).to_string())

# Gli errori si concentrano dove il label stesso e' instabile?
agg = pool["_target_agreement"] if "_target_agreement" in pool.columns else None
if agg is not None:
    ok = oof == y_pool
    print(f"\nStabilita' del label (accordo con la moda mensile):")
    print(f"  clienti predetti correttamente : {agg[ok].mean():.1%}")
    print(f"  clienti predetti male          : {agg[~ok].mean():.1%}")
    print("  -> se il secondo valore e' nettamente piu' basso, una parte "
          "dell'errore\n     residuo e' rumore del label, non del modello.")
