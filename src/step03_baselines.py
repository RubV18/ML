# %%
"""Step 0 (roadmap) — modelli di riferimento a setting di default.

Scopo: misurare il gap lineare / non lineare PRIMA di scegliere il modello del
Filone A. Risponde a una domanda diagnostica ("questo problema e'
lineare-sufficiente?"), non a fare una gara fra algoritmi — per questo i modelli
girano coi parametri di default, al costo piu' basso possibile.

Se il gap fosse risultato piccolo (<3 punti) ci si sarebbe fermati qui, senza
spendere il tuning completo dello step successivo.

Metrica primaria: macro-F1, fissata prima di qualunque fit.
Sbilanciamento: class weighting nella loss, niente resampling/SMOTE.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

import config as C
from evaluation import build_pipeline, load_split, make_cv
from models import CANDIDATES, LINEAR_MODELS

pd.set_option("display.width", 200)

X, y = load_split("pool")
cv = make_cv()

# %%
# I candidati sono quelli dichiarati in models.CANDIDATES: lo Step 0 e il tuning
# del Filone A guardano cosi' la stessa famiglia, per costruzione.
rows = []
for name, spec in {"Dummy (stratified)": None, **CANDIDATES}.items():
    if spec is None:
        est, scale, label = (DummyClassifier(strategy="stratified",
                                             random_state=C.RANDOM_STATE),
                             False, name)
    else:
        est = spec.get("default", spec["estimator"])
        scale, label = spec["scale"], spec.get("default_label", name)

    res = cross_validate(build_pipeline(est, X, scale), X, y, cv=cv,
                         scoring=["f1_macro", "balanced_accuracy", "accuracy"],
                         n_jobs=-1)
    rows.append({"modello": label,
                 "macro_F1": res["test_f1_macro"].mean(),
                 "sd": res["test_f1_macro"].std(),
                 "bal_acc": res["test_balanced_accuracy"].mean(),
                 "accuracy": res["test_accuracy"].mean(),
                 "_famiglia": name})
    print(f"{label:28s} macro-F1 = {rows[-1]['macro_F1']:.4f} "
          f"(+/- {rows[-1]['sd']:.4f})")

res_df = pd.DataFrame(rows).set_index("modello").sort_values("macro_F1",
                                                             ascending=False)
print("\n" + res_df.drop(columns="_famiglia").round(4).to_string())
res_df.drop(columns="_famiglia").to_csv(C.REPORTS / "step0_baselines.csv")

# %%
# --- La domanda dello Step 0: quanto vale la non linearita'? --------------
lin = res_df[res_df["_famiglia"].isin(LINEAR_MODELS)]["macro_F1"].max()
nonlin = res_df[~res_df["_famiglia"].isin(LINEAR_MODELS | {"Dummy (stratified)"})]
best_nl = nonlin["macro_F1"].max()
gap = (best_nl - lin) * 100

print(f"\nmigliore lineare      : {lin:.4f}")
print(f"migliore non lineare  : {best_nl:.4f}  ({nonlin['macro_F1'].idxmax()})")
print(f"GAP                   : {gap:.2f} punti di macro-F1")
print("\nCriterio deciso a priori: gap < 3-5 punti -> problema "
      "'lineare-sufficiente' -> Filone A = Logistic Regression L2.")
esito = ("LINEARE-SUFFICIENTE" if gap < 3
         else "ZONA GRIGIA (3-5 punti)" if gap < 5
         else "NON LINEARE NECESSARIO")
print("ESITO:", esito)
if gap >= 3:
    print("\n-> La diagnosi sui default non e' conclusiva: la scelta passa allo\n"
          "   step04, dove tutti i candidati sono tunati con lo stesso budget.")
