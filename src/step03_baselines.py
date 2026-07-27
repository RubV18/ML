# %%
"""Step 0 (roadmap) — modelli di riferimento a setting di default.

Scopo: misurare il gap lineare / non lineare PRIMA di scegliere il modello del
Filone A. Serve a rispondere a una domanda ("questo problema e' lineare-
sufficiente?"), non a fare una gara fra algoritmi.

Metrica primaria: macro-F1, fissata prima di qualunque fit.
Sbilanciamento: class weighting nella loss, niente resampling/SMOTE.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as C
from features import make_preprocessor
from models import logistic_l2

pd.set_option("display.width", 200)

pool = pd.read_csv(C.ARTIFACTS / "train_pool.csv")
X = pool.drop(columns=[c for c in ["Customer_ID", C.TARGET, "_target_agreement"]
                       if c in pool.columns])
y = pool[C.TARGET]
cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

# %%
MODELS = {
    "Dummy (stratified)": DummyClassifier(strategy="stratified",
                                          random_state=C.RANDOM_STATE),
    "Logistic Regression (L2)": logistic_l2(),
    "Decision Tree (depth 4)": DecisionTreeClassifier(
        max_depth=4, class_weight="balanced", random_state=C.RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting (hist)": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=C.RANDOM_STATE),
}

# %%
rows = []
for name, clf in MODELS.items():
    # Lo scaling serve solo ai lineari; per gli alberi e' inerte.
    scale = isinstance(clf, LogisticRegression)
    pipe = Pipeline([("prep", make_preprocessor(X, scale=scale)), ("clf", clf)])
    t0 = time.perf_counter()
    res = cross_validate(pipe, X, y, cv=cv,
                         scoring=["f1_macro", "balanced_accuracy", "accuracy"],
                         n_jobs=-1)
    rows.append({
        "modello": name,
        "macro_F1": res["test_f1_macro"].mean(),
        "std": res["test_f1_macro"].std(),
        "bal_acc": res["test_balanced_accuracy"].mean(),
        "accuracy": res["test_accuracy"].mean(),
        "sec": time.perf_counter() - t0,
    })
    print(f"{name:28s} macro-F1 = {rows[-1]['macro_F1']:.4f} "
          f"(+/- {rows[-1]['std']:.4f})   [{rows[-1]['sec']:.1f}s]")

res_df = pd.DataFrame(rows).set_index("modello").sort_values("macro_F1", ascending=False)
print("\n" + res_df.round(4).to_string())
res_df.to_csv(C.REPORTS / "step0_baselines.csv")

# %%
# --- La domanda dello Step 0: quanto vale la non linearita'? --------------
lin = res_df.loc["Logistic Regression (L2)", "macro_F1"]
best_nl = res_df.drop(index=["Dummy (stratified)", "Logistic Regression (L2)",
                             "Decision Tree (depth 4)"])["macro_F1"].max()
gap = (best_nl - lin) * 100
print(f"\nmacro-F1 lineare (LogReg L2)      : {lin:.4f}")
print(f"macro-F1 migliore non lineare     : {best_nl:.4f}")
print(f"GAP                               : {gap:.2f} punti di macro-F1")
print("\nCriterio deciso a priori: gap < 3-5 punti -> problema 'lineare-sufficiente'"
      " -> Filone A = Logistic Regression L2.")
print("ESITO:", "LINEARE-SUFFICIENTE" if gap < 3 else
      ("ZONA GRIGIA (3-5 punti)" if gap < 5 else "NON LINEARE NECESSARIO"))
