# %%
"""Step 8 (roadmap) — Filone A: il modello piu' SEMPLICE che basta.

Lo Step 0 ha dato un gap lineare/non lineare di ~3 punti di macro-F1: zona
grigia. La decisione non puo' quindi essere presa sui default, ma su modelli
*tutti* tunati con lo stesso budget e la stessa CV.

Regola di scelta dichiarata PRIMA di guardare i risultati (one-standard-error
rule, standard in letteratura): fra i modelli tunati, si sceglie il piu'
semplice il cui macro-F1 medio in CV sia entro 1 errore standard dal migliore.
Se nessun modello semplice rientra, vince il migliore in assoluto.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as C
from features import make_preprocessor
from models import COMPLEXITY_RANK, logistic_l2

pd.set_option("display.width", 200)

pool = pd.read_csv(C.ARTIFACTS / "train_pool.csv")
X = pool.drop(columns=[c for c in ["Customer_ID", C.TARGET, "_target_agreement"]
                       if c in pool.columns])
y = pool[C.TARGET]
cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

# %%
SEARCHES = {
    "Logistic Regression (L2)": (
        logistic_l2(), True,
        {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=C.RANDOM_STATE), False,
        {"clf__max_depth": [3, 4, 5, 6, 8, 12, None],
         "clf__min_samples_leaf": [1, 10, 50, 100]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=C.RANDOM_STATE,
                               n_jobs=1), False,
        {"clf__n_estimators": [300, 600],
         "clf__max_depth": [None, 12, 20],
         "clf__min_samples_leaf": [1, 3, 10],
         "clf__max_features": ["sqrt", 0.4]},
    ),
    "Gradient Boosting (hist)": (
        HistGradientBoostingClassifier(class_weight="balanced",
                                       random_state=C.RANDOM_STATE), False,
        {"clf__learning_rate": [0.05, 0.1],
         "clf__max_leaf_nodes": [15, 31, 63],
         "clf__max_iter": [200, 400],
         "clf__l2_regularization": [0.0, 1.0]},
    ),
}

# %%
results, fitted = {}, {}
for name, (clf, scale, grid) in SEARCHES.items():
    pipe = Pipeline([("prep", make_preprocessor(X, scale=scale)), ("clf", clf)])
    t0 = time.perf_counter()
    gs = GridSearchCV(pipe, grid, scoring=C.PRIMARY_METRIC, cv=cv, n_jobs=-1,
                      refit=True)
    gs.fit(X, y)
    # media e deviazione standard fra i fold della configurazione migliore
    i = gs.best_index_
    results[name] = {
        "macro_F1": gs.cv_results_["mean_test_score"][i],
        "std_folds": gs.cv_results_["std_test_score"][i],
        "se": gs.cv_results_["std_test_score"][i] / np.sqrt(C.N_FOLDS),
        "complessita": COMPLEXITY_RANK[name],
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "sec": time.perf_counter() - t0,
    }
    fitted[name] = gs.best_estimator_
    print(f"{name:26s} macro-F1 = {results[name]['macro_F1']:.4f} "
          f"(SE {results[name]['se']:.4f})  {results[name]['best_params']}  "
          f"[{results[name]['sec']:.0f}s]")

res = pd.DataFrame(results).T
res.to_csv(C.REPORTS / "filone_a_tuning.csv")

# %%
# --- Applicazione della regola dichiarata a priori ------------------------
best_name = res["macro_F1"].idxmax()
threshold = res.loc[best_name, "macro_F1"] - res.loc[best_name, "se"]
eligible = res[res["macro_F1"] >= threshold].sort_values("complessita")
chosen = eligible.index[0]

print(f"\nmigliore in assoluto : {best_name} ({res.loc[best_name,'macro_F1']:.4f})")
print(f"soglia (best - 1 SE) : {threshold:.4f}")
print("modelli entro soglia :", list(eligible.index))
print(f"\n>>> FILONE A = {chosen}  (il piu' semplice entro 1 SE dal migliore)")
print(f"    macro-F1 CV = {res.loc[chosen,'macro_F1']:.4f}  "
      f"iperparametri = {res.loc[chosen,'best_params']}")

# costo in performance della semplicita', quantificato
delta = (res.loc[best_name, "macro_F1"] - res.loc[chosen, "macro_F1"]) * 100
print(f"    prezzo della semplicita' vs miglior modello: {delta:.2f} punti di macro-F1")

# %%
joblib.dump(fitted[chosen], C.ARTIFACTS / "model_filone_a.joblib")
joblib.dump({"nome": chosen, "cv_macro_f1": float(res.loc[chosen, "macro_F1"]),
             "params": res.loc[chosen, "best_params"]},
            C.ARTIFACTS / "model_filone_a_meta.joblib")
print(f"\nmodello salvato in {C.ARTIFACTS / 'model_filone_a.joblib'}")
