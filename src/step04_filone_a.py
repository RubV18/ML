# %%
"""Step 8 (roadmap) — Filone A: il modello piu' SEMPLICE che basta.

Lo Step 0 ha dato un gap lineare/non lineare di ~3 punti di macro-F1: zona
grigia. La decisione non puo' quindi essere presa sui default, ma su modelli
*tutti* tunati con lo stesso budget e la stessa CV — SVM incluse.

Regola di scelta dichiarata PRIMA di guardare i risultati (one-standard-error
rule): fra i modelli tunati si sceglie il piu' semplice il cui macro-F1 medio in
CV sia entro 1 errore standard dal migliore. Implementata una volta sola in
`evaluation.one_se_selection`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

import config as C
from evaluation import (load_split, make_cv, one_se_selection, results_frame,
                        se_sensitivity, tune)
from models import CANDIDATES, COMPLEXITY_RANK

pd.set_option("display.width", 200)

X, y = load_split("pool")
cv = make_cv()

# %%
results = [tune(name, spec["estimator"], spec["grid"], X, y, cv, spec["scale"])
           for name, spec in CANDIDATES.items()]
fitted = {r["modello"]: r["estimator"] for r in results}

res = results_frame(results)
print("\n" + res.drop(columns=["sec", "n_config"]).round(4).to_string())
res.round(4).to_csv(C.REPORTS / "filone_a_tuning.csv")

# %%
# --- Applicazione della regola dichiarata a priori ------------------------
sel = one_se_selection(res, COMPLEXITY_RANK)
print(f"\nmigliore in assoluto : {sel['migliore']} "
      f"({res.loc[sel['migliore'], 'macro_F1']:.4f})")
print(f"soglia (best - 1 SE) : {sel['soglia']:.4f}")
print("modelli entro soglia :", ", ".join(sel["ammessi"]))
print(f"\n>>> FILONE A = {sel['scelto']}  (il piu' semplice entro 1 SE dal migliore)")
print(f"    macro-F1 CV = {res.loc[sel['scelto'], 'macro_F1']:.4f}  "
      f"iperparametri = {res.loc[sel['scelto'], 'best_params']}")
print(f"    prezzo della semplicita': {sel['prezzo_semplicita']*100:.2f} punti")

# %%
# --- Quanto e' robusta la regola? ----------------------------------------
# Con 5 fold su 10.000 clienti l'errore standard e' piccolo e la regola 1-SE
# tende a degenerare nel "vince il migliore". Renderlo esplicito e' parte del
# risultato, non una nota a pie' di pagina.
sens = se_sensitivity(res)
print("\nsensibilita' della soglia:")
print(sens.round(4).to_string())
sens.round(4).to_csv(C.REPORTS / "filone_a_se_sensitivity.csv")

# %%
joblib.dump(fitted[sel["scelto"]], C.ARTIFACTS / "model_filone_a.joblib")
joblib.dump({"nome": sel["scelto"],
             "cv_macro_f1": float(res.loc[sel["scelto"], "macro_F1"]),
             "params": res.loc[sel["scelto"], "best_params"]},
            C.ARTIFACTS / "model_filone_a_meta.joblib")
print(f"\nmodello salvato in {C.ARTIFACTS / 'model_filone_a.joblib'}")
