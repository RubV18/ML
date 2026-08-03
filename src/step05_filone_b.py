# %%
"""Step 9 (roadmap) — Filone B: massimizzare l'interpretabilita'.

Tre pezzi:
  B1. Logistic Regression con penalita' L1 -> coefficienti sparsi, quindi
      feature selection automatica e leggibile.
  B2. Albero decisionale a profondita' limitata -> nativamente interpretabile,
      non ha bisogno di SHAP come strato aggiuntivo. Confronto diretto con B1.
  B3. statsmodels come step DIAGNOSTICO, eseguito *dopo* la selezione L1 e solo
      sulle feature sopravvissute: con le feature grezze (Annual_Income,
      Monthly_Inhand_Salary, Outstanding_Debt... fortemente correlate) i p-value
      dell'inferenza classica sarebbero inaffidabili per multicollinearita'.
      Il VIF viene controllato prima di dare credito ai p-value.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config as C
from evaluation import build_pipeline, load_split, make_cv
from features import CATEGORICAL, feature_names
from models import logistic_l1

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 120)

X, y = load_split("pool")
cv = make_cv()

# %%
# --- B1. Logistic Regression L1 -------------------------------------------
print("=== B1: Logistic Regression con penalita' L1 ===")
pipe_l1 = build_pipeline(logistic_l1(), X, scale=True)
C_GRID = [0.002, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0]
t0 = time.perf_counter()
gs_l1 = GridSearchCV(pipe_l1, {"clf__C": C_GRID}, scoring=C.PRIMARY_METRIC,
                     cv=cv, n_jobs=-1)
gs_l1.fit(X, y)

# Trade-off parsimonia / performance lungo tutto il percorso di regolarizzazione.
sparsity = []
for j, c_val in enumerate(C_GRID):
    p = build_pipeline(logistic_l1(C=c_val), X, scale=True).fit(X, y)
    k = int((np.abs(p.named_steps["clf"].coef_).max(axis=0) > 1e-8).sum())
    sparsity.append({"C": c_val, "n_feature": k,
                     "macro_F1_cv": gs_l1.cv_results_["mean_test_score"][j]})
sp = pd.DataFrame(sparsity)

# Scegliere C sul solo massimo di macro-F1 vanificherebbe lo scopo della L1 in
# questo filone (con C alto sopravvivono quasi tutte le feature). Regola
# dichiarata a priori, coerente con l'obiettivo "massimizzare l'interpretabilita'":
# si prende il C **piu' parsimonioso** il cui macro-F1 resti entro 1 SE dal migliore.
best_j = int(gs_l1.best_index_)
best_cv = gs_l1.cv_results_["mean_test_score"][best_j]
best_se = gs_l1.cv_results_["std_test_score"][best_j] / np.sqrt(C.N_FOLDS)
ok = sp[sp["macro_F1_cv"] >= best_cv - best_se].sort_values("n_feature")
C_sel = float(ok.iloc[0]["C"])

print(f"C con macro-F1 massimo : {gs_l1.best_params_['clf__C']} "
      f"-> {best_cv:.4f} (SE {best_se:.4f}), "
      f"{int(sp.loc[best_j,'n_feature'])} feature attive")
print("\ntrade-off parsimonia / performance:")
print(sp.round(4).to_string(index=False))
print(f"\nsoglia (best - 1 SE) = {best_cv - best_se:.4f}")
print(f">>> C selezionato = {C_sel} ({int(ok.iloc[0]['n_feature'])} feature attive, "
      f"macro-F1 {ok.iloc[0]['macro_F1_cv']:.4f})")
sp.to_csv(C.REPORTS / "filone_b_l1_sparsity.csv", index=False)

fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(sp["n_feature"], sp["macro_F1_cv"], "o-", color="#4c72b0")
ax.axhline(best_cv - best_se, ls="--", c="grey", lw=0.9, label="soglia best − 1 SE")
for _, r in sp.iterrows():
    ax.annotate(f"C={r['C']:g}", (r["n_feature"], r["macro_F1_cv"]),
                fontsize=6, xytext=(0, 5), textcoords="offset points", ha="center")
ax.set_xlabel("n. feature con coefficiente non nullo"); ax.set_ylabel("macro-F1 (CV)")
ax.set_title("L1: percorso parsimonia / performance", fontsize=9)
ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "07b_l1_path.png"); plt.close(fig)

# Rifit del modello L1 finale al C selezionato
best_l1 = build_pipeline(logistic_l1(C=C_sel), X, scale=True).fit(X, y)
names = np.array(feature_names(best_l1.named_steps["prep"]))
coefs = best_l1.named_steps["clf"].coef_
alive = np.abs(coefs).max(axis=0) > 1e-8
l1_cv = float(ok.iloc[0]["macro_F1_cv"])
l1_se = best_se
print(f"feature sopravvissute: {alive.sum()} / {len(names)}")

# %%
coef_df = (pd.DataFrame(coefs.T, index=names,
                        columns=best_l1.named_steps["clf"].classes_)
           .loc[alive]
           .reindex(columns=C.CREDIT_SCORE_ORDER))
coef_df["max_abs"] = coef_df.abs().max(axis=1)
coef_df = coef_df.sort_values("max_abs", ascending=False)
print("\ncoefficienti L1 (top 20, feature standardizzate -> confrontabili):")
print(coef_df.head(20).round(3).to_string())
coef_df.to_csv(C.REPORTS / "filone_b_l1_coefficients.csv")

fig, ax = plt.subplots(figsize=(7, 6))
top = coef_df.head(18).drop(columns="max_abs")[::-1]
top.plot(kind="barh", ax=ax, width=0.8,
         color={"Poor": "#c0392b", "Standard": "#e08b1f", "Good": "#2d7d46"})
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Logistic Regression L1 — coefficienti per classe (top 18)")
ax.set_xlabel("coefficiente (feature standardizzate)")
ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "07_l1_coefficients.png"); plt.close(fig)

# %%
# --- B2. Albero decisionale poco profondo ---------------------------------
print("\n=== B2: albero decisionale a profondita' limitata ===")
pipe_dt = build_pipeline(
    DecisionTreeClassifier(class_weight="balanced", random_state=C.RANDOM_STATE),
    X, scale=False)
grid_dt = {"clf__max_depth": [3, 4], "clf__min_samples_leaf": [10, 50, 100, 200],
           "clf__criterion": ["gini", "entropy"]}
gs_dt = GridSearchCV(pipe_dt, grid_dt, scoring=C.PRIMARY_METRIC, cv=cv, n_jobs=-1)
gs_dt.fit(X, y)
dt_cv = gs_dt.cv_results_["mean_test_score"][gs_dt.best_index_]
dt_se = gs_dt.cv_results_["std_test_score"][gs_dt.best_index_] / np.sqrt(C.N_FOLDS)
print(f"parametri = {gs_dt.best_params_}   macro-F1 CV = {dt_cv:.4f} (SE {dt_se:.4f})")

dt_names = feature_names(gs_dt.best_estimator_.named_steps["prep"])
fig, ax = plt.subplots(figsize=(17, 8))
plot_tree(gs_dt.best_estimator_.named_steps["clf"], feature_names=dt_names,
          class_names=list(gs_dt.best_estimator_.named_steps["clf"].classes_),
          filled=True, rounded=True, fontsize=6, impurity=False, proportion=True, ax=ax)
ax.set_title(f"Filone B — albero decisionale (profondita' "
             f"{gs_dt.best_params_['clf__max_depth']})")
fig.tight_layout(); fig.savefig(C.FIGURES / "08_decision_tree.png"); plt.close(fig)

# %%
# --- Scelta del modello finale del Filone B -------------------------------
# Entrambi sono nativamente interpretabili: a parita' di leggibilita' vince
# semplicemente il macro-F1 in CV; se la differenza e' entro 1 SE si preferisce
# l'albero, che si legge senza bisogno di SHAP.
print("\n--- confronto B1 vs B2 ---")
print(f"  L1 logistic  : {l1_cv:.4f} (SE {l1_se:.4f}), {alive.sum()} feature attive")
print(f"  albero       : {dt_cv:.4f} (SE {dt_se:.4f}), profondita' "
      f"{gs_dt.best_params_['clf__max_depth']}")
if abs(dt_cv - l1_cv) <= max(l1_se, dt_se):
    b_name, b_model, b_cv = ("Decision Tree", gs_dt.best_estimator_, dt_cv)
    reason = "differenza entro 1 SE -> si preferisce l'albero (leggibile senza SHAP)"
elif dt_cv > l1_cv:
    b_name, b_model, b_cv = ("Decision Tree", gs_dt.best_estimator_, dt_cv)
    reason = "l'albero e' significativamente migliore"
else:
    b_name, b_model, b_cv = ("Logistic Regression (L1)", best_l1, l1_cv)
    reason = "la L1 e' significativamente migliore"
print(f">>> FILONE B = {b_name}  ({reason})")

joblib.dump(b_model, C.ARTIFACTS / "model_filone_b.joblib")
joblib.dump(best_l1, C.ARTIFACTS / "model_l1.joblib")
joblib.dump({"nome": b_name, "cv_macro_f1": float(b_cv),
             "l1_cv": float(l1_cv), "dt_cv": float(dt_cv),
             "n_feature_l1": int(alive.sum()), "n_feature_tot": int(len(names)),
             "l1_C": gs_l1.best_params_["clf__C"], "dt_params": gs_dt.best_params_},
            C.ARTIFACTS / "model_filone_b_meta.joblib")

# %%
# --- B3. statsmodels: diagnostica inferenziale sul sottoinsieme L1 --------
print("\n=== B3: diagnostica statsmodels sulle feature selezionate da L1 ===")
prep = best_l1.named_steps["prep"]
Z = pd.DataFrame(prep.transform(X), columns=names)[names[alive]].copy()

# Le dummy one-hot complete sono linearmente dipendenti: per l'inferenza serve
# una categoria di riferimento per ciascuna variabile categoriale.
for cat in CATEGORICAL:
    dummies = [c for c in Z.columns if c.startswith(cat + "_")]
    if len(dummies) > 1:
        Z = Z.drop(columns=[dummies[0]])          # prima categoria = riferimento
        print(f"  categoria di riferimento per {cat}: {dummies[0]}")

# Collinearita' STRUTTURALE, non statistica: n_loan_types e' per costruzione la
# somma esatta delle dummy has_*. Lasciarla dentro rende la matrice singolare e
# manda tutti i VIF a infinito. Va rimossa a priori, non in base al VIF.
if "n_loan_types" in Z.columns and any(c.startswith("has_") for c in Z.columns):
    Z = Z.drop(columns=["n_loan_types"])
    print("  rimossa n_loan_types: combinazione lineare esatta delle dummy has_*")


def vif_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [variance_inflation_factor(frame.values, i) for i in range(frame.shape[1])],
        index=frame.columns,
    ).sort_values(ascending=False)


vif_full = vif_series(Z)
print("\nVIF sul sottoinsieme selezionato da L1 (soglia convenzionale: 5-10)")
print(vif_full.round(2).to_string())
vif_full.to_csv(C.REPORTS / "filone_b_vif.csv", header=["VIF"])

# Potatura greedy: si elimina una alla volta la feature con VIF piu' alto finche'
# tutte scendono sotto 10. Serve a rendere leggibili i p-value; il modello
# predittivo NON viene toccato (la potatura vive solo nello strato inferenziale).
dropped, vif = [], vif_full
while vif.iloc[0] > 10 and Z.shape[1] > 2:
    worst = vif.index[0]
    dropped.append((worst, float(vif.iloc[0])))
    Z = Z.drop(columns=[worst])
    vif = vif_series(Z)

if dropped:
    print("\nfeature rimosse per VIF > 10 (solo ai fini inferenziali):")
    for f, v in dropped:
        print(f"  {f:34s} VIF = {v:,.1f}")
    print(f"\nVIF residui (max = {vif.iloc[0]:.2f}):")
    print(vif.round(2).to_string())
else:
    print("\nNessun VIF > 10: i p-value sul sottoinsieme ridotto sono leggibili.")

# %%
# MNLogit non supporta i pesi di classe: e' uno strato di *inferenza*, non il
# modello di produzione. I coefficienti vanno letti per segno e significativita',
# non come sostituti di quelli del modello pesato usato per predire.
y_codes = pd.Categorical(y, categories=C.CREDIT_SCORE_ORDER).codes
mn = sm.MNLogit(y_codes, sm.add_constant(Z)).fit(method="newton", maxiter=200, disp=0)
print(f"\nMNLogit — baseline = '{C.CREDIT_SCORE_ORDER[0]}', "
      f"pseudo R^2 = {mn.prsquared:.4f}")
with open(C.REPORTS / "filone_b_statsmodels.txt", "w") as f:
    f.write(str(mn.summary()))
print(str(mn.summary())[:4000])
print(f"\n(summary completo in {C.REPORTS / 'filone_b_statsmodels.txt'})")
