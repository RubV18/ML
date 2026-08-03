# %%
"""Step 11 (roadmap) — XAI: SHAP globale + esempi locali.

Le spiegazioni sono calcolate sul test set: sono un artefatto *post-hoc*, non
entrano in nessuna decisione di modellazione, quindi non violano la regola del
"test toccato una volta sola" per la stima di performance.

Tre esempi locali, scelti per rappresentare i casi d'uso reali di un ufficio
crediti: una predizione corretta, un errore, e un caso borderline.
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
import shap
from sklearn.linear_model import LogisticRegression

import config as C
from evaluation import load_split
from features import feature_names

pd.set_option("display.width", 200)

X_test, y_test, test = load_split("test", with_diagnostics=True)
X_pool, _ = load_split("pool")

model = joblib.load(C.ARTIFACTS / "model_filone_a.joblib")
meta = joblib.load(C.ARTIFACTS / "model_filone_a_meta.joblib")
prep, clf = model.named_steps["prep"], model.named_steps["clf"]
names = feature_names(prep)
classes = list(clf.classes_)
print(f"modello spiegato: Filone A — {meta['nome']}")

Z_test = pd.DataFrame(prep.transform(X_test), columns=names)
Z_pool = pd.DataFrame(prep.transform(X_pool), columns=names)

# %%
# --- Explainer adatto alla famiglia del modello ---------------------------
if isinstance(clf, LogisticRegression):
    bg = shap.sample(Z_pool, 200, random_state=C.RANDOM_STATE)
    explainer = shap.LinearExplainer(clf, bg)
    sv = explainer(Z_test)
else:
    explainer = shap.TreeExplainer(clf)
    sv = explainer(Z_test, check_additivity=False)

# Normalizza la forma a (n_campioni, n_feature, n_classi)
vals = sv.values
if vals.ndim == 2:                      # binario o lineare a una uscita
    vals = vals[:, :, None]
print("shape dei valori SHAP:", vals.shape)

# %%
# --- 1. Importanza globale: media di |SHAP| -------------------------------
imp = pd.DataFrame(np.abs(vals).mean(axis=0), index=names,
                   columns=classes if vals.shape[2] == len(classes) else ["shap"])
imp = imp.reindex(columns=[c for c in C.CREDIT_SCORE_ORDER if c in imp.columns])
imp["media"] = imp.mean(axis=1)
imp = imp.sort_values("media", ascending=False)
print("\nImportanza globale SHAP (top 15, media di |SHAP| per classe):")
print(imp.head(15).round(4).to_string())
imp.to_csv(C.REPORTS / "shap_global_importance.csv")

fig, ax = plt.subplots(figsize=(7, 6))
top = imp.head(18).drop(columns="media")[::-1]
top.plot(kind="barh", stacked=True, ax=ax, width=0.8,
         color={"Poor": "#c0392b", "Standard": "#e08b1f", "Good": "#2d7d46"})
ax.set_title("SHAP — importanza globale per classe (top 18)")
ax.set_xlabel("media di |valore SHAP|")
ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "10_shap_global_bar.png"); plt.close(fig)

# %%
# --- 2. Beeswarm sulla classe di maggiore interesse operativo -------------
# 'Poor' e' la classe su cui si prendono le decisioni piu' delicate (rifiuto,
# pricing peggiorativo): e' quella la cui spiegazione va difesa.
k_poor = classes.index("Poor")
exp_poor = shap.Explanation(
    values=vals[:, :, k_poor] if vals.shape[2] > 1 else vals[:, :, 0],
    base_values=(sv.base_values[:, k_poor] if np.ndim(sv.base_values) > 1
                 else sv.base_values),
    data=Z_test.values, feature_names=names,
)
plt.figure()
shap.plots.beeswarm(exp_poor, max_display=18, show=False)
plt.title("SHAP — contributi alla classe 'Poor'", fontsize=10)
plt.tight_layout(); plt.savefig(C.FIGURES / "11_shap_beeswarm_poor.png"); plt.close()

# %%
# --- 3. Dependence plot sulle due feature piu' importanti -----------------
for i, feat in enumerate(imp.head(2).index):
    plt.figure()
    shap.plots.scatter(exp_poor[:, feat], show=False)
    plt.title(f"SHAP dependence — {feat} (classe 'Poor')", fontsize=10)
    plt.tight_layout()
    plt.savefig(C.FIGURES / f"12_shap_dependence_{i+1}_{feat[:28]}.png")
    plt.close()

# %%
# --- 4. Tre casi locali ----------------------------------------------------
proba = model.predict_proba(X_test)
pred = model.predict(X_test)
top2 = np.sort(proba, axis=1)[:, -2:]
margin = top2[:, 1] - top2[:, 0]          # margine fra 1a e 2a classe

correct_poor = np.where((pred == "Poor") & (y_test.values == "Poor"))[0]
wrong = np.where(pred != y_test.values)[0]
borderline = np.argsort(margin)[:50]

cases = {
    "A_corretto_Poor": correct_poor[np.argmax(proba[correct_poor].max(axis=1))],
    "B_errore": wrong[np.argmax(proba[wrong].max(axis=1))],   # errore piu' "sicuro"
    "C_borderline": borderline[0],
}

print("\n--- Tre casi locali ---")
for label, idx in cases.items():
    k = classes.index(pred[idx])
    exp_i = shap.Explanation(
        values=vals[idx, :, k] if vals.shape[2] > 1 else vals[idx, :, 0],
        base_values=(sv.base_values[idx, k] if np.ndim(sv.base_values) > 1
                     else sv.base_values[idx]),
        data=Z_test.values[idx], feature_names=names,
    )
    print(f"\n[{label}] cliente {test['Customer_ID'].iloc[idx]}   "
          f"reale = {y_test.iloc[idx]}   predetto = {pred[idx]}   "
          f"p = {proba[idx].max():.2f}   margine = {margin[idx]:.2f}")
    contrib = pd.Series(exp_i.values, index=names).sort_values(key=abs, ascending=False)
    print("  fattori principali:")
    for f, v in contrib.head(5).items():
        verso = "verso" if v > 0 else "contro"
        print(f"    {f:38s} {v:+.3f}  ({verso} '{pred[idx]}')")

    plt.figure()
    shap.plots.waterfall(exp_i, max_display=14, show=False)
    plt.title(f"{label} — reale {y_test.iloc[idx]}, predetto {pred[idx]}", fontsize=9)
    plt.tight_layout(); plt.savefig(C.FIGURES / f"13_shap_local_{label}.png"); plt.close()

print(f"\nfigure SHAP salvate in {C.FIGURES}")
