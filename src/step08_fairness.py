# %%
"""Step 12 (roadmap) — fairness check sull'eta'.

Perimetro deciso a monte: SOLO eta' (binned) come proxy di gruppo protetto.
Occupation e' esclusa di proposito — sia per non rendere circolare l'analisi
(se la si usasse anche come feature con target encoding), sia perche' dopo
l'aggregazione per cliente i sottogruppi occupazionali sarebbero troppo piccoli
per stime stabili.

Il test set (2.500 clienti) e' piccolo per stime per-gruppo: le metriche sono
calcolate anche sulle predizioni out-of-fold del train pool (10.000 clienti),
che danno intervalli piu' stretti. Le due viste vengono riportate entrambe.
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
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

import config as C
from evaluation import build_pipeline, load_split, make_cv

pd.set_option("display.width", 220)

X_pool, y_pool = load_split("pool")
X_test, y_test = load_split("test")

model = joblib.load(C.ARTIFACTS / "model_filone_a.joblib")
meta = joblib.load(C.ARTIFACTS / "model_filone_a_meta.joblib")
print(f"modello sotto esame: Filone A — {meta['nome']}")


def age_band(df):
    return pd.cut(df["Age"], bins=C.AGE_BINS, labels=C.AGE_LABELS)


# %%
# --- Metriche di fairness --------------------------------------------------
def fairness_table(y_true, y_pred, groups):
    """Demographic parity + equalized odds, per gruppo e per classe."""
    rows = []
    for g in [b for b in C.AGE_LABELS if (groups == b).sum() > 0]:
        m = (groups == g).values
        yt, yp = np.asarray(y_true)[m], np.asarray(y_pred)[m]
        row = {"gruppo": g, "n": int(m.sum()),
               "macro_F1": f1_score(yt, yp, average="macro",
                                    labels=C.CREDIT_SCORE_ORDER, zero_division=0)}
        for k in C.CREDIT_SCORE_ORDER:
            row[f"base_rate_{k}"] = (yt == k).mean()          # tasso reale
            row[f"sel_rate_{k}"] = (yp == k).mean()           # demographic parity
            pos = yt == k
            row[f"TPR_{k}"] = (yp[pos] == k).mean() if pos.sum() else np.nan
            neg = ~pos
            row[f"FPR_{k}"] = (yp[neg] == k).mean() if neg.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("gruppo")


def disparities(tbl):
    """Differenza max-min fra gruppi (0 = parita' perfetta)."""
    out = {}
    for k in C.CREDIT_SCORE_ORDER:
        out[f"DP gap ({k})"] = tbl[f"sel_rate_{k}"].max() - tbl[f"sel_rate_{k}"].min()
        out[f"EO gap TPR ({k})"] = tbl[f"TPR_{k}"].max() - tbl[f"TPR_{k}"].min()
        out[f"EO gap FPR ({k})"] = tbl[f"FPR_{k}"].max() - tbl[f"FPR_{k}"].min()
    out["gap macro-F1"] = tbl["macro_F1"].max() - tbl["macro_F1"].min()
    return pd.Series(out)


# %%
# --- Vista 1: out-of-fold sul train pool (n = 10.000) ---------------------
cv = make_cv()
oof = cross_val_predict(model, X_pool, y_pool, cv=cv, n_jobs=-1)
g_pool = age_band(X_pool)
tbl_oof = fairness_table(y_pool, oof, g_pool)

print("\n=== Vista OOF (train pool, n=10.000) ===")
print(tbl_oof[["n", "macro_F1"] + [f"base_rate_{k}" for k in C.CREDIT_SCORE_ORDER]
              + [f"sel_rate_{k}" for k in C.CREDIT_SCORE_ORDER]].round(3).to_string())
print("\nEqualized odds — TPR per classe:")
print(tbl_oof[[f"TPR_{k}" for k in C.CREDIT_SCORE_ORDER]].round(3).to_string())
print("\nEqualized odds — FPR per classe:")
print(tbl_oof[[f"FPR_{k}" for k in C.CREDIT_SCORE_ORDER]].round(3).to_string())
print("\nDisparita' (max - min fra gruppi):")
print(disparities(tbl_oof).round(3).to_string())

# %%
# --- Vista 2: test set (n = 2.500) ----------------------------------------
tbl_test = fairness_table(y_test, model.predict(X_test), age_band(X_test))
print("\n\n=== Vista TEST (n=2.500) ===")
print(tbl_test[["n", "macro_F1"] + [f"sel_rate_{k}" for k in C.CREDIT_SCORE_ORDER]]
      .round(3).to_string())
print("\nDisparita' (max - min fra gruppi):")
print(disparities(tbl_test).round(3).to_string())

tbl_oof.round(4).to_csv(C.REPORTS / "fairness_oof.csv")
tbl_test.round(4).to_csv(C.REPORTS / "fairness_test.csv")

# %%
# --- Grafico: tasso reale vs tasso predetto, per fascia -------------------
bands = list(tbl_oof.index)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, k in zip(axes, C.CREDIT_SCORE_ORDER):
    x = np.arange(len(bands))
    ax.bar(x - 0.2, tbl_oof[f"base_rate_{k}"], 0.4, label="tasso reale",
           color="#8899aa")
    ax.bar(x + 0.2, tbl_oof[f"sel_rate_{k}"], 0.4, label="tasso predetto",
           color={"Poor": "#c0392b", "Standard": "#e08b1f", "Good": "#2d7d46"}[k])
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_title(f"classe '{k}'", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("quota di clienti"); axes[0].legend(fontsize=7)
fig.suptitle("Demographic parity per fascia d'eta' — reale vs predetto (OOF)",
             fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGURES / "14_fairness_parity.png"); plt.close(fig)

# %%
# --- Test di "fairness through unawareness" -------------------------------
# Rimuovere l'eta' dalle feature elimina la disparita' o la lascia intatta
# perche' passa da variabili proxy (es. Credit_History_Age_Months, che con
# l'eta' e' strutturalmente correlata)? E' la domanda che distingue un fix
# cosmetico da uno reale.
age_proxy_corr = X_pool.corr(numeric_only=True)["Age"].drop("Age").abs()
print("\n\n=== Unawareness test ===")
print("feature piu' correlate con l'eta' (proxy candidate):")
print(age_proxy_corr.sort_values(ascending=False).head(5).round(3).to_string())

# Il preprocessor referenzia le colonne per nome: va ricostruito sullo spazio
# ridotto, mantenendo identici classificatore e iperparametri.
X_pool_na = X_pool.drop(columns=["Age"])
scale = any("sc" in dict(step[1].steps) for step in model.named_steps["prep"].transformers
            if hasattr(step[1], "steps"))
model_na = build_pipeline(clone(model.named_steps["clf"]), X_pool_na, scale)
oof_na = cross_val_predict(model_na, X_pool_na, y_pool, cv=cv, n_jobs=-1)
tbl_na = fairness_table(y_pool, oof_na, g_pool)

cmp = pd.DataFrame({"con Age": disparities(tbl_oof), "senza Age": disparities(tbl_na)})
cmp["variazione"] = cmp["senza Age"] - cmp["con Age"]
print("\ndisparita' con e senza la feature Age:")
print(cmp.round(3).to_string())
print(f"\nmacro-F1 OOF con Age    : {f1_score(y_pool, oof, average='macro'):.4f}")
print(f"macro-F1 OOF senza Age  : {f1_score(y_pool, oof_na, average='macro'):.4f}")
cmp.round(4).to_csv(C.REPORTS / "fairness_unawareness.csv")
