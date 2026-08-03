# %%
"""Step 2 — EDA sul solo train pool.

L'EDA gira sul train pool, mai sul test: guardare le distribuzioni del test
per decidere feature o trasformazioni sarebbe leakage dell'analista.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import config as C
from evaluation import load_split
from features import split_columns

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
pd.set_option("display.width", 200)

_, y, pool = load_split("pool", with_diagnostics=True)
num_cols, cat_cols = split_columns(pool)
PALETTE = {"Poor": "#c0392b", "Standard": "#e08b1f", "Good": "#2d7d46"}

# %%
# --- 1. Distribuzione del target ------------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 3))
counts = y.value_counts().loc[C.CREDIT_SCORE_ORDER]
ax.bar(counts.index, counts.values, color=[PALETTE[k] for k in counts.index])
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n{v/len(y):.1%}", ha="center", va="bottom", fontsize=8)
ax.set_title("Distribuzione del target (train pool)")
ax.set_ylim(0, counts.max() * 1.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "01_target_distribution.png"); plt.close(fig)

# rapporto di sbilanciamento -> motiva il class weighting
print("imbalance ratio (max/min):", (counts.max() / counts.min()).round(2))

# %%
# --- 2. Correlazioni fra numeriche (multicollinearita' attesa) ------------
corr = pool[num_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=6)
fig.colorbar(im, shrink=0.7)
ax.set_title("Correlazione di Spearman fra feature numeriche")
fig.tight_layout(); fig.savefig(C.FIGURES / "02_correlation.png"); plt.close(fig)

hi = (corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
        .sort_values(ascending=False).drop_duplicates())
print("\ncoppie con |rho| > 0.7:")
print(hi[hi > 0.7].round(3).to_string())

# %%
# --- 3. Potere discriminante univariato -----------------------------------
X_num = pool[num_cols].fillna(pool[num_cols].median())
mi = pd.Series(
    mutual_info_classif(X_num, y, random_state=C.RANDOM_STATE), index=num_cols
).sort_values(ascending=False)

cat_mi = {}
for c in cat_cols:
    codes = pool[c].astype("category").cat.codes.values.reshape(-1, 1)
    cat_mi[c] = mutual_info_classif(codes, y, discrete_features=True,
                                    random_state=C.RANDOM_STATE)[0]
mi_all = pd.concat([mi, pd.Series(cat_mi)]).sort_values(ascending=False)
print("\nMutual information con il target (top 15)")
print(mi_all.head(15).round(4).to_string())

fig, ax = plt.subplots(figsize=(6, 6))
top = mi_all.head(20)[::-1]
ax.barh(top.index, top.values, color="#4c72b0")
ax.set_title("Mutual information con Credit_Score (top 20)")
ax.set_xlabel("MI (nats)"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "03_mutual_information.png"); plt.close(fig)

# %%
# --- 4. Le feature piu' informative, per classe ---------------------------
key = mi_all.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, col in zip(axes.ravel(), key):
    if col in cat_cols:
        ct = pd.crosstab(pool[col], y, normalize="index")[C.CREDIT_SCORE_ORDER]
        ct.plot(kind="bar", stacked=True, ax=ax, legend=False,
                color=[PALETTE[k] for k in C.CREDIT_SCORE_ORDER], width=0.8)
        ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=6, rotation=30)
    else:
        data = [pool.loc[y == k, col].dropna() for k in C.CREDIT_SCORE_ORDER]
        bp = ax.boxplot(data, tick_labels=C.CREDIT_SCORE_ORDER, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, k in zip(bp["boxes"], C.CREDIT_SCORE_ORDER):
            patch.set_facecolor(PALETTE[k]); patch.set_alpha(0.75)
    ax.set_title(col, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Le 6 feature piu' informative, per classe di Credit_Score", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIGURES / "04_top_features_by_class.png"); plt.close(fig)

# %%
# --- 5. Eta': base per il fairness check ----------------------------------
pool["age_band"] = pd.cut(pool["Age"], bins=C.AGE_BINS, labels=C.AGE_LABELS)
print("\nnumerosita' per fascia d'eta':")
print(pool["age_band"].value_counts().reindex(C.AGE_LABELS).to_string())
print("\ndistribuzione del target per fascia d'eta' (%):")
band_tbl = pd.crosstab(pool["age_band"], y, normalize="index")[C.CREDIT_SCORE_ORDER] * 100
print(band_tbl.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].hist(pool["Age"].dropna(), bins=40, color="#4c72b0")
axes[0].set_title("Distribuzione dell'eta'"); axes[0].set_xlabel("Age")
band_tbl.plot(kind="bar", stacked=True, ax=axes[1],
              color=[PALETTE[k] for k in C.CREDIT_SCORE_ORDER], width=0.8)
axes[1].set_title("Composizione del target per fascia d'eta'")
axes[1].set_ylabel("%"); axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(fontsize=7, loc="lower right")
for ax in axes: ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "05_age_and_target.png"); plt.close(fig)

# %%
# --- 6. Tipi di prestito vs classe ----------------------------------------
loan_cols = [c for c in pool.columns if c.startswith("has_")]
rates = pd.DataFrame({
    lc: pool.loc[pool[lc] == 1, C.TARGET].value_counts(normalize=True)
    for lc in loan_cols
}).T[C.CREDIT_SCORE_ORDER] * 100
base = y.value_counts(normalize=True)[C.CREDIT_SCORE_ORDER] * 100
print("\n% di classe 'Poor' per tipo di prestito posseduto (baseline "
      f"{base['Poor']:.1f}%):")
print(rates["Poor"].sort_values(ascending=False).round(1).to_string())

fig, ax = plt.subplots(figsize=(6.5, 3.5))
d = (rates["Poor"] - base["Poor"]).sort_values()
ax.barh(d.index, d.values, color=np.where(d.values > 0, "#c0392b", "#2d7d46"))
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Scarto dalla quota media di 'Poor', per tipo di prestito (p.p.)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "06_loan_types.png"); plt.close(fig)

print(f"\nfigure salvate in {C.FIGURES}")
