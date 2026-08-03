# %%
"""Step 11 — appendice metodologica su un campione allargato.

Estensione fuori dal dataset assegnato: `newtrain.csv` contiene 37.500 clienti
(sovrainsieme dei 12.500 originali). Serve a due scopi che il dataset di
progetto non permetteva di raggiungere, ed e' etichettata come appendice
proprio perche' non fa parte dello scope principale.

  A. QUANTITA' DI DATI. In §8.3 la curva di apprendimento si appiattiva gia' a
     8.000 clienti e la conclusione ("non e' un problema di numerosita'") era
     un'estrapolazione oltre il range osservato. Qui la si verifica fino a
     28.000, e la si confronta end-to-end su un held-out immutato.

  B. PROTOCOLLO A TRE BLOCCHI. Con 12.500 clienti frammentare in train /
     validation / test sarebbe stato uno spreco: la CV usa i dati meglio di un
     blocco fisso. Con 37.500 diventa permissibile, e permette di separare la
     *selezione* dalla *stima* e di misurare la distorsione da selezione
     stimata solo teoricamente in §4.2.

ATTENZIONE al leakage: `newtrain.csv` contiene anche i 2.500 clienti del test
set originale. Ovunque si confronti con quel test set, essi vengono esclusi dal
pool di addestramento; nella parte B i blocchi sono invece ricavati interamente
da `newtrain`, quindi il problema non si pone.

Tempo di esecuzione indicativo: ~55 minuti (il grosso e' il tuning della
parte B su 22.500 clienti).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (cross_val_predict, learning_curve,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

import config as C
from data_prep import build_customer_dataset
from evaluation import (build_pipeline, load_split, make_cv, one_se_selection,
                        results_frame, tune)
from features import NON_FEATURES
from models import CANDIDATES, COMPLEXITY_RANK

pd.set_option("display.width", 210)

NEW_DIR = C.ROOT / "Data" / "new Syntetic data"
meta_a = joblib.load(C.ARTIFACTS / "model_filone_a_meta.joblib")
meta_b = joblib.load(C.ARTIFACTS / "model_filone_b_meta.joblib")
RF_PARAMS = dict(meta_a["params"])
DT_PARAMS = {k.replace("clf__", ""): v for k, v in meta_b["dt_params"].items()}


def rf(**kw):
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                  random_state=C.RANDOM_STATE, **{**RF_PARAMS, **kw})


def dt():
    return DecisionTreeClassifier(class_weight="balanced",
                                  random_state=C.RANDOM_STATE, **DT_PARAMS)


def xy(df):
    return df.drop(columns=[c for c in NON_FEATURES if c in df.columns]), df[C.TARGET]


# %%
print("costruzione del dataset a livello cliente (stessa pipeline di step01)...",
      flush=True)
raw = pd.read_csv(NEW_DIR / "newtrain.csv", low_memory=False)
cust, _ = build_customer_dataset(raw, use_dispersion=C.USE_DISPERSION_FEATURES,
                                 use_ratios=C.USE_DOMAIN_RATIOS)
print(f"  {len(cust):,} clienti", flush=True)

X_small, y_small = load_split("pool")
X_test, y_test = load_split("test")

# %%
# =========================================================================
# Controllo preliminare: la relazione feature -> etichette e' intatta?
# =========================================================================
# Un campione generato altrove va verificato prima di usarlo. Il test e' il
# confronto con due riferimenti a parita' di numerosita': il dataset originale
# (segnale reale) e lo stesso dataset con etichette permutate (rumore puro).
print("\n" + "=" * 72)
print("CONTROLLO DI INTEGRITA' DEL CAMPIONE")
print("=" * 72, flush=True)


def cv_macro_f1(X, y, model=None):
    p = build_pipeline(model or dt(), X, scale=False)
    return f1_score(y, cross_val_predict(p, X, y, cv=make_cv(), n_jobs=-1),
                    average="macro")


ids = pd.Index(raw[C.CUSTOMER_ID].unique())[:2500]
sub, _ = build_customer_dataset(raw[raw[C.CUSTOMER_ID].isin(ids)])
X_sub, y_sub = xy(sub)
X_ref, y_ref = X_small.iloc[:2500], y_small.iloc[:2500]
rng = np.random.default_rng(C.RANDOM_STATE)
y_perm = pd.Series(rng.permutation(y_ref.values), index=y_ref.index)

integ = pd.Series({
    "dataset originale, etichette vere": cv_macro_f1(X_ref, y_ref),
    "campione allargato": cv_macro_f1(X_sub, y_sub),
    "dataset originale, etichette permutate (rumore puro)": cv_macro_f1(X_ref, y_perm),
}, name="macro_F1 (CV, n=2.500)")
print("\n" + integ.round(4).to_string(), flush=True)
integ.round(4).to_csv(C.REPORTS / "scaling_integrita.csv", header=True)

# %%
# =========================================================================
# A1. Curva di apprendimento estesa
# =========================================================================
print("\n" + "=" * 72)
print("A1. CURVA DI APPRENDIMENTO ESTESA")
print("=" * 72, flush=True)

# Il pool allargato esclude i clienti del test originale: senza questo passo il
# confronto sarebbe vinto dal leakage, non dai dati.
test_ids = set(pd.read_csv(C.ARTIFACTS / "test.csv")[C.CUSTOMER_ID])
X_big, y_big = xy(cust[~cust[C.CUSTOMER_ID].isin(test_ids)].reset_index(drop=True))
print(f"pool allargato: {len(X_big):,} clienti "
      f"(esclusi i {len(test_ids):,} del test originale)", flush=True)

sizes, tr, va = learning_curve(
    build_pipeline(rf(), X_big, scale=False), X_big, y_big, cv=make_cv(),
    scoring=C.PRIMARY_METRIC, n_jobs=1, random_state=C.RANDOM_STATE,
    train_sizes=[1200, 2500, 5000, 8000, 12000, 18000, 28000])

lc = pd.DataFrame({"n_train": sizes.astype(int), "train_macro_F1": tr.mean(axis=1),
                   "cv_macro_F1": va.mean(axis=1), "sd": va.std(axis=1)})
lc["guadagno"] = lc["cv_macro_F1"].diff()
print("\n" + lc.round(4).to_string(index=False))
lc.round(5).to_csv(C.REPORTS / "scaling_learning_curve.csv", index=False)

d = (lc.cv_macro_F1.iloc[-1] - lc.loc[lc.n_train == 8000, "cv_macro_F1"].iloc[0]) * 100
print(f"\nda 8.000 a 28.000 clienti: {d:+.2f} punti di macro-F1")
print(f"gap train - CV: {lc.train_macro_F1.iloc[0]-lc.cv_macro_F1.iloc[0]:.4f} "
      f"(a 1.200) -> {lc.train_macro_F1.iloc[-1]-lc.cv_macro_F1.iloc[-1]:.4f} (a 28.000)",
      flush=True)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
old = C.REPORTS / "diagnostica_t3_learning_curve.csv"
if old.exists():
    o = pd.read_csv(old)
    ax.plot(o.n_train, o.cv_macro_F1, "o--", color="#999999", ms=4,
            label="dataset di progetto (12.500 clienti)")
ax.plot(lc.n_train, lc.cv_macro_F1, "o-", color="#4c72b0",
        label="campione allargato (37.500)")
ax.fill_between(lc.n_train, lc.cv_macro_F1 - lc.sd, lc.cv_macro_F1 + lc.sd,
                alpha=0.2, color="#4c72b0")
ax.set_xlabel("clienti nel training set"); ax.set_ylabel("macro-F1 (CV)")
ax.set_title("Curva di apprendimento estesa a 28.000 clienti", fontsize=10)
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "16_learning_curve_estesa.png"); plt.close(fig)

# %%
# =========================================================================
# A2. Confronto end-to-end sullo stesso held-out
# =========================================================================
print("\n" + "=" * 72)
print("A2. STESSI IPERPARAMETRI, POOL 10.000 vs 35.000")
print("=" * 72, flush=True)

rows = []
for nome, make in [("Random Forest (Filone A)", rf), ("Decision Tree (Filone B)", dt)]:
    for pool, Xp, yp in [("pool 10.000", X_small, y_small),
                         ("pool 35.000", X_big, y_big)]:
        pr = build_pipeline(make(), Xp, scale=False).fit(Xp, yp).predict(X_test)
        rows.append({"modello": nome, "training": pool, "n_train": len(Xp),
                     "macro_F1_test": f1_score(y_test, pr, average="macro"),
                     "accuracy_test": accuracy_score(y_test, pr)})
        print(f"  {nome:26s} {pool:12s} macro-F1 = {rows[-1]['macro_F1_test']:.4f}",
              flush=True)

sc = pd.DataFrame(rows)
sc.round(4).to_csv(C.REPORTS / "scaling_piu_dati.csv", index=False)
print("\nguadagno da 3,5x i dati, a iperparametri invariati:")
for m in sc.modello.unique():
    s = sc[sc.modello == m]
    print(f"  {m:26s} {(s.macro_F1_test.iloc[1]-s.macro_F1_test.iloc[0])*100:+.2f} punti")

# %%
# =========================================================================
# B. Protocollo a tre blocchi 60/20/20
# =========================================================================
print("\n" + "=" * 72)
print("B. PROTOCOLLO A TRE BLOCCHI 60/20/20")
print("=" * 72, flush=True)

X_all, y_all = xy(cust)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_all, y_all, test_size=0.40, stratify=y_all, random_state=C.RANDOM_STATE)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=C.RANDOM_STATE)
print(f"train {len(X_tr):,} (tuning in CV) | validation {len(X_va):,} (selezione) "
      f"| test {len(X_te):,} (stima finale)", flush=True)

results = [tune(n, s["estimator"], s["grid"], X_tr, y_tr, scale=s["scale"])
           for n, s in CANDIDATES.items()]
fitted = {r["modello"]: r["estimator"] for r in results}
res = results_frame(results)
for r in results:
    m = fitted[r["modello"]]
    res.loc[r["modello"], "validation"] = f1_score(y_va, m.predict(X_va), average="macro")
    res.loc[r["modello"], "test"] = f1_score(y_te, m.predict(X_te), average="macro")

tab = res[["macro_F1", "SE", "validation", "test"]].rename(columns={"macro_F1": "CV (train)"})
tab["scarto val-CV"] = tab["validation"] - tab["CV (train)"]
print("\n" + tab.round(4).to_string())
tab.round(4).to_csv(C.REPORTS / "holdout_tre_stime.csv")
print(f"\nscarto medio |validation - CV|: {tab['scarto val-CV'].abs().mean():.4f}"
      f"   massimo: {tab['scarto val-CV'].abs().max():.4f}", flush=True)

# %%
# --- Le due selezioni concordano? ----------------------------------------
sel_cv = one_se_selection(res, COMPLEXITY_RANK)
best_val = tab["validation"].idxmax()
print(f"\nselezione via regola 1-SE sulla CV : {sel_cv['scelto']}")
print(f"selezione via blocco di validazione: {best_val}")
print(f"-> {'CONCORDANO' if sel_cv['scelto'] == best_val else 'DIVERGONO'}")

v = sel_cv["scelto"]
print(f"\nvincitore: {v}")
print(f"  CV (usata per selezionare): {tab.loc[v,'CV (train)']:.4f}")
print(f"  validation (mai vista)    : {tab.loc[v,'validation']:.4f}")
print(f"  test (mai visto)          : {tab.loc[v,'test']:.4f}")
print(f"  ottimismo CV vs test      : {(tab.loc[v,'CV (train)']-tab.loc[v,'test'])*100:+.2f} punti"
      "   (negativo = la CV era conservativa)", flush=True)

# %%
# --- Controllo: tenuta su blocchi mai visti ------------------------------
y_te_perm = pd.Series(rng.permutation(y_te.values), index=y_te.index)
best = fitted[v]
conf = pd.DataFrame({
    "macro_F1": [tab.loc[v, "CV (train)"], tab.loc[v, "validation"], tab.loc[v, "test"],
                 f1_score(y_te_perm, best.predict(X_te), average="macro")],
    "accuracy": [np.nan, accuracy_score(y_va, best.predict(X_va)),
                 accuracy_score(y_te, best.predict(X_te)),
                 accuracy_score(y_te_perm, best.predict(X_te))],
}, index=["CV interna (train)", "validation (mai vista)", "test (mai visto)",
          "test con etichette permutate (rumore puro)"])
print("\n" + conf.round(4).to_string())
conf.round(4).to_csv(C.REPORTS / "holdout_controllo.csv")

s = conf["macro_F1"].iloc[:3]
print(f"\nescursione fra le tre stime: {s.max()-s.min():.4f}")
print(f"distanza dal rumore puro   : {s.min()-conf['macro_F1'].iloc[3]:.4f}", flush=True)
