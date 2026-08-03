# %%
"""Step 13 — diagnostica: perche' l'accuratezza si ferma al 77%?

Tre ipotesi, testate separatamente:

  T1. LEAKAGE. I notebook pubblici su questo dataset splittano le 100.000 righe
      a caso, mettendo mesi diversi dello STESSO cliente in train e in test.
      Quanto gonfia il risultato? Confronto controllato: stesso dataset a
      livello riga, stesso modello, stessi iperparametri, cambia solo il modo
      di splittare (casuale per riga vs raggruppato per cliente).

  T2. METRICA. Il class weighting sacrifica deliberatamente accuratezza per
      recall sulle classi minoritarie. Quanto costa in accuratezza?
      (misurato in CV sul train pool: e' una scelta fra configurazioni, e le
      scelte non si arbitrano sul test set)

  T3. NUMEROSITA'. Se la curva di apprendimento e' ancora in salita a 10.000
      clienti, il limite e' il campione; se e' piatta, il limite e' altrove
      (rumore del label o potere informativo delle feature).
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
from sklearn.model_selection import (GroupShuffleSplit, cross_val_predict,
                                     learning_curve, train_test_split)

import config as C
from data_prep import build_customer_dataset, clean_rows, loan_type_multihot
from evaluation import build_pipeline, load_split, make_cv

pd.set_option("display.width", 200)

meta_a = joblib.load(C.ARTIFACTS / "model_filone_a_meta.joblib")
RF_PARAMS = {k: v for k, v in meta_a["params"].items()}
print("iperparametri RF (dal tuning del Filone A):", RF_PARAMS)


def rf():
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                  random_state=C.RANDOM_STATE, **RF_PARAMS)


# %%
# =========================================================================
# T1 — quanto vale il leakage fra mesi dello stesso cliente?
# =========================================================================
print("\n" + "=" * 72)
print("T1. LEAKAGE: split casuale per riga vs split raggruppato per cliente")
print("=" * 72)

raw = pd.read_csv(C.DATA_RAW, low_memory=False)
clean = clean_rows(raw.drop(columns=["ID", "Name", "SSN"]))

# Stesso feature engineering, ma SENZA aggregare: una riga = un mese.
row = loan_type_multihot(clean).drop(
    columns=["Type_of_Loan", "Month", "_month_idx"])

groups = row[C.CUSTOMER_ID]
y_row = row[C.TARGET]
X_row = row.drop(columns=[C.CUSTOMER_ID, C.TARGET])
print(f"dataset a livello riga: {X_row.shape[0]:,} righe x {X_row.shape[1]} colonne")

res_t1 = {}

# (a) split CASUALE per riga: lo stesso cliente finisce in train E in test
Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
    X_row, y_row, test_size=C.TEST_SIZE, stratify=y_row, random_state=C.RANDOM_STATE)
pipe = build_pipeline(rf(), X_row, scale=False)
pipe.fit(Xa_tr, ya_tr)
pa = pipe.predict(Xa_te)
res_t1["riga (casuale) — CON leakage"] = {
    "accuracy": accuracy_score(ya_te, pa), "macro_F1": f1_score(ya_te, pa, average="macro")}
print(f"\n(a) split casuale per riga   accuracy = {res_t1['riga (casuale) — CON leakage']['accuracy']:.4f}"
      f"  macro-F1 = {res_t1['riga (casuale) — CON leakage']['macro_F1']:.4f}")

# quanti clienti del test compaiono anche nel train?
overlap = len(set(groups.loc[Xa_te.index]) & set(groups.loc[Xa_tr.index]))
print(f"    clienti presenti sia in train sia in test: {overlap:,} "
      f"({overlap/groups.loc[Xa_te.index].nunique():.1%} di quelli del test)")

# (b) stesso dataset, stesso modello, ma split RAGGRUPPATO per cliente
gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)
itr, ite = next(gss.split(X_row, y_row, groups))
pipe = build_pipeline(rf(), X_row, scale=False)
pipe.fit(X_row.iloc[itr], y_row.iloc[itr])
pb = pipe.predict(X_row.iloc[ite])
res_t1["riga (per cliente) — SENZA leakage"] = {
    "accuracy": accuracy_score(y_row.iloc[ite], pb),
    "macro_F1": f1_score(y_row.iloc[ite], pb, average="macro")}
print(f"(b) split per cliente       accuracy = {res_t1['riga (per cliente) — SENZA leakage']['accuracy']:.4f}"
      f"  macro-F1 = {res_t1['riga (per cliente) — SENZA leakage']['macro_F1']:.4f}")
print(f"    clienti condivisi fra train e test: "
      f"{len(set(groups.iloc[itr]) & set(groups.iloc[ite]))}")

d_acc = (res_t1["riga (casuale) — CON leakage"]["accuracy"]
         - res_t1["riga (per cliente) — SENZA leakage"]["accuracy"]) * 100
print(f"\n>>> Il solo cambio di split vale {d_acc:+.1f} punti di accuratezza. "
      "Tutto leakage.")

# (b-bis) split casuale per riga E Customer_ID lasciato fra le feature.
# E' l'errore piu' comune: l'identificativo viene codificato come numero e il
# modello impara la tabella cliente -> label invece della relazione di rischio.
X_id = X_row.copy()
X_id["Customer_ID_enc"] = groups.astype("category").cat.codes.values
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    X_id, y_row, test_size=C.TEST_SIZE, stratify=y_row, random_state=C.RANDOM_STATE)
pipe = build_pipeline(rf(), X_id, scale=False)
pipe.fit(Xi_tr, yi_tr)
pi = pipe.predict(Xi_te)
res_t1["riga + Customer_ID fra le feature"] = {
    "accuracy": accuracy_score(yi_te, pi), "macro_F1": f1_score(yi_te, pi, average="macro")}
print(f"(b-bis) riga + Customer_ID  accuracy = {res_t1['riga + Customer_ID fra le feature']['accuracy']:.4f}"
      f"  macro-F1 = {res_t1['riga + Customer_ID fra le feature']['macro_F1']:.4f}")

# (c) la nostra pipeline: aggregata per cliente
# Unico uso del test set in questo step: ri-riportare il numero gia' pubblicato
# da step06 come termine di paragone. Nessuna nuova decisione ne dipende.
X_test, y_test = load_split("test")
model_a = joblib.load(C.ARTIFACTS / "model_filone_a.joblib")
pc = model_a.predict(X_test)
res_t1["cliente (aggregato) — questo progetto"] = {
    "accuracy": accuracy_score(y_test, pc),
    "macro_F1": f1_score(y_test, pc, average="macro")}

t1 = pd.DataFrame(res_t1).T
print("\n" + t1.round(4).to_string())
t1.round(4).to_csv(C.REPORTS / "diagnostica_t1_leakage.csv")

# %%
# =========================================================================
# T2 — quanto costa in accuratezza il class weighting?
# =========================================================================
print("\n" + "=" * 72)
print("T2. METRICA: class weighting acceso vs spento")
print("=" * 72)
# Confronto in CV sul train pool, NON sul test set: e' un'ablation fra due
# configurazioni, cioe' esattamente il tipo di scelta che il test set non deve
# arbitrare. Usa le predizioni out-of-fold, coerentemente con il resto del
# progetto.
X_pool, y_pool = load_split("pool")
cv = make_cv()

rows = []
for label, cw in [("class_weight='balanced' (scelta del progetto)", "balanced"),
                  ("class_weight=None", None)]:
    clf = RandomForestClassifier(class_weight=cw, n_jobs=-1,
                                 random_state=C.RANDOM_STATE, **RF_PARAMS)
    pr = cross_val_predict(build_pipeline(clf, X_pool, scale=False),
                           X_pool, y_pool, cv=cv, n_jobs=-1)
    rows.append({"config": label, "accuracy": accuracy_score(y_pool, pr),
                 "macro_F1": f1_score(y_pool, pr, average="macro"),
                 **{f"recall_{k}": (pr[(y_pool == k).values] == k).mean()
                    for k in C.CREDIT_SCORE_ORDER}})

# baseline banale, per dare una scala all'accuratezza
maj = y_pool.value_counts(normalize=True).max()
rows.append({"config": "baseline: predice sempre 'Standard'", "accuracy": maj,
             "macro_F1": f1_score(y_pool, np.full(len(y_pool), "Standard"),
                                  average="macro", zero_division=0),
             "recall_Poor": 0.0, "recall_Standard": 1.0, "recall_Good": 0.0})

t2 = pd.DataFrame(rows).set_index("config")
print("\n" + t2.round(4).to_string())
t2.round(4).to_csv(C.REPORTS / "diagnostica_t2_class_weight.csv")

# %%
# =========================================================================
# T3 — la curva di apprendimento e' ancora in salita?
# =========================================================================
print("\n" + "=" * 72)
print("T3. NUMEROSITA': curva di apprendimento")
print("=" * 72)

cv = make_cv()
sizes, train_sc, val_sc = learning_curve(
    build_pipeline(rf(), X_pool, scale=False), X_pool, y_pool, cv=cv, scoring=C.PRIMARY_METRIC,
    train_sizes=np.linspace(0.15, 1.0, 7), n_jobs=1, random_state=C.RANDOM_STATE)

lc = pd.DataFrame({"n_train": sizes.astype(int),
                   "train_macro_F1": train_sc.mean(axis=1),
                   "cv_macro_F1": val_sc.mean(axis=1),
                   "sd": val_sc.std(axis=1)})
lc["guadagno_vs_step_prec"] = lc["cv_macro_F1"].diff()
print("\n" + lc.round(4).to_string(index=False))
lc.round(5).to_csv(C.REPORTS / "diagnostica_t3_learning_curve.csv", index=False)

tail = lc["cv_macro_F1"].iloc[-1] - lc["cv_macro_F1"].iloc[-3]
print(f"\nguadagno negli ultimi due incrementi (da {lc['n_train'].iloc[-3]:,} a "
      f"{lc['n_train'].iloc[-1]:,} clienti): {tail:+.4f} macro-F1")
print("gap train - CV (overfitting residuo): "
      f"{lc['train_macro_F1'].iloc[-1] - lc['cv_macro_F1'].iloc[-1]:.4f}")

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(lc["n_train"], lc["train_macro_F1"], "o-", label="train", color="#c0392b")
ax.plot(lc["n_train"], lc["cv_macro_F1"], "o-", label="validazione (CV)", color="#4c72b0")
ax.fill_between(lc["n_train"], lc["cv_macro_F1"] - lc["sd"], lc["cv_macro_F1"] + lc["sd"],
                alpha=0.2, color="#4c72b0")
ax.set_xlabel("clienti nel training set"); ax.set_ylabel("macro-F1")
ax.set_title("Curva di apprendimento — Random Forest", fontsize=10)
ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(C.FIGURES / "15_learning_curve.png"); plt.close(fig)

# %%
# =========================================================================
# T4 — la pulizia di dominio sta buttando via segnale?
# =========================================================================
# Ipotesi: i valori corrotti (eta' -500, tassi al 5.797%) potrebbero essere
# stati iniettati in modo correlato al target. Se cosi' fosse, nullificarli
# costerebbe performance e chi NON pulisce otterrebbe numeri piu' alti.
print("\n" + "=" * 72)
print("T4. ABLATION: con e senza i vincoli di plausibilita' di dominio")
print("=" * 72)

# Anche qui il confronto vive in CV sul train pool: i clienti sono gli stessi
# dello split ufficiale (stessi Customer_ID, stesso ordinamento -> stessi fold),
# cambia solo se i vincoli di plausibilita' sono applicati o no.
pool_ids = set(pd.read_csv(C.ARTIFACTS / "train_pool.csv")[C.CUSTOMER_ID])

rows4 = []
for label, rules in [("con pulizia di dominio (progetto)", True),
                     ("senza pulizia di dominio", False)]:
    cu, _ = build_customer_dataset(
        raw, use_dispersion=C.USE_DISPERSION_FEATURES,
        use_ratios=C.USE_DOMAIN_RATIOS, apply_domain_rules=rules)
    sub = cu[cu[C.CUSTOMER_ID].isin(pool_ids)].reset_index(drop=True)
    Xc = sub.drop(columns=[C.TARGET, "_target_agreement", C.CUSTOMER_ID])
    yc = sub[C.TARGET]
    pr = cross_val_predict(build_pipeline(rf(), Xc, scale=False), Xc, yc,
                           cv=make_cv(), n_jobs=-1)
    rows4.append({"config": label, "accuracy": accuracy_score(yc, pr),
                  "macro_F1": f1_score(yc, pr, average="macro")})
    print(f"  {label:36s} accuracy = {rows4[-1]['accuracy']:.4f}  "
          f"macro-F1 = {rows4[-1]['macro_F1']:.4f}")

t4 = pd.DataFrame(rows4).set_index("config")
d4 = (t4["accuracy"].iloc[1] - t4["accuracy"].iloc[0]) * 100
print(f"\n>>> Non pulire varrebbe {d4:+.2f} punti di accuratezza.")
t4.round(4).to_csv(C.REPORTS / "diagnostica_t4_cleaning_ablation.csv")

# %%
# =========================================================================
# Tetto imposto dal rumore del label
# =========================================================================
print("\n" + "=" * 72)
print("Tetto strutturale: quanto e' auto-coerente il label?")
print("=" * 72)
cust = pd.read_csv(C.ARTIFACTS / "customers.csv")
agree = cust["_target_agreement"]
print(f"accordo medio del label mensile con la moda del cliente: {agree.mean():.1%}")
print(f"clienti con label costante sugli 8 mesi                : {(agree == 1).mean():.1%}")
print("\nSe il giudizio originale, sullo stesso cliente e con feature quasi\n"
      "identiche, cambia nel 15,6% dei mesi, nessun modello puo' raggiungere\n"
      "il 100%: una parte della relazione feature -> label e' stocastica.")
