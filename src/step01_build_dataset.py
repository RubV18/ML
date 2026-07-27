# %%
"""Step 1 — Cleaning, aggregazione per cliente, split 80/20 stratificato.

Esegue anche i controlli diagnostici che giustificano le scelte fatte in
data_prep.py (stabilita' del target, missing residui, missingness informativa).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C
from data_prep import build_customer_dataset

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 80)

# %%
raw = pd.read_csv(C.DATA_RAW, low_memory=False)
print(f"raw: {raw.shape[0]:,} righe x {raw.shape[1]} colonne, "
      f"{raw[C.CUSTOMER_ID].nunique():,} clienti")

# %%
cust, clean = build_customer_dataset(
    raw, use_dispersion=C.USE_DISPERSION_FEATURES, use_ratios=C.USE_DOMAIN_RATIOS
)
print(f"dataset cliente: {cust.shape[0]:,} righe x {cust.shape[1]} colonne")

# %%
# --- Diagnostica 1: quanto e' stabile il target entro cliente? -------------
agree = cust["_target_agreement"]
print("\n[D1] Stabilita' del label nei ~8 mesi dello stesso cliente")
print(f"  clienti con label costante : {(agree == 1).sum():,} ({(agree == 1).mean():.1%})")
print(f"  accordo medio con la moda  : {agree.mean():.1%}")
print(f"  accordo minimo             : {agree.min():.1%}")

# %%
# --- Diagnostica 1b: quanto nullifica ogni regola di plausibilita'? -------
rules = pd.Series(clean.attrs["nulled_by_rule"]).sort_values(ascending=False) * 100
print("\n[D1b] % di righe nullificate da ciascuna regola di dominio")
print(rules[rules > 0].round(2).to_string())

# %%
# --- Diagnostica 2: missing residui dopo l'aggregazione -------------------
miss = cust.drop(columns=["_target_agreement"]).isna().mean().sort_values(ascending=False)
print("\n[D2] % missing dopo aggregazione (top 10)")
print((miss.head(10) * 100).round(3).to_string())

# %%
# --- Diagnostica 3: la missingness a livello riga e' informativa? ---------
# Se la % di mesi mancanti per cliente non discrimina il target, i flag
# is_missing_X previsti nel piano non hanno segnale e non vanno aggiunti.
flag_cols = ["Monthly_Inhand_Salary", "Num_of_Delayed_Payment", "Credit_Mix",
             "Amount_invested_monthly", "Num_Credit_Inquiries"]
mfrac = clean.groupby(C.CUSTOMER_ID)[flag_cols].apply(lambda g: g.isna().mean())
mfrac = mfrac.join(cust.set_index(C.CUSTOMER_ID)[C.TARGET])
print("\n[D3] Frazione media di mesi mancanti, per classe del target")
print((mfrac.groupby(C.TARGET).mean() * 100).round(2).to_string())

# %%
# --- Split 80/20 stratificato, DOPO l'aggregazione ------------------------
# L'aggregazione garantisce gia' che un cliente non possa comparire in due
# split diversi (una riga = un cliente): niente leakage inter-split.
# _target_agreement resta nei file salvati come colonna DIAGNOSTICA (serve a
# capire se l'errore residuo si concentra dove il label stesso e' instabile);
# e' esclusa dalle feature in features.NON_FEATURES, quindi non entra nei modelli.
X = cust.drop(columns=[C.TARGET])
y = cust[C.TARGET]

X_pool, X_test, y_pool, y_test = train_test_split(
    X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
)
print(f"\ntrain pool: {len(X_pool):,}   test finale: {len(X_test):,}")
print("\ndistribuzione classi (%)")
print(pd.DataFrame({
    "totale": y.value_counts(normalize=True) * 100,
    "pool": y_pool.value_counts(normalize=True) * 100,
    "test": y_test.value_counts(normalize=True) * 100,
}).round(2).loc[C.CREDIT_SCORE_ORDER].to_string())

# %%
cust.to_csv(C.ARTIFACTS / "customers.csv", index=False)
X_pool.assign(**{C.TARGET: y_pool}).to_csv(C.ARTIFACTS / "train_pool.csv", index=False)
X_test.assign(**{C.TARGET: y_test}).to_csv(C.ARTIFACTS / "test.csv", index=False)
print(f"\nsalvato in {C.ARTIFACTS}")
