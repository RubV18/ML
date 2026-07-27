"""Definizione dello spazio delle feature e del preprocessing condiviso."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
NON_FEATURES = ["Customer_ID", "Credit_Score", "_target_agreement"]

# Feature che, nel dominio reale, sono *output* di un processo di scoring a monte
# (il credit mix e il tasso applicato dipendono gia' dal merito creditizio).
# Non vengono rimosse - fanno parte del dataset - ma il loro peso va letto con
# cautela: e' correlazione con una valutazione preesistente, non causalita'.
BUREAU_DERIVED = ["Credit_Mix", "Interest_Rate"]


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Ritorna (colonne numeriche, colonne categoriali) fra le feature."""
    feats = [c for c in df.columns if c not in NON_FEATURES]
    cat = [c for c in feats if c in CATEGORICAL]
    num = [c for c in feats if c not in cat]
    return num, cat


def make_preprocessor(df: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Imputazione + (opzionale) standardizzazione + one-hot.

    Dopo l'aggregazione per cliente non restano missing (cfr. step 1), ma
    l'imputer resta nella pipeline per robustezza su dati nuovi ed e' fittato
    dentro la CV, quindi non introduce leakage.
    Lo scaling serve ai modelli lineari penalizzati (L1/L2 penalizzano i
    coefficienti: senza scala comune la penalita' non e' confrontabile fra
    feature); per gli alberi e' inerte ma innocuo.
    """
    num, cat = split_columns(df)
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
