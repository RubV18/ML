"""Pipeline di cleaning + aggregazione per cliente.

Tutte le scelte qui sotto sono motivate da domini/audit, non da regole
statistiche cieche (niente capping IQR automatico): un valore viene
dichiarato *implausibile* solo se viola un vincolo del dominio del credito,
e in quel caso diventa missing, non un valore troncato.

Output: una riga per Customer_ID.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CREDIT_SCORE_ORDER, CUSTOMER_ID, TARGET

# --------------------------------------------------------------------------- #
# 1. Colonne
# --------------------------------------------------------------------------- #
DROP_COLS = ["ID", "Name", "SSN"]          # nessun valore predittivo (+ PII fittizia)
HELPER_COLS = ["Month"]                    # serve solo per ordinare/aggregare

# Colonne numeriche salvate come stringa e "sporcate" con caratteri spuri.
DIRTY_NUMERIC = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    "Monthly_Balance",
]
CLEAN_NUMERIC = [
    "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Delay_from_due_date", "Num_Credit_Inquiries",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
]
CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]

# Stringhe sentinella che il dataset usa al posto di NaN (verificate nell'audit).
SENTINELS = {
    "Occupation": ["_______"],
    "Credit_Mix": ["_"],
    "Payment_of_Min_Amount": ["NM"],          # "Not Mentioned" -> informazione assente
    "Payment_Behaviour": ["!@9#%8"],
    "Amount_invested_monthly": ["__10000__"],
    "Monthly_Balance": ["__-333333333333333333333333333__"],
    "Changed_Credit_Limit": ["_"],
}

# Vincoli di plausibilita' di dominio: (min, max) inclusivi. None = nessun vincolo.
PLAUSIBLE_RANGE = {
    # 14 e non 18: l'audit mostra 688 clienti (5.5%) con eta' 14-17 *stabile* su
    # tutti gli 8 mesi -> sono record generati, non rumore iniettato. Nullificarli
    # perderebbe dati reali; il fatto che siano sotto l'eta' legale per il credito
    # e' un limite del dataset, documentato nella model card.
    "Age": (14, 100),
    "Annual_Income": (0, 1_000_000),      # oltre 1M -> outlier iniettato, non reddito retail
    "Monthly_Inhand_Salary": (0, 100_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (1, 50),             # tasso % annuo su credito retail
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 60),    # su ~8 mesi di storia osservata
    "Num_Credit_Inquiries": (0, 50),
    "Outstanding_Debt": (0, None),
    "Credit_Utilization_Ratio": (0, 100),
    "Total_EMI_per_month": (0, None),
    "Amount_invested_monthly": (0, None),
    "Monthly_Balance": (0, None),
    "Delay_from_due_date": (-30, 200),    # negativo = pagamento anticipato: plausibile
    "Changed_Credit_Limit": (None, None), # negativo = limite ridotto: plausibile
}

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August"]

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Not Specified", "Payday Loan",
    "Personal Loan", "Student Loan",
]


def _slug(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


# --------------------------------------------------------------------------- #
# 2. Cleaning a livello di riga
# --------------------------------------------------------------------------- #
def _to_numeric(s: pd.Series) -> pd.Series:
    """Strip dei caratteri non numerici -> to_numeric(errors='coerce')."""
    out = (
        s.astype("string")
        .str.strip()
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace({"": pd.NA, "-": pd.NA, ".": pd.NA})
    )
    return pd.to_numeric(out, errors="coerce")


def _parse_credit_history_age(s: pd.Series) -> pd.Series:
    """'22 Years and 1 Months' -> 265 (mesi totali)."""
    parts = s.astype("string").str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?")
    return parts[0].astype("Float64") * 12 + parts[1].astype("Float64")


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning riga per riga, prima dell'aggregazione."""
    df = df.copy()

    # a) sentinelle -> NaN (fatto PRIMA della coercizione numerica, perche'
    #    alcune sentinelle sono numeri validi una volta ripuliti: '__10000__')
    for col, vals in SENTINELS.items():
        if col in df.columns:
            df[col] = df[col].replace(vals, np.nan)

    # b) colonne numeriche sporche -> float
    for col in DIRTY_NUMERIC:
        df[col] = _to_numeric(df[col])

    # c) Credit_History_Age testuale -> mesi
    df["Credit_History_Age_Months"] = _parse_credit_history_age(df["Credit_History_Age"])
    df = df.drop(columns=["Credit_History_Age"])

    # d) vincoli di plausibilita' di dominio -> i violatori diventano missing.
    #    Traccio quanto nullifica ogni regola: una regola che azzera troppo e'
    #    un vincolo sbagliato, non un dato sporco (cfr. il caso Age 14-17).
    nulled = {}
    for col, (lo, hi) in PLAUSIBLE_RANGE.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        nulled[col] = float((mask & df[col].notna()).mean())
        df.loc[mask, col] = np.nan

    # e) regola di dominio inter-colonna: la rata mensile totale non puo'
    #    superare il reddito mensile disponibile (debt-service ratio > 1).
    monthly_income = df["Monthly_Inhand_Salary"].fillna(df["Annual_Income"] / 12)
    emi_mask = df["Total_EMI_per_month"] > monthly_income
    nulled["Total_EMI_per_month (DSR>1)"] = float((emi_mask & df["Total_EMI_per_month"].notna()).mean())
    df.loc[emi_mask, "Total_EMI_per_month"] = np.nan
    df.attrs["nulled_by_rule"] = nulled

    # f) ordinamento temporale esplicito
    df["_month_idx"] = df["Month"].map({m: i for i, m in enumerate(MONTH_ORDER)})
    return df.sort_values([CUSTOMER_ID, "_month_idx"])


# --------------------------------------------------------------------------- #
# 3. Aggregazione per cliente
# --------------------------------------------------------------------------- #
def _mode_with_order(s: pd.Series, order: list[str] | None = None):
    """Moda; in caso di parita' si sceglie la classe *piu' conservativa*
    (la prima in `order`, che va dal peggiore al migliore). Senza `order`
    il tie-break e' alfabetico: deterministico e senza semantica implicita."""
    s = s.dropna()
    if s.empty:
        return np.nan
    counts = s.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    if len(top) == 1:
        return top[0]
    if order is not None:
        for lvl in order:
            if lvl in top:
                return lvl
    return sorted(top)[0]


def aggregate_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Una riga per Customer_ID: mediana per i numerici, moda per i categoriali."""
    g = df.groupby(CUSTOMER_ID, sort=True)
    numeric_cols = DIRTY_NUMERIC + CLEAN_NUMERIC + ["Credit_History_Age_Months"]

    out = g[numeric_cols].median()

    for col in CATEGORICAL:
        out[col] = g[col].apply(_mode_with_order)

    # Target: moda sui mesi. Il label NON e' costante nel tempo per lo stesso
    # cliente (cfr. audit); in caso di parita' si assegna la classe peggiore
    # (principio prudenziale del credit risk: in dubbio, non sottostimare il rischio).
    out[TARGET] = g[TARGET].apply(lambda s: _mode_with_order(s, CREDIT_SCORE_ORDER))

    # Diagnostica sulla stabilita' del label (usata nel report, non come feature).
    out["_target_agreement"] = g[TARGET].apply(lambda s: s.value_counts().iloc[0] / len(s))

    # Type_of_Loan e' costante entro cliente (verificato): prendo il primo non nullo.
    out["Type_of_Loan"] = g["Type_of_Loan"].apply(
        lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
    )
    return out.reset_index()


# --------------------------------------------------------------------------- #
# 4. Feature engineering
# --------------------------------------------------------------------------- #
def add_loan_type_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot sui tipi di prestito + conteggio dei tipi distinti.

    Il *tipo* di debito e' piu' informativo del numero nel credit risk
    (un Payday Loan segnala un rischio diverso da un Auto Loan), quindi il
    conteggio resta come feature aggiuntiva, non sostitutiva.
    NaN in Type_of_Loan e' missing *strutturale* (= nessun prestito, verificato
    contro Num_of_Loan == 0): diventa multi-hot tutto a zero, non imputazione.
    """
    agg = agg.copy()
    txt = agg["Type_of_Loan"].fillna("")
    for lt in LOAN_TYPES:
        agg[f"has_{_slug(lt)}"] = txt.str.contains(lt, regex=False).astype(int)
    agg["n_loan_types"] = agg[[f"has_{_slug(lt)}" for lt in LOAN_TYPES]].sum(axis=1)
    return agg.drop(columns=["Type_of_Loan"])


def add_dispersion_features(clean: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Volatilita'/picco delle sole colonne realmente time-varying.

    L'audit mostra che gran parte delle colonne e' costante entro cliente
    (le differenze sono rumore iniettato): per quelle la dispersione non ha
    significato. Restano davvero variabili nel tempo utilizzo del credito,
    ritardi e inquiries: nel credit risk il *picco* di delinquenza e la
    *volatilita'* dell'utilizzo sono segnali distinti dal loro livello medio.
    """
    g = clean.groupby(CUSTOMER_ID, sort=True)
    extra = pd.DataFrame({
        "Credit_Utilization_Ratio_std": g["Credit_Utilization_Ratio"].std(),
        "Delay_from_due_date_max": g["Delay_from_due_date"].max(),
        "Num_Credit_Inquiries_max": g["Num_Credit_Inquiries"].max(),
    }).reset_index()
    return agg.merge(extra, on=CUSTOMER_ID, how="left")


def add_domain_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    """Due indicatori canonici del credit risk, costruiti da colonne gia' presenti."""
    agg = agg.copy()
    agg["debt_to_income"] = agg["Outstanding_Debt"] / agg["Annual_Income"].replace(0, np.nan)
    monthly_income = agg["Monthly_Inhand_Salary"].fillna(agg["Annual_Income"] / 12)
    agg["debt_service_ratio"] = agg["Total_EMI_per_month"] / monthly_income.replace(0, np.nan)
    return agg


def build_customer_dataset(
    df_raw: pd.DataFrame,
    use_dispersion: bool = True,
    use_ratios: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw (100k righe) -> dataset a livello cliente (~12.5k righe).

    Ritorna (dataset_cliente, dataframe_pulito_a_livello_riga)."""
    df = df_raw.drop(columns=[c for c in DROP_COLS if c in df_raw.columns])
    clean = clean_rows(df)
    agg = aggregate_by_customer(clean)
    agg = add_loan_type_features(agg)
    if use_dispersion:
        agg = add_dispersion_features(clean, agg)
    if use_ratios:
        agg = add_domain_ratios(agg)
    return agg, clean
