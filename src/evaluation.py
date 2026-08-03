"""Caricamento degli split, tuning in CV e regole di selezione.

Raccoglie in un solo posto le operazioni che altrimenti sarebbero ricopiate in
ogni step: leggere il pool/test scartando le colonne non-feature, lanciare una
GridSearchCV registrando media ed errore standard fra i fold, e applicare la
regola 1-SE.

Motivo per cui vive qui e non dentro i singoli step: la regola di selezione e'
una *decisione metodologica* del progetto, dichiarata a priori. Se fosse
ricopiata in tre file potrebbe divergere in tre varianti senza che nessuno se ne
accorga.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import config as C
from features import NON_FEATURES, make_preprocessor


# --------------------------------------------------------------------------- #
# Caricamento degli split
# --------------------------------------------------------------------------- #
def load_split(which: str = "pool", with_diagnostics: bool = False):
    """Ritorna (X, y) per 'pool' o 'test'.

    Le colonne in features.NON_FEATURES (Customer_ID, target, diagnostiche)
    sono escluse da X: e' l'unico punto del progetto che decide cosa e' feature.
    """
    fname = {"pool": "train_pool.csv", "test": "test.csv"}[which]
    df = pd.read_csv(C.ARTIFACTS / fname)
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns])
    y = df[C.TARGET]
    return (X, y, df) if with_diagnostics else (X, y)


def make_cv() -> StratifiedKFold:
    """La CV usata ovunque nel progetto: stessi fold, stesso seed."""
    return StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True,
                           random_state=C.RANDOM_STATE)


def build_pipeline(estimator, X: pd.DataFrame, scale: bool) -> Pipeline:
    """Preprocessing + modello. `scale` va acceso per i modelli che dipendono
    dalla scala delle feature (lineari penalizzati, SVM); per gli alberi e'
    inerte."""
    return Pipeline([("prep", make_preprocessor(X, scale=scale)),
                     ("clf", estimator)])


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def tune(name: str, estimator, grid: dict, X, y, cv=None, scale: bool = False,
         verbose: bool = True) -> dict:
    """GridSearchCV su macro-F1, con media ed errore standard fra i fold.

    L'errore standard e' sd/sqrt(k): quantifica l'incertezza *sulla media* dei
    k fold, ed e' cio' che la regola 1-SE usa come soglia.
    """
    cv = cv or make_cv()
    t0 = time.perf_counter()
    gs = GridSearchCV(build_pipeline(estimator, X, scale), grid,
                      scoring=C.PRIMARY_METRIC, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X, y)
    i = gs.best_index_
    out = {
        "modello": name,
        "macro_F1": float(gs.cv_results_["mean_test_score"][i]),
        "sd_folds": float(gs.cv_results_["std_test_score"][i]),
        "SE": float(gs.cv_results_["std_test_score"][i]) / np.sqrt(cv.get_n_splits()),
        "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        "n_config": len(gs.cv_results_["mean_test_score"]),
        "sec": time.perf_counter() - t0,
        "estimator": gs.best_estimator_,
    }
    if verbose:
        print(f"{name:26s} macro-F1 = {out['macro_F1']:.4f} (SE {out['SE']:.4f})  "
              f"{out['best_params']}  [{out['sec']:.0f}s, {out['n_config']} config]")
    return out


def results_frame(results: list[dict]) -> pd.DataFrame:
    """Tabella ordinata per performance, senza gli oggetti stimatore."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"}
                       for r in results])
    return df.set_index("modello").sort_values("macro_F1", ascending=False)


# --------------------------------------------------------------------------- #
# Regola di selezione (dichiarata a priori)
# --------------------------------------------------------------------------- #
def one_se_selection(df: pd.DataFrame, complexity: dict[str, int],
                     n_se: float = 1.0) -> dict:
    """One-standard-error rule.

    Fra i modelli il cui macro-F1 medio in CV resta entro `n_se` errori standard
    dal migliore, si sceglie il **piu' semplice** secondo `complexity`.
    E' un criterio lessicografico (prima il vincolo di performance, poi la
    semplicita'), non una somma pesata: non richiede di tarare un lambda fra
    accuratezza e interpretabilita'.
    """
    best = df["macro_F1"].idxmax()
    soglia = df.loc[best, "macro_F1"] - n_se * df.loc[best, "SE"]
    ammessi = df[df["macro_F1"] >= soglia]
    scelto = min(ammessi.index, key=lambda m: complexity.get(m, 99))
    return {"migliore": best, "soglia": float(soglia),
            "ammessi": list(ammessi.index), "scelto": scelto,
            "prezzo_semplicita": float(df.loc[best, "macro_F1"]
                                       - df.loc[scelto, "macro_F1"])}


def se_sensitivity(df: pd.DataFrame, ks=(1, 2, 3, 4)) -> pd.DataFrame:
    """Quanto e' fragile la regola 1-SE al variare della soglia?

    Con molti fold e molti campioni l'errore standard e' piccolo e la regola
    degenera nel 'vince il migliore': questa tabella lo rende esplicito invece
    di lasciarlo implicito.
    """
    best = df["macro_F1"].max()
    se = df.loc[df["macro_F1"].idxmax(), "SE"]
    return pd.DataFrame([
        {"soglia": f"{k} SE", "valore": best - k * se,
         "ammessi": ", ".join(df[df["macro_F1"] >= best - k * se].index)}
        for k in ks
    ]).set_index("soglia")
