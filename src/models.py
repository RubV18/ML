"""Costruttori dei modelli, con terminologia coerente.

Nota terminologica: per un problema di *classificazione* i modelli lineari
penalizzati si chiamano "Logistic Regression con penalita' L2" e "... L1".
"Ridge" e "Lasso" sono i nomi degli analoghi nella regressione lineare
continua e non vanno usati qui.

Nell'API sklearn >= 1.8 la penalita' si esprime con `l1_ratio`
(0 = L2 pura, 1 = L1 pura); il vecchio argomento `penalty` e' deprecato.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from config import RANDOM_STATE


def logistic_l2(C: float = 1.0, **kw) -> LogisticRegression:
    """Logistic Regression multinomiale con penalita' L2."""
    return LogisticRegression(
        l1_ratio=0.0, C=C, solver="lbfgs", class_weight="balanced",
        max_iter=5000, random_state=RANDOM_STATE, **kw
    )


def logistic_l1(C: float = 1.0, **kw) -> LogisticRegression:
    """Logistic Regression multinomiale con penalita' L1 (coefficienti sparsi).

    Richiede il solver `saga`, l'unico che supporta L1 nel caso multinomiale.
    """
    return LogisticRegression(
        l1_ratio=1.0, C=C, solver="saga", class_weight="balanced",
        max_iter=8000, random_state=RANDOM_STATE, **kw
    )


# Ordinamento di complessita' dichiarato a priori: serve alla regola di scelta
# "il modello piu' semplice che raggiunge la performance necessaria".
COMPLEXITY_RANK = {
    "Logistic Regression (L1)": 0,
    "Logistic Regression (L2)": 1,
    "Decision Tree": 2,
    "Random Forest": 3,
    "Gradient Boosting (hist)": 4,
}
