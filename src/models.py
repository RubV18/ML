"""Catalogo dei modelli candidati: costruttori, griglie, ordine di complessita'.

Un solo posto in cui sono definiti i modelli confrontati, cosi' che lo Step 0
(default) e il tuning del Filone A guardino per costruzione la *stessa*
famiglia di candidati.

Nota terminologica: per un problema di classificazione i modelli lineari
penalizzati si chiamano "Logistic Regression con penalita' L2" e "... L1".
"Ridge" e "Lasso" sono i nomi degli analoghi nella regressione lineare continua
e non vanno usati qui.

Nell'API sklearn >= 1.8 la penalita' si esprime con `l1_ratio` (0 = L2 pura,
1 = L1 pura); il vecchio argomento `penalty` e' deprecato. Attenzione: `C`
moltiplica la *loss*, non la penalita' -> C grande = regolarizzazione debole.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


# --------------------------------------------------------------------------- #
# Costruttori dei modelli lineari
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Candidati per lo Step 0 e per il tuning del Filone A
# --------------------------------------------------------------------------- #
# `scale`: True per i modelli che dipendono dalla scala delle feature. Per le
# SVM non e' opzionale — il kernel RBF si basa su distanze euclidee, quindi una
# feature su scala grande dominerebbe il kernel.
#
# `grid`: griglia di tuning. `default`: configurazione usata nello Step 0, dove
# i modelli girano volutamente senza tuning per rispondere a una domanda
# diagnostica ("quanto vale la non linearita'?") al costo piu' basso possibile.
CANDIDATES: dict[str, dict] = {
    "Logistic Regression (L2)": {
        "estimator": logistic_l2(),
        "scale": True,
        "grid": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
    },
    "Linear SVC": {
        "estimator": LinearSVC(class_weight="balanced", max_iter=20000,
                               random_state=RANDOM_STATE),
        "scale": True,
        "grid": {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(class_weight="balanced",
                                            random_state=RANDOM_STATE),
        "scale": False,
        "grid": {"clf__max_depth": [3, 4, 5, 6, 8, 12, None],
                 "clf__min_samples_leaf": [1, 10, 50, 100]},
        # Nello Step 0 l'albero gira a profondita' 4 fissata: non e' un ottimo,
        # e' un riferimento "albero leggibile" scelto a priori per la diagnosi.
        "default": DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                          random_state=RANDOM_STATE),
        "default_label": "Decision Tree (depth 4)",
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=1),
        "scale": False,
        "grid": {"clf__n_estimators": [300, 600],
                 "clf__max_depth": [None, 12, 20],
                 "clf__min_samples_leaf": [1, 3, 10],
                 "clf__max_features": ["sqrt", 0.4]},
    },
    "Gradient Boosting (hist)": {
        "estimator": HistGradientBoostingClassifier(class_weight="balanced",
                                                    random_state=RANDOM_STATE),
        "scale": False,
        "grid": {"clf__learning_rate": [0.05, 0.1],
                 "clf__max_leaf_nodes": [15, 31, 63],
                 "clf__max_iter": [200, 400],
                 "clf__l2_regularization": [0.0, 1.0]},
    },
    "SVC (kernel RBF)": {
        "estimator": SVC(kernel="rbf", class_weight="balanced",
                         random_state=RANDOM_STATE),
        "scale": True,
        "grid": {"clf__C": [0.3, 1.0, 3.0, 10.0, 30.0],
                 "clf__gamma": ["scale", 0.003, 0.01, 0.03]},
    },
}


# --------------------------------------------------------------------------- #
# Ordine di complessita'
# --------------------------------------------------------------------------- #
# Dichiarato a priori, e ordinato per **costo di interpretabilita'**, non per
# numero di parametri. Serve alla regola "il modello piu' semplice che
# raggiunge la performance necessaria".
#
# Razionale delle posizioni meno ovvie:
#  - Linear SVC dopo la logistica: stessa classe di ipotesi, ma senza
#    probabilita' native, quindi meno utilizzabile per motivare una decisione.
#  - SVC RBF in fondo, dopo gli ensemble di alberi: per RF e boosting esiste
#    TreeSHAP, esatto e veloce; per un kernel RBF servirebbe KernelSHAP, che e'
#    un'approssimazione e costa ordini di grandezza in piu'.
COMPLEXITY_RANK = {
    "Logistic Regression (L1)": 0,
    "Logistic Regression (L2)": 1,
    "Linear SVC": 2,
    "Decision Tree": 3,
    "Random Forest": 4,
    "Gradient Boosting (hist)": 5,
    "SVC (kernel RBF)": 6,
}

# Modelli che nello Step 0 servono come riferimento lineare, per misurare il gap
# lineare / non lineare.
LINEAR_MODELS = {"Logistic Regression (L2)", "Linear SVC"}
