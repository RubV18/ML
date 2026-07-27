"""Configurazione centrale del progetto Credit Score Classification."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "Data" / "train.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
for _p in (ARTIFACTS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

TARGET = "Credit_Score"
CUSTOMER_ID = "Customer_ID"

# Ordine ordinale del target: dal peggiore al migliore.
# Usato (a) per il tie-break prudenziale nell'aggregazione, (b) per l'ordine
# delle classi in matrici di confusione e report.
CREDIT_SCORE_ORDER = ["Poor", "Standard", "Good"]

# Metrica primaria, fissata PRIMA di qualunque fit (cfr. handoff).
PRIMARY_METRIC = "f1_macro"

# Bin d'età per il fairness check (proxy di gruppo protetto).
# Il bin più basso parte da 13 e non da 17: il dataset contiene clienti di 14-17
# anni con età stabile su tutti i mesi (cfr. audit). La fascia 60+ risulta vuota
# (età massima osservata: 56) → il check di fairness opera di fatto su 3 gruppi.
AGE_BINS = [13, 25, 40, 60, 200]
AGE_LABELS = ["<25", "25-40", "40-60", "60+"]

# Estensioni motivate dall'audit (vedi reports/01_data_audit.md).
# Impostare a False per tornare esattamente al set di feature del piano iniziale.
USE_DISPERSION_FEATURES = True   # volatilita'/picco delle colonne davvero time-varying
USE_DOMAIN_RATIOS = True         # debt-to-income, debt-service ratio
