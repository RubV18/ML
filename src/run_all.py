#!/usr/bin/env python
"""Esegue l'intera pipeline nell'ordine corretto.

Gli step hanno dipendenze di sequenza: step01 produce gli split usati da tutti,
step04 e step05 producono i modelli che step06-08 valutano e spiegano.

    python src/run_all.py              # tutto
    python src/run_all.py --from 06    # riparte da step06
    python src/run_all.py --only 03 04 # solo alcuni step
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

STEPS = [
    ("01", "step01_build_dataset.py", "cleaning, aggregazione per cliente, split 80/20"),
    ("02", "step02_eda.py", "EDA sul solo train pool"),
    ("03", "step03_baselines.py", "Step 0: gap lineare / non lineare sui default"),
    ("04", "step04_filone_a.py", "Filone A: tuning completo + regola 1-SE  (~15 min)"),
    ("05", "step05_filone_b.py", "Filone B: L1, albero, diagnostica statsmodels"),
    ("06", "step06_final_test.py", "valutazione finale sul test set (una sola volta)"),
    ("07", "step07_shap.py", "SHAP globale + tre casi locali"),
    ("08", "step08_fairness.py", "fairness su eta' + unawareness test"),
    ("09", "step09_diagnostics.py", "diagnostica: leakage, learning curve, ablation"),
    ("10", "step10_summary.py", "sintesi per paper/slide: RISULTATI.md + figure selezionate"),
    ("11", "step11_scaling.py",
     "appendice: campione allargato, curva estesa + protocollo 60/20/20 (opz., ~55 min)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", metavar="NN",
                    help="riparte dallo step indicato")
    ap.add_argument("--only", nargs="+", metavar="NN", help="esegue solo questi step")
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [s for s in steps if s[0] in args.only]
    elif args.start:
        idx = next((i for i, s in enumerate(steps) if s[0] == args.start), None)
        if idx is None:
            print(f"step '{args.start}' inesistente", file=sys.stderr)
            return 2
        steps = steps[idx:]

    t_start = time.perf_counter()
    for num, script, desc in steps:
        print(f"\n{'='*72}\n[{num}] {desc}\n{'='*72}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(SRC / script)])
        if r.returncode != 0:
            print(f"\n>>> step {num} fallito (exit {r.returncode}); pipeline "
                  "interrotta.", file=sys.stderr)
            return r.returncode
        print(f"\n[{num}] completato in {time.perf_counter() - t0:.0f}s", flush=True)

    print(f"\nPipeline completata in {(time.perf_counter() - t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
