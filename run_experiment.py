#!/usr/bin/env python3
"""Relationship Intelligence Layer -- reproducible experiments.

    python run_experiment.py main     # all models x 5 seeds; C1 ring timing, G4 segments, G6 entity age
    python run_experiment.py main --preset hard   # the same on the HARD preset (paper headline setting)
    python run_experiment.py c2       # label delay x ring lifetime (x infra reuse) phase diagram
    python run_experiment.py c3       # infrastructure-reuse dose-response (leaky vs safe protocols)
    python run_experiment.py c4       # cost of evasion per feature family
    python run_experiment.py g5       # CI coverage study (i.i.d. vs ring-clustered bootstrap)
    python run_experiment.py examples # explanations + real neighbourhood plot for illustrative cases
    python run_experiment.py sensitivity  # generator-assumption sweep + hard preset (fix for circularity/ceiling)
    python run_experiment.py adaptive     # adaptive (greedy, model-aware) attacker for C4
    python run_experiment.py calibration  # sim-to-real: place IEEE-CIS on the synthetic reuse dial
    python run_experiment.py aml          # IBM AMLworld replication (needs data/AMLworld/)
    python run_experiment.py tgn_reference  # reference TGN (PyTorch Geometric) on the main setting
    python run_experiment.py ieee     # IEEE-CIS real-data replication (needs data/IEEE_CIS/)
    python run_experiment.py tvw      # C5 method: takedown-value weighting (synthetic + AMLworld + IEEE-CIS)
    python run_experiment.py tvw_gnn  # C5 on the temporal GNNs (GraphSAGE, GAT, CARE-style), hard preset
    python run_experiment.py originals_tuning  # small validation sweep for the original CARE-GNN / PC-GNN
    python run_experiment.py all      # every synthetic experiment (not ieee / aml / calibration)

Results are written to results/<experiment>/.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import src  # noqa: F401  (loads XGBoost before torch; see src/__init__.py)

EXPERIMENTS = ["main", "c2", "c3", "c4", "g5", "examples", "sensitivity", "adaptive", "calibration", "ieee", "aml", "tgn_reference", "tvw", "tvw_gnn", "originals_tuning"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=EXPERIMENTS + ["all"])
    ap.add_argument("--seeds", type=int, nargs="+", help="override the experiment's default seeds")
    ap.add_argument("--preset", choices=["default", "hard"], default="default",
                    help="generator preset for 'main' (results/main or results/main_hard)")
    args = ap.parse_args()
    real = ("ieee", "aml", "calibration", "tvw", "originals_tuning")
    todo = [e for e in EXPERIMENTS if e not in real] if args.experiment == "all" else [args.experiment]
    for name in todo:
        mod = __import__(f"experiments.{ {'c2': 'c2_label_delay', 'c3': 'c3_reuse', 'c4': 'c4_evasion', 'g5': 'g5_coverage', 'ieee': 'ieee_cis', 'adaptive': 'c4_adaptive', 'aml': 'amlworld'}.get(name, name) }", fromlist=["run"])
        kwargs = {"seeds": tuple(args.seeds)} if args.seeds and name not in ("g5", "ieee", "examples", "calibration", "aml") else {}
        if name == "main":
            kwargs["preset"] = args.preset
        mod.run(**kwargs)


if __name__ == "__main__":
    main()
