"""Regenerate the current thesis figures/tables from the portable data bundle.

Writes to ausgabe/ by default. --output lets you choose a separate directory;
--output . writes into the thesis. Training data and logs are never required.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ("fig_konzepte", []), ("fig_phase1_uebersicht", []), ("fig_phase1_verluste", []),
    ("fig_inshop10", []), ("fig_phase2", []), ("fig_geometrie_weitere", []),
    ("fig_geometrie", []), ("fig_verlauf", []), ("fig_lp_cars", []),
    ("fig_retrieval", ["758,5348", "dino,sup,lp", "retrieval_ergebnisse"]),
    ("tab_hauptteil", []), ("tab_anhang", []), ("tab_extern", []),
    ("tab_phase02", []), ("tab_phase2_concat", []), ("tab_phase2_k8", []),
    ("tab_slade_sweeps", []), ("tab_aufwand", []), ("k4_ensemble", []), ("fig_verzoegerung", []),
    ("tab_hpo", []), ("tab_suchraeume", []), ("tab_datensaetze", []), ("zahlen_ergebnisse", []),
    ("erstelle_uebersicht", []),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "daten")
    p.add_argument("--output", type=Path, default=ROOT / "ausgabe")
    p.add_argument("--only", nargs="+", choices=[n for n, _ in JOBS])
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # UTF-8-Modus: Ausgaben und Dateien der Skripte auch unter Windows (sonst cp1252) in UTF-8
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               THESIS_DATA_DIR=str(args.data.resolve()), THESIS_OUTPUT_DIR=str(output),
               MPLBACKEND="Agg", MPLCONFIGDIR=str(output / ".matplotlib"),
               STIL_PREVIEW=str(output / "vorschau"))
    src = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(src / "pruefe_daten.py"), "--report", str(output / "pruefung.json")], env=env, check=True)
    logdir = output / "protokoll"
    logdir.mkdir(exist_ok=True)
    only = set(args.only or [])
    if "fig_geometrie" in only:  # liest die Korrelationen, die fig_geometrie_weitere.py schreibt
        only.add("fig_geometrie_weitere")
    jobs = [(n, a) for n, a in JOBS if args.only is None or n in only]
    for name, extra in jobs:
        print(name, flush=True)
        with (logdir / (name + ".txt")).open("w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, str(src / (name + ".py")), *extra], env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{name} failed. See {logdir / (name + '.txt')}")
    print(f"Generated {len(jobs)} groups of figures/tables in {output}")


if __name__ == "__main__":
    main()
