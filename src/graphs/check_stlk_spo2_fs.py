#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_stlk_spo2_fs.py
----------------------
Comprueba la fs_spo2 (y de paso fs_airflow) que traen guardados varios
archivos STLK*_dta_results.npz elegidos al azar, para detectar si alguno
se desvía de lo esperado (500 Hz).

Uso:
    python check_stlk_spo2_fs.py
    python check_stlk_spo2_fs.py --n 15
    python check_stlk_spo2_fs.py --seed 3
"""

import argparse
import random
from pathlib import Path

import numpy as np

DTA_RESULTS_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/dta_results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="Nº de archivos STLK a comprobar (por defecto 10).")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para el sorteo (reproducibilidad).")
    args = parser.parse_args()

    if not DTA_RESULTS_DIR.exists():
        raise FileNotFoundError(f"No existe DTA_RESULTS_DIR: {DTA_RESULTS_DIR}")

    stlk_files = sorted(DTA_RESULTS_DIR.glob("STLK*_dta_results.npz"))
    print(f"[INFO] {len(stlk_files)} archivos STLK*_dta_results.npz encontrados en total.")
    if not stlk_files:
        return

    rng = random.Random(args.seed)
    n = min(args.n, len(stlk_files))
    sample = rng.sample(stlk_files, n)
    sample.sort()

    print(f"[INFO] Comprobando {n} al azar:\n")
    print(f"{'archivo':30s}  {'fs_spo2':>10s}  {'fs_airflow':>10s}  {'len(spo2)':>10s}  {'len(airflow)':>12s}  {'duration_s':>10s}")
    print("-" * 92)

    fs_spo2_vals = []
    for f in sample:
        try:
            d = np.load(f, allow_pickle=True)
            fs_spo2 = float(d["fs_spo2"])
            fs_airflow = float(d["fs_airflow"])
            len_spo2 = len(d["spo2"])
            len_airflow = len(d["airflow"])
            dur = float(d["duration_s"])
            fs_spo2_vals.append(fs_spo2)
            print(f"{f.name:30s}  {fs_spo2:10.1f}  {fs_airflow:10.1f}  {len_spo2:10d}  {len_airflow:12d}  {dur:10.1f}")
        except Exception as e:
            print(f"{f.name:30s}  [ERROR] No se pudo leer: {e}")

    if fs_spo2_vals:
        uniq = sorted(set(fs_spo2_vals))
        print(f"\n[INFO] Valores distintos de fs_spo2 encontrados: {uniq}")
        if len(uniq) > 1:
            print("[WARN] Hay más de un valor de fs_spo2 entre los archivos comprobados -- revisar.")
        elif uniq[0] != 500.0:
            print(f"[WARN] fs_spo2 = {uniq[0]} Hz, distinto de los 500 Hz esperados para STLK -- revisar.")
        else:
            print("[OK] Todos los archivos comprobados tienen fs_spo2 = 500 Hz, como se esperaba.")


if __name__ == "__main__":
    main()