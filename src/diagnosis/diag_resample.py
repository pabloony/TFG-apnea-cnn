#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
diag_resample.py
----------------
Diagnóstico del re-muestreo:

- Analiza hasta 100 sujetos aleatorios
- Comprueba coherencia de longitudes
- Compara resample_spo2_exact vs resample_poly
- Imprime solo resumen estadístico (sin gráficos)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

from preprocessing.config import (
    DTA_RESULTS_DIR,
    RESAMPLED_RESULTS_DIR,
    FS_TARGET,
)
from preprocessing.resample import resample_spo2_exact


# =========================
# Utilidades
# =========================

def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def subject_from_dta(stem: str) -> str:
    return stem.replace("_dta_results", "")


def find_resampled(subject: str) -> Path | None:
    p = RESAMPLED_RESULTS_DIR / f"resampled_{subject}.npz"
    return p if p.exists() else None


# =========================
# Diagnóstico individual
# =========================

def analyze_one(dta_path: Path):
    dta = load_npz(dta_path)
    subject = subject_from_dta(dta_path.stem)

    out_path = find_resampled(subject)
    if out_path is None:
        return None

    out = load_npz(out_path)

    fs_spo2 = float(dta["fs_spo2"])
    duration_in = float(dta["duration_s"])

    # Longitud esperada
    n_expected = int(round(duration_in * FS_TARGET))
    n_out = int(out.get("n_samples_resampled", len(out["airflow"])))
    n_diff = n_out - n_expected

    spo2_out = out["spo2"].astype(float)

    # Método alternativo con resample_poly (referencia)
    if fs_spo2 == FS_TARGET:
        spo2_poly = dta["spo2"].astype(float)
    else:
        gcd = np.gcd(int(FS_TARGET), int(fs_spo2))
        up = int(FS_TARGET / gcd)
        down = int(fs_spo2 / gcd)
        spo2_poly = resample_poly(dta["spo2"].astype(float), up, down)

    # Igualar longitudes
    L = min(len(spo2_out), len(spo2_poly))
    if L == 0:
        return None

    diff = spo2_out[:L] - spo2_poly[:L]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {
        "subject": subject,
        "fs_spo2": fs_spo2,
        "n_diff": n_diff,
        "mae": mae,
        "rmse": rmse,
    }


# =========================
# Main
# =========================

def main(n_cases: int = 100, seed: int = 0):

    dta_files = sorted(Path(DTA_RESULTS_DIR).glob("*_dta_results.npz"))

    if not dta_files:
        print("[ERROR] No se encontraron archivos DTA.")
        return

    rng = np.random.default_rng(seed)

    n_cases = min(n_cases, len(dta_files))
    selected = rng.choice(dta_files, size=n_cases, replace=False)

    results = []

    for path in selected:
        r = analyze_one(path)
        if r is not None:
            results.append(r)

    if not results:
        print("[ERROR] No se pudo analizar ningún caso.")
        return

    # Convertir a arrays
    n_diff = np.array([r["n_diff"] for r in results])
    mae = np.array([r["mae"] for r in results])
    rmse = np.array([r["rmse"] for r in results])

    # =========================
    # Resumen global
    # =========================

    print("\n==============================")
    print(" DIAGNÓSTICO RESAMPLE (100)")
    print("==============================\n")

    print(f"Casos analizados: {len(results)}")

    print("\n--- Longitud vs esperada ---")
    print(f"mean(n_diff) = {np.mean(n_diff):.3f}")
    print(f"std(n_diff)  = {np.std(n_diff):.3f}")
    print(f"min/max      = {np.min(n_diff)} / {np.max(n_diff)}")
    print(f"percentiles  = {np.percentile(n_diff, [5, 50, 95])}")

    print("\n--- SpO2 exact vs poly ---")
    print(f"MAE  mean/std = {np.mean(mae):.5f} / {np.std(mae):.5f}")
    print(f"RMSE mean/std = {np.mean(rmse):.5f} / {np.std(rmse):.5f}")
    print(f"MAE percentiles = {np.percentile(mae, [5, 50, 95])}")

    # Top 5 peores casos
    worst = sorted(results, key=lambda x: x["mae"], reverse=True)[:5]

    print("\n--- Top 5 peores MAE ---")
    for w in worst:
        print(
            f"{w['subject']} | fs_spo2={w['fs_spo2']} | "
            f"n_diff={w['n_diff']} | MAE={w['mae']:.5f}"
        )

    print("\n[FIN DIAGNÓSTICO]\n")


if __name__ == "__main__":
    main()