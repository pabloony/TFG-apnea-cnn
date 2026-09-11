#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_w20_tradeoff.py
------------------------
Reutiliza event_durations_STNF.npy (ya generado por analyze_event_durations.py)
para responder: si subo W de 15 a 20, ¿qué eventos se pierden con
thr_event=0.50, y se recuperan bajando thr? ¿Son eventos "típicos" o cola larga?

No entrena nada. No toca GPU. Corre en segundos.

Uso:
    python analyze_w20_tradeoff.py
"""

from pathlib import Path
import numpy as np

BASE_PROJECT = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0"
DURATIONS_NPY = Path(f"{BASE_PROJECT}/data/module_results/event_duration_analysis/event_durations_STNF.npy")

WINDOWS = [15, 20]
THRESHOLDS = [0.30, 0.40, 0.50]


def main():
    if not DURATIONS_NPY.exists():
        print(f"[ERROR] No existe {DURATIONS_NPY}. "
              f"Corre antes analyze_event_durations.py (ya lo hiciste para F7, "
              f"comprueba la ruta o busca con: find . -name 'event_durations_STNF.npy')")
        return

    d = np.load(DURATIONS_NPY)
    n = len(d)
    print(f"[INFO] {n} eventos cargados de {DURATIONS_NPY.name}\n")

    print("=" * 70)
    print("TABLA 1: % eventos excluidos por (W, thr_event)")
    print("=" * 70)
    header = f"{'W':>5s} | " + " | ".join([f"thr={t:.2f}" for t in THRESHOLDS])
    print(header)
    print("-" * len(header))
    for W in WINDOWS:
        frac_max = np.minimum(1.0, d / W)
        row = [100.0 * np.mean(frac_max < t) for t in THRESHOLDS]
        print(f"{W:>5d} | " + " | ".join([f"{v:6.2f}%" for v in row]))

    print()
    print("=" * 70)
    print("TABLA 2: eventos EXCLUIDOS en W=20/thr=0.50 que SÍ estaban")
    print("         incluidos en W=15/thr=0.50 (los que 'perderías' al subir W)")
    print("=" * 70)
    incl_w15 = np.minimum(1.0, d / 15) >= 0.50
    incl_w20 = np.minimum(1.0, d / 20) >= 0.50
    perdidos = incl_w15 & (~incl_w20)
    n_perdidos = perdidos.sum()
    print(f"N eventos perdidos: {n_perdidos} ({100*n_perdidos/n:.2f}% del total)")
    if n_perdidos > 0:
        dur_perdidos = d[perdidos]
        print(f"  Duración media  : {dur_perdidos.mean():.2f}s")
        print(f"  Duración mediana: {np.median(dur_perdidos):.2f}s")
        print(f"  Rango           : [{dur_perdidos.min():.2f}, {dur_perdidos.max():.2f}]s")
        print(f"  (para comparar: mediana global de TODOS los eventos = {np.median(d):.2f}s)")

    print()
    print("=" * 70)
    print("TABLA 3: si bajo thr_event con W=20, ¿cuántos de esos perdidos recupero?")
    print("=" * 70)
    for t in THRESHOLDS:
        incl_w20_t = np.minimum(1.0, d / 20) >= t
        recuperados = perdidos & incl_w20_t
        print(f"  W=20, thr={t:.2f} -> recupera {recuperados.sum()}/{n_perdidos} "
              f"de los perdidos ({100*recuperados.sum()/max(n_perdidos,1):.1f}%)")

    print()
    print("[INFO] Pégame esta salida completa y la interpreto.")


if __name__ == "__main__":
    main()