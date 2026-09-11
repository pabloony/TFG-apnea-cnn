#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_long_events.py
------------------------
Re-escanea los delay_<ID>.npz de STNF, pero esta vez guarda también el
sujeto y la posición (inicio/fin en segundos dentro del registro) de cada
evento, para poder identificar concretamente cuáles son los eventos
anómalamente largos (>120s) y en qué sujetos aparecen.

No reemplaza analyze_event_durations.py, es un script aparte solo para
este diagnóstico puntual.
"""

from pathlib import Path
import numpy as np

BASE_PROJECT = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0"
DELAY_RESULTS_DIR = Path(f"{BASE_PROJECT}/data/module_results/delay_data/")
COHORT_PREFIX = "STNF"

LONG_EVENT_THRESHOLD_S = 120.0  # 2 minutos, fisiológicamente sospechoso


def find_event_runs_with_position(mask: np.ndarray, fs: float):
    """
    Igual que en analyze_event_durations.py, pero devuelve también
    (start_s, end_s) de cada evento, no solo la duración.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return []

    diff = np.diff(mask.astype(np.int16))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0] == 1:
        starts = np.r_[0, starts]
    if mask[-1] == 1:
        ends = np.r_[ends, len(mask)]

    results = []
    for s, e in zip(starts, ends):
        dur_s = (e - s) / fs
        results.append((s / fs, e / fs, dur_s))
    return results


def main():
    npz_files = sorted(DELAY_RESULTS_DIR.glob(f"delay_{COHORT_PREFIX}*.npz"))
    print(f"[INFO] Re-escaneando {len(npz_files)} ficheros para buscar eventos > {LONG_EVENT_THRESHOLD_S}s\n")

    long_events = []  # (subject_id, start_s, end_s, dur_s)
    total_events = 0

    for npz_path in npz_files:
        subject_id = npz_path.stem.replace("delay_", "")
        d = np.load(npz_path, allow_pickle=True)
        mask_events = d["mask_events"]
        fs = float(d["fs_delay"]) if "fs_delay" in d.files else float(d.get("fs", 50))

        events = find_event_runs_with_position(mask_events, fs)
        total_events += len(events)

        for start_s, end_s, dur_s in events:
            if dur_s > LONG_EVENT_THRESHOLD_S:
                long_events.append((subject_id, start_s, end_s, dur_s))

    print(f"[RESUMEN] Total de eventos: {total_events}")
    print(f"[RESUMEN] Eventos > {LONG_EVENT_THRESHOLD_S}s: {len(long_events)} "
          f"({100*len(long_events)/total_events:.3f}% del total)\n")

    if long_events:
        long_events.sort(key=lambda x: -x[3])
        print(f"--- Top eventos largos (máx. 20 mostrados) ---")
        print(f"{'Sujeto':<15} {'Inicio (s)':>12} {'Fin (s)':>12} {'Duración (s)':>14} {'Duración (min)':>16}")
        for subj, start_s, end_s, dur_s in long_events[:20]:
            print(f"{subj:<15} {start_s:>12.1f} {end_s:>12.1f} {dur_s:>14.1f} {dur_s/60:>16.2f}")

        if len(long_events) > 20:
            print(f"... y {len(long_events) - 20} más.")

        # Cuántos sujetos distintos están afectados
        affected_subjects = set(e[0] for e in long_events)
        print(f"\n[INFO] Sujetos distintos afectados: {len(affected_subjects)} de {len(npz_files)}")
        print(f"[INFO] Lista de sujetos afectados: {sorted(affected_subjects)}")
    else:
        print("[OK] No se encontraron eventos sospechosamente largos.")


if __name__ == "__main__":
    main()