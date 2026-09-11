#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_labels.py
---------------
Segmenta las señales corregidas por delay en ventanas deslizantes de 10 s.
Descarta ventanas con mask_sleep = 0.
Etiqueta ventanas según mask_events (≥50% → 1).

Entrada:
    delay_<ID>.npz

Salida:
    segments_<ID>.npz

También calcula estadísticas útiles para Tukey:
    - n_segments_total
    - n_segments_pos
    - n_segments_neg
    - pct_positive
"""

from pathlib import Path
import numpy as np

from .labels import segment_signals
from .config import DELAY_RESULTS_DIR, SEGMENTS_RESULTS_DIR,COHORT_PREFIX


def process_npz(npz_path: Path, output_dir: Path):
    print(f"\n[INFO] Segmentando: {npz_path.name}")

    data = np.load(npz_path, allow_pickle=True)

    airflow      = data["airflow"]
    spo2         = data["spo2"]
    mask_sleep   = data["mask_sleep"]
    mask_events  = data["mask_events"]
    fs           = int(data.get("fs_delay", data.get("fs", 50)))

    # ---- Segmentación ----
    airflow_windows, spo2_windows, labels, win_times = segment_signals(
        airflow, spo2, mask_sleep, mask_events, fs
    )

    n_total = len(labels)
    n_pos   = int(labels.sum())
    n_neg   = int(n_total - n_pos)
    pct_pos = float(n_pos / n_total) if n_total > 0 else 0.0

    print(f"[OK] Ventanas generadas: {n_total}")
    print(f"     Positivas: {n_pos}, Negativas: {n_neg}, %Pos={pct_pos*100:.2f}%")

    # ---- Nombre salida ----
    ID = npz_path.stem.replace("delay_", "")
    out_path = output_dir / f"segments_{ID}.npz"

    # ---- Guardar ----
    np.savez_compressed(
        out_path,
        airflow_windows=airflow_windows,
        spo2_windows=spo2_windows,
        labels=labels.astype(np.uint8),
        win_times=win_times,
        fs=fs,
        # estadísticas para Tukey
        n_segments_total=n_total,
        n_segments_pos=n_pos,
        n_segments_neg=n_neg,
        pct_positive=pct_pos
    )

    print(f"[GUARDADO] → {out_path.name}")


def main():
    input_dir  = Path(DELAY_RESULTS_DIR)
    output_dir = Path(SEGMENTS_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Leyendo de : {input_dir.resolve()}")
    print(f"[INFO] Guardando en: {output_dir.resolve()}")

    npz_files = sorted(input_dir.glob(f"delay_{COHORT_PREFIX}*.npz"))
    print(f"[INFO] Encontrados {len(npz_files)} archivos delay_*.npz.")

    for npz_path in npz_files:
        try:
            process_npz(npz_path, output_dir)
        except Exception as e:
            print(f"[ERROR] Falló {npz_path.name}: {e}")


if __name__ == "__main__":
    main()
