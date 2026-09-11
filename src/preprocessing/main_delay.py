#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_delay.py
--------------
Procesa archivos events_masked_*.npz aplicando el adelanto fisiológico
de SPO2 respecto a Airflow (25 s por defecto).

Entrada:
    events_masked_<ID>.npz #333 creo que es solo  events_<ID>.npz 

Salida:
    delay_<ID>.npz
"""

from pathlib import Path
import numpy as np

from .delay import apply_spo2_delay
from .config import MASKED_EVENTS_RESULTS_DIR, DELAY_RESULTS_DIR, SPO2_DELAY_S


def process_npz(npz_path: Path, output_dir: Path):
    print(f"\n[INFO] Procesando delay para: {npz_path.name}")

    # Cargar NPZ
    data = np.load(npz_path, allow_pickle=True)

    airflow      = data["airflow"]
    spo2         = data["spo2"]
    mask_sleep   = data["mask_sleep"]
    mask_events  = data["mask_events"]
    fs           = int(data["fs"])

    # Aplicar delay fisiológico
    spo2_c, airflow_c, mask_sleep_c, mask_events_c = apply_spo2_delay(
        airflow, spo2, mask_sleep, mask_events, fs=fs, delay_s=SPO2_DELAY_S
    )

    # Nombre salida: delay_<ID>.npz
    original_id = npz_path.stem.replace("events_", "")
    out_path = output_dir / f"delay_{original_id}.npz"
    # Conservar metadatos originales (excepto señales)
    meta = {
        k: data[k] for k in data.files
        if k not in ("airflow", "spo2", "mask_sleep", "mask_events")
    }

    # Actualizar duración y número de muestras
    L = len(airflow_c)
    meta["n_samples_resampled"] = int(L)
    meta["duration_s"] = float(L / fs)

    # Guardar NPZ corregido
    np.savez_compressed(
        out_path,
        airflow=airflow_c,
        spo2=spo2_c,
        mask_sleep=mask_sleep_c.astype(np.uint8),
        mask_events=mask_events_c.astype(np.uint8),
        fs_delay=fs,
        **meta
    )

    print(f"[OK] Delay aplicado → {out_path.name}")
    print(f"   [INFO] Muestras finales: {len(airflow_c)}")

    #return "ok" ????


def main():
    input_dir  = Path(MASKED_EVENTS_RESULTS_DIR)
    output_dir = Path(DELAY_RESULTS_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Leyendo desde  : {input_dir.resolve()}")
    print(f"[INFO] Guardando en   : {output_dir.resolve()}")

    npz_files = sorted(input_dir.glob("events_*.npz"))

    print(f"[INFO] Encontrados {len(npz_files)} archivos events_*.npz.")

    for npz_path in npz_files:
        try:
            process_npz(npz_path, output_dir)
        except Exception as e:
            print(f"[ERROR] Falló {npz_path.name}: {e}")


if __name__ == "__main__":
    main()
