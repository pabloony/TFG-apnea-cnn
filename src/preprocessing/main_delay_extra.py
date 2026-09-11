#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_delay.py
--------------
Procesa archivos events_*.npz aplicando el adelanto fisiológico
de SPO2 respecto a Airflow (25 s por defecto).

Entrada:
    
    events_<ID>.npz

Salida:
    delay_<ID>.npz
"""

from pathlib import Path
import numpy as np

from .delay import apply_spo2_delay
from .config import DELAY_RESULTS_DIR,MASKED_EVENTS_RESULTS_DIR,SPO2_DELAY_S,COHORT_PREFIX 

def process_npz(npz_path: Path, output_dir: Path):
    print(f"\n[INFO] Procesando delay para: {npz_path.name}")

    data = np.load(npz_path, allow_pickle=True)

    airflow = data["airflow"]
    spo2 = data["spo2"]
    mask_sleep   = data["mask_sleep"]
    mask_events  = data["mask_events"]
    fs = int(data["fs"])

    # Longitudes originales antes del delay
    print(f"   [DEBUG] Longitudes originales:")
    print(f"           airflow = {len(airflow)}")
    print(f"           spo2    = {len(spo2)}")
    print(f"           mask_sleep = {len(mask_sleep)}")
    print(f"           mask_events = {len(mask_events)}")

    # Aplicar delay fisiológico
    """airflow_c, spo2_c, mask_sleep_c, mask_events_c = apply_spo2_delay(
        airflow, spo2, mask_sleep, mask_events, fs=fs, delay_s=SPO2_DELAY_S
    )
    #333 canales intercambiados 
    """
    #333 corregido
    spo2_c, airflow_c, mask_sleep_c, mask_events_c = apply_spo2_delay(
    airflow, spo2, mask_sleep, mask_events, fs=fs, delay_s=SPO2_DELAY_S
    )

    # Longitudes justo DESPUÉS del shift,
    # pero ANTES del recorte final L
    print(f"   [DEBUG] Tras aplicar shift/cortes iniciales:")
    print(f"           airflow_c = {len(airflow_c)}")
    print(f"           spo2_c    = {len(spo2_c)}")
    print(f"           mask_sleep_c    = {len(mask_sleep_c)}")
    print(f"           mask_events_c    = {len(mask_events_c)}")

    # Calcular L y recortar
    L = min(len(airflow_c), len(spo2_c), len(mask_sleep_c), len(mask_events_c))

    print(f"   [DEBUG] L = {L} (mínimo común tras corte)")

    airflow_c = airflow_c[:L]
    spo2_c = spo2_c[:L]
    mask_sleep_c = mask_sleep_c[:L]
    mask_events_c = mask_events_c[:L]

    # Nombre salida: delay_<ID>.npz
    original_id = npz_path.stem.replace("events_", "")
    out_path = output_dir / f"delay_{original_id}.npz"
    
    # Metadatos originales
    meta = {k: data[k] for k in data.files if k not in ("airflow", "spo2", "mask_sleep", "mask_events")}
    meta["n_samples_resampled"] = int(L)
    meta["duration_s"] = float(L / fs)
    
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
    print(f"     [INFO] Muestras finales: {len(airflow_c)}")



def main():
    input_dir  = Path(MASKED_EVENTS_RESULTS_DIR)
    output_dir = Path(DELAY_RESULTS_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Leyendo desde  : {input_dir.resolve()}")
    print(f"[INFO] Guardando en   : {output_dir.resolve()}")

    npz_files = sorted(input_dir.glob(f"events_{COHORT_PREFIX}*.npz"))
    print(f"[INFO] Encontrados {len(npz_files)} archivos events_*.npz.\n")

    for npz_path in npz_files:
        try:
            process_npz(npz_path, output_dir)
        except Exception as e:
            print(f"[ERROR] Falló {npz_path.name}: {e}")


if __name__ == "__main__":
    main()


