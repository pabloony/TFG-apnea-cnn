#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_mask.py
------------
Paso 2 del Preprocessing:
- Lee los archivos remuestreados (.npz) generados en main_resample.py
- Busca el CSV correspondiente con las anotaciones de sueño
- Construye la máscara binaria de sueño (0/1)
- Guarda un nuevo NPZ con: airflow, spo2, mask, fs, duración, start_time, n_samples_resampled

Autor: Pablo E. Ortega Ureña (TFG - UMA)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA_CSV_DIR, RESAMPLED_RESULTS_DIR,SLEEP_EVENTS,MASKED_RESULTS_DIR,COHORT_PREFIX
from .mask_sleep_STLK import build_sleep_mask #Depende si estamos usando STNF o STLK, cambiar la importación a build_sleep_mask_STNF

def process_file(npz_path : Path, output_dir : Path, csv_dir : Path):
    """Procesa un archivo resampleado y genera su máscara de sueño."""
    
    print(f"\n[INFO] Procesando: {npz_path.name}")
    
    # Cargar archivo .npz remuestreado
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {npz_path.name}: {e}")
        return "error"
    
    # Verificar las claves requeridas
    required_keys = {
        "airflow", "spo2",
        "fs_airflow", "fs_spo2",
        "duration_s", "start_time",
        "n_samples_resampled"
    }

    for key in required_keys:
        if key not in data:
            print(f"[WARNING] Falta clave '{key}' en {npz_path.name}. Saltando archivo...")
            return "skip"

    # Reconstruir diccionarios signals e info
    airflow = data["airflow"]
    spo2 = data["spo2"]

    info = {
        "fs_airflow": int(data["fs_airflow"]),
        "fs_spo2": int(data["fs_spo2"]),
        "duration_s": float(data["duration_s"]),
        "start_time": data["start_time"],
        "n_samples_resampled": int(data["n_samples_resampled"])
    }

    fs = info["fs_airflow"] # Asumimos fs_airflow == fs_spo2 = 50 Hz tras resampleo

    """Módulo busca el CSV correspondiente al NPZ actual"""
    base_name = npz_path.stem.replace("resampled_","")
    csv_path = csv_dir / f"{base_name}.csv"

    if not csv_path.exists():
         print(f"[ERROR] No se encontró CSV para {npz_path.name} → {csv_path.name} (skip)")
         return "error"
    
    """Construir la máscara de sueño"""
    try:
        mask = build_sleep_mask(csv_path,info,events_to_mark=SLEEP_EVENTS,fs=fs)
    except Exception as e:
        print(f"[ERROR] Falló build_sleep_mask para {csv_path.name}: {e}")
        return "error"
    
    """Guardar nuevo archivo NPZ con la máscara"""
    output_path = output_dir / f"masked_{npz_path.name}"

    if output_path.exists():
        print(f"[INFO] El archivo {output_path.name} ya existe. (skip)")
        return "skip"
    
    try:
        np.savez_compressed(
            output_path,
            airflow=airflow,
            spo2=spo2,
            mask=mask.astype(np.uint8),
            fs=fs,
            duration_s=info["duration_s"],
            start_time=str(info["start_time"]),
            n_samples_resampled=info["n_samples_resampled"]
        )
        print(f"[SUCCESS] Guardado: {output_path.name}")
        return "ok"
    except Exception as e:
        print(f"[ERROR] No se pudo guardar {output_path.name}: {e}")
        return "error"

def main():

    input_dir = Path(RESAMPLED_RESULTS_DIR)  # carpeta donde se guardaron los "resampled_*.npz"
    output_dir = Path(MASKED_RESULTS_DIR)
    csv_dir = Path(DATA_CSV_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Buscando archivos resampleados en: {input_dir.resolve()}")
    print(f"[INFO] Buscando CSVs en: {csv_dir.resolve()}")

    npz_files = list(input_dir.glob(f"resampled_{COHORT_PREFIX}*.npz"))

    if not npz_files:
        print("[WARNING] No hay archivos resampled_*.npz para procesar.")
        return "error"

    print(f"[INFO] {len(npz_files)} archivos encontrados.\n")

    # Contadores
    procesados_ok = 0
    saltados = 0
    errores = 0

    for npz_path in npz_files:
        result = process_file(npz_path, output_dir, csv_dir)

        if result == "ok":
            procesados_ok += 1
        elif result == "skip":
            saltados += 1
        elif result == "error":
            errores += 1

    """Mensaje resumen final"""

    print("\n" + "-" * 50)
    print("RESUMEN DEL PROCESAMIENTO")
    print("-" * 50)
    print(f"Procesados correctamente : {procesados_ok}")
    print(f"Saltados                 : {saltados}")
    print(f"Errores                  : {errores}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()