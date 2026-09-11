#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_mask_events.py
-----------------------
Diagnóstico de la estructura real de mask_events en un único delay_<ID>.npz,
antes de intentar el análisis de duraciones a escala.
"""

from pathlib import Path
import numpy as np

DELAY_RESULTS_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/delay_data/")
SUBJECT_ID = "STNF00497"  # el que te dio el warning

npz_path = DELAY_RESULTS_DIR / f"delay_{SUBJECT_ID}.npz"
d = np.load(npz_path, allow_pickle=True)

print(f"[INFO] Claves disponibles en el npz: {d.files}")

mask_events = d["mask_events"]
print(f"\n[mask_events] type={type(mask_events)}")
print(f"[mask_events] dtype={mask_events.dtype}")
print(f"[mask_events] shape={mask_events.shape}")

if mask_events.dtype == object:
    print(f"\n[INFO] Es un array de objetos con {len(mask_events)} elementos.")
    for i, sub in enumerate(mask_events):
        sub_arr = np.asarray(sub)
        print(f"   elemento {i}: shape={sub_arr.shape}, dtype={sub_arr.dtype}, "
              f"sample={sub_arr[:5] if sub_arr.size > 0 else 'VACÍO'}")
else:
    print(f"\n[INFO] Es un array plano normal. Primeros 20 valores: {mask_events[:20]}")
    print(f"[INFO] Valores únicos: {np.unique(mask_events)}")

# También miramos airflow y fs_delay por si acaso tienen la misma estructura rara
for key in ("airflow", "spo2", "mask_sleep", "fs_delay"):
    if key in d.files:
        val = d[key]
        print(f"\n[{key}] type={type(val)}, shape={getattr(val, 'shape', 'N/A')}, "
              f"dtype={getattr(val, 'dtype', 'N/A')}")