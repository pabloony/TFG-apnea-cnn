#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_sleep_STLK.py, versión si Custom User 4
--------------
Genera una máscara binaria de sueño (1 = sueño, 0 = no sueño) 
a partir de las anotaciones CSV asociadas a un EDF.
Se basa SOLO en los Start Time (HH:MM:SS) y NO en Duration,
porque los eventos en el CSV pueden solaparse o interrumpirse.

Autor: Pablo E. Ortega Ureña
"""

import numpy as np
import pandas as pd
from pathlib import Path




def hms_to_seconds(hms : str):
    """Convierte una cadena HH:MM:SS a segundos desde medianoche."""
    h,m,s = hms.strip().split(":") # Realmente no es necesario strip(), no hay espacio en las CSV que he visto hasta ahora
    return int(h)*3600 + int(m)*60 + int(s)

def read_csv_simple(path):
    """Lee el CSV y devuelve solo Start Time, Duration, Event."""
    df = pd.read_csv(path,sep=",",on_bad_lines="skip")
    df = df.iloc[:, :3].copy() # df.iloc[filas,columnas] (: = todas) , copy() para evitar SettingWithCopyWarning
    df.columns = ["Start Time", "Duration", "Event"] # Renombrar columnas
    df["Event"] = df["Event"].astype(str).str.strip()  #astype(str) pasa a string, str.strip() quita espacios, " Stage1 " → "Stage1"
    return df

def build_sleep_mask(csv_path, edf_info, events_to_mark, fs=50):
    """Construye la máscara binaria de sueño a partir del CSV y la info del EDF."""
   
    # Carga del CSV 
    csv_path = Path(csv_path)
    df = read_csv_simple(csv_path)

    # En esta nueva base: CSV y EDF empiezan a la misma hora -> NO recortamos por 'Custom User Event 4'
    if df.empty:
        print(f"[WARNING] CSV vacío en {csv_path.name}. Generando máscara de ceros.")
        total = int(edf_info["duration_s"] * fs)
        return np.zeros(total, dtype=np.uint8)

    # Convertir Start Time (CSV) a segundos desde medianoche
    df["start_s"] = df["Start Time"].apply(hms_to_seconds)

    # Extraer hora de inicio del EDF desde edf_info["start_time"]
    edf_hms = str(edf_info["start_time"]).split(" ")[1]      # "22:15:00+00:00"
    edf_start_s = hms_to_seconds(edf_hms.split("+")[0])      # "22:15:00" -> segundos

    # (Opcional) Warning si el primer evento del CSV no coincide con el inicio del EDF (mismo HH:MM:SS)
    first_csv_hms = str(df.iloc[0]["Start Time"]).strip()
    first_csv_s = hms_to_seconds(first_csv_hms)
    if first_csv_s != edf_start_s:
        # Puede ser normal si el CSV empieza en el primer evento y no exactamente en t=0,
        # pero en esta base tú esperas igualdad, así que lo avisamos.
        print(
            f"[WARNING] Inicio CSV ({first_csv_hms}) != inicio EDF ({edf_hms.split('+')[0]}) "
            f"en {csv_path.name}. Se continuará igualmente."
        )

    # Calcular tiempos relativos al inicio del EDF (manejo de cruce de medianoche)
    rel_times = []
    for t in df["start_s"]:
        if t < edf_start_s:
            t += 24 * 3600
        rel_times.append(t - edf_start_s)

    df["rel_s"] = rel_times
    df = df.sort_values("rel_s").reset_index(drop=True)

    total_samples = int(edf_info["duration_s"] * fs)
    mask = np.zeros(total_samples, dtype=np.uint8)

    # Construir máscara por intervalos [t0, t0+Duration]
    for i in range(len(df)):
        row = df.iloc[i]

        t0 = float(row["rel_s"])
        dur = float(row["Duration"])
        t1 = t0 + dur

        i0 = int(t0 * fs)
        i1 = int(t1 * fs)

        if i0 >= total_samples:
            continue
        if i1 > total_samples:
            i1 = total_samples
        if i1 <= i0:
            continue

        if row["Event"] in events_to_mark:
            mask[i0:i1] = 1

    return mask

     
