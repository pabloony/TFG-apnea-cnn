#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_sleep.py
--------------
Genera una máscara binaria de sueño (1 = sueño, 0 = no sueño) 
a partir de las anotaciones CSV asociadas a un EDF.
Se basa en los Start Time (HH:MM:SS) y  en Duration,


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

def build_sleep_mask(csv_path, edf_info,events_to_mark, fs=50):
    """Construye la máscara binaria de sueño a partir del CSV y la info del EDF."""
   
    #Carga del CSV 
    csv_path = Path(csv_path)
    df = read_csv_simple(csv_path)

    #Lista indices de filas con Custom User Event 4 (evento de sueño)
    cue4_row = df.index[df["Event"].str.contains("Custom User Event 4")]

    if len(cue4_row) > 0:
        start_idx = cue4_row[0] # primer índice donde aparece CUE 4
        print(start_idx) #333 antes solo estaba print # para verificar que se detecta correctamente, si no se detecta, start_idx no se define y da error, por eso el else
    else:
        # NO cambiamos el main, así que NO devolvemos None
        print(f"[WARNING] No se encontró 'Custom User Event 4' en {csv_path.name}. Usando CSV completo.")
        start_idx = 0
    
    df = df.iloc[start_idx:].reset_index(drop=True) # Reseteamos índices tras recortar filas previas a CUE 4

    if df.empty:
        print(f"[WARNING] CSV quedó vacío tras recorte en {csv_path.name}. Generando máscara de ceros.")
        total = int(edf_info["duration_s"] * fs) #duration_s es el tiempo total de las señales en segundos despues de ser resampleadas
        return np.zeros(total, dtype=np.uint8)
    
    #start_s =! edf_info["start_time"] !!!
    df["start_s"] = df["Start Time"].apply(hms_to_seconds) # No confundir columna "Start Time" con edf_info["start_time"] → "2023-10-01 22:15:00+00:00"
    #df["Start Time"] = 19:44:48,...,...	df["start_s"] = 71088,...,... [listas]

    edf_hms = str(edf_info["start_time"]).split(" ")[1]  # "2023-10-01 22:15:00+00:00" -> "22:15:00+00:00"
    edf_start_s = hms_to_seconds(edf_hms.split("+")[0])  # "22:15:00+00:00" -> "22:15:00" -> segundos desde medianoche

   

    rel_times =[] # lista de tiempos relativos al inicio del EDF en segundos
    for t in df["start_s"]:
        if t < edf_start_s: # evento al día siguiente
            t += 24*3600 # sumar 24 horas en segundos
        rel_times.append(t-edf_start_s) # tiempo relativo al inicio del EDF

    df["rel_s"]= rel_times # nueva columna con tiempos relativos, desde inicio , en segundos

    df = df.sort_values("rel_s").reset_index(drop=True) # reset_index(drop=True) para evitar que se añada una columna 'index' extra

    total_samples = int(edf_info["duration_s"] * fs)
    mask = np.zeros(total_samples, dtype=np.uint8)

    for i in range(len(df)):
        row = df.iloc[i] # fila iterada

        t0 = row ["rel_s"] # tiempo de inicio relativo en segundos
        dur = row ["Duration"] # duración en segundos
        t1 = t0 + dur # tiempo de fin relativo en segundos

        i0 = int(t0 * fs) # índice inicial en muestras
        i1 = int(t1 * fs) # índice final en muestras

        if i0 >= total_samples: # si el índice inicial está fuera del rango total de muestras
            continue
        if i1 > total_samples: # ajustar el índice final si excede el total
            i1 = total_samples

        # marcar solo eventos de sueño
        if row["Event"] in events_to_mark:
            mask[i0:i1] = 1 # marcar entre indices i0 e i1 como sueño (1)
    return mask

     
