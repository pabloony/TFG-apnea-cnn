#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resample.py
-----------
Funciones para re-muestreo de las señales DTA (airflow y SpO₂)
a una frecuencia objetivo común.

Usa scipy.signal.resample_poly para mayor estabilidad y velocidad.
Para ver comparaciones de otros tipos de resample consultar carpeta de pruebas en scbi.
"""

import numpy as np
from scipy.signal import resample_poly


def resample_spo2_exact(data,fs_original,fs_target):
    """Re-muestrea SpO₂ asegurando que la longitud final sea exacta. Diseñado para señales discretas""" 
    if fs_original == fs_target:
        return data  # No es necesario re-muestrear 
     # Calcular  mcd y los factores de up y down
    gcd = np.gcd(int(fs_target),int(fs_original))
    up = int(fs_target / gcd)
    down = int(fs_original / gcd)

    upsample = np.repeat(data,up)
    downsampled = upsample[::down]

    return downsampled
    

def resample_signal(data,fs_original,fs_target):
    """Re-muestrea una señal 1D a una frecuencia objetivo."""
    if fs_original == fs_target:
        return data  # No es necesario re-muestrear
    
    # Calcular  mcd y los factores de up y down
    gcd = np.gcd(int(fs_target),int(fs_original))
    up = int(fs_target / gcd)
    down = int(fs_original / gcd)

    # Re-muestrear usando resample_poly(x, up, down)
    resampled_data = resample_poly(data,up,down)

    return resampled_data

def resample_pair(signals,info,fs_target = 50):
    """
    Re-muestrea simultáneamente las señales de airflow y SpO₂
    a una frecuencia objetivo común (fs_target).

    Parámetros
    ----------
    signals : dict
        Diccionario con:
            - "airflow": señal de flujo aéreo
            - "spo2": señal de SpO₂

    info : dict
        Diccionario con metadatos originales:
            - "fs_airflow"
            - "fs_spo2"
            - "duration_s"
            - "start_time"
            ...

    fs_target : float
        Frecuencia objetivo para ambas señales (default = 50 Hz).

    Returns
    -------
    signals_resampled : dict
        Señales re-muestreadas y alineadas.
    
    info_updated : dict
        Copia de info con frecuencias actualizadas y nueva longitud.
    """
   
   
    airflow = signals["airflow"]
    spo2 = signals["spo2"]

    fs_airflow = info["fs_airflow"]
    fs_spo2 = info["fs_spo2"]

    # Re-muestreo de ambas señales
    airflow_resampled =resample_signal(airflow,fs_airflow,fs_target)
    spo2_resampled = resample_spo2_exact(spo2,fs_spo2,fs_target)

    # Alineamiento: truncar a la longitud mínima común
    L = min(len(airflow_resampled),len(spo2_resampled))
    airflow_resampled = airflow_resampled[:L]
    spo2_resampled = spo2_resampled[:L]

    # Nueva duración en segundos, despeues de truncar y resamplear
    duration_resampled = L / fs_target # Nueva duración en segundos, despeues de truncar y resamplear


    # Diccionario con nuevas señales (solo podemos updatear info, no signals)
    signals_resampled = {
        "airflow": airflow_resampled,
        "spo2": spo2_resampled  
    }

    # Metadatos actualizados , la razon de no hacer un nuevo dic nuevo es mantener info que no cambia despues de resample
    info_updated = info.copy()
    info_updated.update({
    "fs_airflow": float(fs_target),
    "fs_spo2": float(fs_target),
    "fs_target": float(fs_target),
    "n_samples_resampled": int(L),
    "duration_s": float(duration_resampled)
    })

    return signals_resampled,info_updated

    





  