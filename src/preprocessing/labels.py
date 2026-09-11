#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
labels.py
-----------
Funciones para segmentar señales en ventanas deslizantes de 10 s/15 s con 90 % de solapamiento.
Descarta ventanas donde mask_sleep contenga algún 0.
Etiqueta cada ventana según mask_events (≥x% → 1, si no 0).
"""

import numpy as np
from .config import WINDOW_S, OVERLAP_S, THR_EVENT

def segment_signals(airflow, spo2, mask_sleep, mask_events, fs, win_sec=WINDOW_S, overlap=OVERLAP_S, thr_event=THR_EVENT):

    N = len(airflow) # longitud total de la señal(número de muestras)
    
    win_len = int(win_sec * fs) # longitud de ventana en muestras
    overlap_len= int(overlap * fs)
    step = int(win_len - overlap_len) # avance entre ventanas
    if step <= 0:
        raise ValueError("El solapamiento debe ser menor que la longitud de la ventana.")
    
    airflow_segmets, spo2_segments,event,window = [], [], [], [] # creamos listas vacías 

    for start in range(0, N - win_len + 1, step): # start, actualiza. +1 para incluir la última ventana
        end = start + win_len #end y start , indices variables

        seg_air = airflow[start:end]
        seg_spo2 = spo2[start:end]
        seg_mask_sleep = mask_sleep[start:end]
        seg_mask_events = mask_events[start:end]

        if np.any(seg_mask_sleep == 0): # Si hay algún 0 en la máscara de sueño, descartamos la ventana
            continue
       
        frac_event = seg_mask_events.mean() # media de la máscara de eventos en la ventana
        label = 0
        
        if frac_event >= thr_event:
            label = 1
        else:
            label = 0
        
        airflow_segmets.append(seg_air)
        spo2_segments.append(seg_spo2)
        event.append(label)
        window.append((start/fs, end/fs))
        
    return (
        np.array(airflow_segmets), # Ventanas de airflow
        np.array(spo2_segments), # Ventanas de SpO₂
        np.array(event), # Etiquetas de eventos (0/1)
        np.array(window) # Tiempos de inicio y fin de cada ventana         
    )  


       