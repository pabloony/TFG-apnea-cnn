#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
delay.py
"""

import numpy as np


def apply_spo2_delay(airflow,spo2,mask_sleep,mask_events,fs,delay_s):

    """Aplica el retardo de SpO₂ respecto a Airflow y máscara de sueño."""

    shift = int(delay_s * fs)
   
    spo2_delayed = spo2[shift:]
    airflow_truncated = airflow[:-shift]
    mask_sleep_truncated = mask_sleep[:-shift]
    mask_events_truncated = mask_events[:-shift]

    L = min(len(airflow_truncated), len(spo2_delayed), len(mask_sleep_truncated), len(mask_events_truncated))
    
   # 333 
   # OJO: el orden de salida es:
   # spo2, airflow, mask_sleep, mask_events
    return(
       spo2_delayed[:L],
       airflow_truncated[:L],
       mask_sleep_truncated[:L],
       mask_events_truncated[:L]
    )
   


