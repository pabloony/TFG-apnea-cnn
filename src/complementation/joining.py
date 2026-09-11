#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
joining.py
----------
Une airflow y SpO2 en un solo bloque (2 x window_size) por segmento.

Entrada:
    airflow_windows : ndarray (N, W)
    spo2_windows    : ndarray (N, W)

Salida:
    joined_windows  : ndarray (N, 2, W)
"""

import numpy as np


def join_signals(airflow_windows: np.ndarray,
                 spo2_windows: np.ndarray) -> np.ndarray:
    """
    Une airflow y SpO2 en un array (N, 2, W).

    Se comprueba que ambas entradas tengan exactamente la misma forma.
    """

    if airflow_windows.shape != spo2_windows.shape:
        raise ValueError(
            f"Shapes distintas entre airflow y spo2: "
            f"airflow={airflow_windows.shape}, spo2={spo2_windows.shape}"
        )

    # Estructura final: (N, 2, W)
    joined = np.stack([airflow_windows, spo2_windows], axis=1)

    return joined