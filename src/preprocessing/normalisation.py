#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalisation.py
----------------
Funciones de normalización de sujetos:
 - Detección de NaNs
 - Cálculo de TST
 - Eliminación por TST mínimo
 - Eliminación por ratio excesivo de positivos
 - Aplicación de Tukey para limitar positivos
 - Actualización de estructuras tras recorte
"""

import numpy as np
from typing import Dict, Tuple


def has_nans(data_dict: Dict) -> bool:
    """Devuelve True si airflow o spo2 contienen NaNs."""
    return (
        np.isnan(data_dict["airflow"]).any() or
        np.isnan(data_dict["spo2"]).any()
    )


def compute_tst_hours(n_segments: int, step_sec: float = 1.0) -> float:
    """
    Calcula TST en horas.
    Con step=1s, TST_hours = n_segments / 3600.
    """
    return (n_segments * step_sec) / 3600.0


def compute_positive_ratio(n_pos: int, n_total: int) -> float:
    """Devuelve el ratio positivo (0–1)."""
    return n_pos / n_total if n_total > 0 else 0.0


def tukey_limit(values: np.ndarray) -> float:
    """
    Calcula el límite superior según Tukey:
        limit = Q3 + 1.5 * IQR
    values debe contener n_pos por sujeto.
    """
    Q1, Q3 = np.percentile(values, [25, 75])
    IQR = Q3 - Q1
    limit = Q3 + 1.5 * IQR
    return limit


def apply_tukey_to_subject(
        airflow_windows: np.ndarray,
        spo2_windows: np.ndarray,
        labels: np.ndarray,
        max_pos_allowed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recorta SOLO los segmentos positivos si superan el máximo permitido.
    Los negativos se dejan completos.
    """

    # Índices positivos y negativos
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = len(pos_idx)

    # Si no se excede límite → devolver tal cual
    if n_pos <= max_pos_allowed:
        return airflow_windows, spo2_windows, labels

    # Seleccionar aleatoriamente los positivos que se mantienen
    keep_pos_idx = np.random.choice(pos_idx, size=max_pos_allowed, replace=False)

    # Combinar con todos los negativos (no se recortan)
    final_idx = np.concatenate([neg_idx, keep_pos_idx])
    final_idx.sort()

    return (
        airflow_windows[final_idx],
        spo2_windows[final_idx],
        labels[final_idx]
    )


def spo2_invalid_segment_mask(
        spo2_windows: np.ndarray,
        min_valid: float = 50.0,
        max_valid: float = 100.0,
    ) -> np.ndarray:
    """
    Devuelve una máscara booleana de shape (n_segments,) donde:
    - True  -> segmento válido
    - False -> segmento inválido por contener al menos un valor de SpO2
               fuera del rango [min_valid, max_valid].
    """
    invalid = (spo2_windows < min_valid) | (spo2_windows > max_valid)
    valid_mask = ~np.any(invalid, axis=1)
    return valid_mask


def filter_invalid_spo2_segments(
        airflow_windows: np.ndarray,
        spo2_windows: np.ndarray,
        labels: np.ndarray,
        win_times: np.ndarray,
        min_valid: float = 50.0,
        max_valid: float = 100.0,
    ):
    """
    Elimina segmentos que contienen al menos un valor de SpO2 no fisiológico.
    Mantiene alineados airflow, spo2, labels y win_times.
    """
    valid_mask = spo2_invalid_segment_mask(
        spo2_windows,
        min_valid=min_valid,
        max_valid=max_valid
    )

    return (
        airflow_windows[valid_mask],
        spo2_windows[valid_mask],
        labels[valid_mask],
        win_times[valid_mask],
        int((~valid_mask).sum())
    )