#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
event_counter.py
----------------
FASE 9 — Contador de eventos respiratorios a partir de una secuencia binaria
por-ventana (0/1) ordenada temporalmente.

Módulo PURO: solo funciones, sin rutas ni config dentro. Todos los parámetros
(tolerancia, mínimo de ventanas, ancho de ventana, stride) se pasan como
argumentos. Así se puede reutilizar tal cual en:
  - Etapa A: aplicado a y_true (etiquetas verdaderas) para calibrar el contador.
  - Etapa B: aplicado a la binarización de y_prob (predicciones del modelo).

Concepto
--------
Una "secuencia binaria" es, para un sujeto, la tira de 0/1 de sus ventanas en
orden temporal (win_idx = 0..N-1). Con stride=1s, la ventana i empieza en el
segundo i del registro.

El contador hace tres cosas, en este orden:
  1. Detecta tiradas ("runs") de 1s consecutivos.
  2. FUSIÓN CON TOLERANCIA: dos runs separados por un hueco corto de 0s
     (<= tolerance_windows) se unen en un solo evento. Un parpadeo del modelo
     no parte en dos lo que es un mismo evento respiratorio.
  3. FILTRO POR DURACIÓN: descarta los eventos con menos de min_windows
     ventanas consecutivas (ruido: positivos sueltos de 1-2 ventanas).

Duración de un evento (OJO al solape)
-------------------------------------
Como las ventanas se solapan (W=15s, stride=1s), un evento de varias ventanas
NO dura nº_ventanas × 15s. La duración real, en segundos, es:

    duración = (idx_última - idx_primera) * stride_s + window_s

Con window_s=15 y stride_s=1:  duración = (idx_última - idx_primera) + 15.
Ejemplo: run de ventanas 8,9,10 -> (10-8) + 15 = 17s (no 45s).
Una sola ventana ya son window_s segundos (15s > 10s AASM), por eso el filtro
de duración se hace por nº de ventanas, no por segundos.

IMPORTANTE sobre la tolerancia y el stride
-------------------------------------------
Con stride_s=1s, un hueco de k ventanas de 0 equivale a k segundos. Por eso la
tolerancia se expresa en VENTANAS, que con stride=1 coincide con segundos. Si
algún día se usara stride distinto de 1, la equivalencia ventana<->segundo
cambia y habría que revisar este punto.
"""

from __future__ import annotations

import numpy as np


def _find_runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """
    Detecta tiradas de 1s consecutivos.
    Devuelve lista de (inicio, fin) en índices de ventana, AMBOS inclusive.
    Ej.: [0,1,1,0,1] -> [(1,2), (4,4)]
    """
    b = np.asarray(binary).astype(np.int8)
    if b.size == 0 or b.max() == 0:
        return []
    # Diferencias para localizar flancos de subida (0->1) y bajada (1->0)
    padded = np.concatenate(([0], b, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]        # posiciones donde empieza un run
    ends = np.where(diff == -1)[0] - 1     # posiciones donde acaba (inclusive)
    return list(zip(starts.tolist(), ends.tolist()))


def _merge_with_tolerance(runs: list[tuple[int, int]],
                          tolerance_windows: int) -> list[tuple[int, int]]:
    """
    Fusiona runs separados por un hueco <= tolerance_windows ventanas de 0.
    hueco (en ventanas de 0) entre run actual y siguiente:
        gap = next_start - cur_end - 1
    Si gap <= tolerance_windows -> se fusionan.
    """
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1          # nº de ventanas de 0 entre ambos
        if gap <= tolerance_windows:
            merged[-1] = (prev_start, end)  # extender el evento anterior
        else:
            merged.append((start, end))
    return merged


def find_events(binary: np.ndarray,
                tolerance_windows: int,
                min_windows: int) -> list[tuple[int, int]]:
    """
    Pipeline completo: runs -> fusión por tolerancia -> filtro por nº mínimo
    de ventanas. Devuelve la lista de eventos que sobreviven, como pares
    (idx_primera_ventana, idx_última_ventana), ambos inclusive.
    """
    runs = _find_runs(binary)
    merged = _merge_with_tolerance(runs, tolerance_windows)
    # longitud del evento en ventanas = (fin - inicio + 1)
    kept = [(s, e) for (s, e) in merged if (e - s + 1) >= min_windows]
    return kept


def count_events(binary: np.ndarray,
                 tolerance_windows: int,
                 min_windows: int) -> int:
    """Número de eventos discretos tras fusión y filtro."""
    return len(find_events(binary, tolerance_windows, min_windows))


def event_durations_s(events: list[tuple[int, int]],
                      window_s: float,
                      stride_s: float) -> list[float]:
    """
    Duración real (en segundos) de cada evento, teniendo en cuenta el solape:
        duración = (idx_última - idx_primera) * stride_s + window_s
    """
    return [(e - s) * stride_s + window_s for (s, e) in events]


def summarize_subject(binary: np.ndarray,
                      tolerance_windows: int,
                      min_windows: int,
                      window_s: float,
                      stride_s: float) -> dict:
    """
    Resumen del contador para un sujeto: nº de eventos y estadísticas de
    duración. Útil para inspección/diagnóstico.
    """
    events = find_events(binary, tolerance_windows, min_windows)
    durs = event_durations_s(events, window_s, stride_s)
    return {
        "n_events": len(events),
        "dur_mean_s": float(np.mean(durs)) if durs else 0.0,
        "dur_min_s": float(np.min(durs)) if durs else 0.0,
        "dur_max_s": float(np.max(durs)) if durs else 0.0,
        "events": events,
    }