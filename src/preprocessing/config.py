#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuración global para el módulo preprocessing."""

# ============================================================
# 1) COHORTE  (definir primero: de aquí derivan las rutas de datos)
# ============================================================
COHORT_PREFIX = "STLK"   # "STLK" o "STNF"
# Cambiar COHORT_PREFIX ajusta AUTOMÁTICAMENTE las rutas de datos (sección 2).
# Lo único que sigue siendo manual es el import de build_sleep_mask (STLK↔STNF)
# en main_mask.py y main_mask_events.py (un import no se puede cambiar desde config).

# ============================================================
# 2) RUTAS
# ============================================================
# Carpeta base de STAGES; la subcarpeta de cada cohorte coincide con COHORT_PREFIX.
# [VERIFICAR] que la carpeta de STNF sigue el mismo patrón ({STAGES_BASE}/STNF/).
STAGES_BASE  = "/mnt/home/users/ac_aux/portega/fscratch/stages/original/STAGES PSGs"
DATA_EDF_DIR = f"{STAGES_BASE}/{COHORT_PREFIX}/"
DATA_CSV_DIR = f"{STAGES_BASE}/{COHORT_PREFIX}/"

BASE_PROJECT = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0"

DTA_RESULTS_DIR           = f"{BASE_PROJECT}/data/module_results/dta_results/"
RESAMPLED_RESULTS_DIR     = f"{BASE_PROJECT}/data/module_results/resampled_data/"
MASKED_RESULTS_DIR        = f"{BASE_PROJECT}/data/module_results/mask_data/"
MASKED_EVENTS_RESULTS_DIR = f"{BASE_PROJECT}/data/module_results/mask_events_data/"
DELAY_RESULTS_DIR         = f"{BASE_PROJECT}/data/module_results/delay_data/"
SEGMENTS_RESULTS_DIR      = f"{BASE_PROJECT}/data/module_results/segmented_data/"
NORMALISED_RESULTS_DIR    = f"{BASE_PROJECT}/data/module_results/normalised_data/"

# ============================================================
# 3) SEÑAL: remuestreo y delay
# ============================================================
FS_TARGET    = 50     # Hz, frecuencia objetivo tras el resampleo
SPO2_DELAY_S = 25.0   # s, retardo fisiológico de SpO₂ respecto a Airflow

# ============================================================
# 4) EVENTOS
# ============================================================
# Máscara de "región de sueño válida": estadios de sueño + eventos respiratorios/arousals.
SLEEP_EVENTS = {
    "Stage1", "Stage2", "Stage3", "N1", "N2", "N3", "REM",
    # Respiratorios / arousals
    "ObstructiveApnea", "CentralApnea", "MixedApnea", "Hypopnea",
    "Desaturation",            # opcional
    "Arousal w/ Respiratory",
    "Arousal",                 # para STLK
    "Flow Limitation",         # opcional
}

# Máscara estrecha de eventos apneicos (positivos para la CNN).
APNEA_EVENTS = {
    "ObstructiveApnea", "CentralApnea", "MixedApnea", "Hypopnea",
}

# ============================================================
# 5) VENTANAS (Fase 7: W=15 s)
# ============================================================
WINDOW_S  = 15    # s por ventana
OVERLAP_S = 14    # s de solape (stride = WINDOW_S - OVERLAP_S = 1 s)
THR_EVENT = 0.5   # fracción mínima de evento para etiquetar la ventana como positiva

# ============================================================
# 6) CONTROL DE CALIDAD (paso "normalisation")
# ============================================================
MIN_TST_HOURS      = 2      # h mínimas de sueño (ventanas útiles) por sujeto
MAX_POSITIVE_RATIO = 0.40   # ratio máximo de segmentos positivos por sujeto