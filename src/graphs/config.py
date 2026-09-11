# src/graphs/config.py
# =============================================================================
# CONFIGURACIÓN — MÓDULO GRAPHS
# =============================================================================

# --- Directorios de entrada ---
NORMALISED_DATA_DIR = "data/module_results/normalised_data/"
SEGMENTS_DATA_DIR   = "data/module_results/segmented_data/"
TRAIN_RESULTS_DIR   = "data/experiments/train_results/"

# --- Directorio de salida ---
GRAPHS_OUTPUT_DIR   = "src/graphs/results/"
DTA_RESULTS_DIR = "data/module_results/dta_results/"

# --- Parámetros de normalización (deben coincidir con preprocessing/config.py) ---
MAX_POSITIVE_RATIO  = 0.40
MIN_TST_HOURS       = 2.0

# --- Calidad de exportación ---
DPI = 150  # 150 suficiente para TFG; sube a 300 para publicación

# =============================================================================
# BASELINE — usado como referencia en gráficas comparativas
# =============================================================================
BASELINE_EXPERIMENT = "t050_bs2000"

# =============================================================================
# FASE 1 — experimentos de threshold
# =============================================================================
PHASE1_EXPERIMENTS = [
    "t020_bs2000",
    "t050_bs2000",
    "t080_bs2000",
]

# =============================================================================
# FASE 2 — experimentos de batch size (rellena cuando tengas resultados)
# =============================================================================
PHASE2_EXPERIMENTS = [
    "t050_bs500",
    "t050_bs1000",
    "t050_bs2000",
]

# =============================================================================
# FASE 3 — experimentos de dropout (rellena cuando tengas resultados)
# =============================================================================
PHASE3_EXPERIMENTS = [
    "t050_bs2000",   # baseline ya hecho (con drop conv 0.3 y drop dense 0.5)
    "t050_bs2000_d56",
    "t050_bs2000_d66",
]


# =============================================================================
# FASE 4 — con/sin RLRP (rellena cuando tengas resultados)
# =============================================================================
PHASE4_EXPERIMENTS = [
    "t050_bs2000",      # mejor sin RLRP
    "t050_bs2000_d35_rlrp", # con RLRP
]

# =============================================================================
# FASE 5 — efecto de añadir STLK al training
# =============================================================================
PHASE5_EXPERIMENTS = [
    "t050_bs2000_d35_rlrp",
    "t050_bs2000_d35_rlrp_stnf_stlk",
]

# =============================================================================
# Colores y etiquetas por experimento (consistentes en todas las gráficas)
# =============================================================================
EXPERIMENT_COLORS = {
    "t020_bs2000":      "#378ADD",
    "t050_bs2000":      "#1D9E75",
    "t080_bs2000":      "#D85A30",
    "t050_bs500":       "#7F77DD",
    "t050_bs1000":      "#BA7517",
    "t050_bs2000_d56":  "#7F77DD",
    "t050_bs2000_d66":  "#E24B4A",
    "t050_bs2000_d35_rlrp": "#BA7517",
    "t050_bs2000_d35_rlrp_stnf_stlk": "#2CA02C",  
}

EXPERIMENT_LABELS = {
    "t020_bs2000":      "t=0.20",
    "t050_bs2000":      "d=0.3/0.5",# "t=0.50 (baseline)" Cambiar según prueba 
    "t080_bs2000":      "t=0.80",
    "t050_bs500":       "bs=500",
    "t050_bs1000":      "bs=1000",
    "t050_bs2000_d35_rlrp": "t=0.50 + RLRP",
    "t050_bs2000_d56":  "d=0.5/0.6",
    "t050_bs2000_d66":  "d=0.6/0.6",
    "t050_bs2000_d35_rlrp_stnf_stlk": "t=0.50 + RLRP + STNF+STLK",
}

# =============================================================================
# GRÁFICAS — activa o desactiva individualmente
# =============================================================================

# -- Comparativas de curvas de entrenamiento --
PLOT_VAL_AUC_CURVES     = True   # val AUC superpuesto por experimento
PLOT_VAL_LOSS_CURVES    = True   # val loss superpuesto por experimento
PLOT_TRAIN_VAL_AUC      = True   # train vs val AUC por experimento (subplots)
PLOT_TRAIN_VAL_LOSS     = True   # train vs val loss por experimento (subplots)

# -- Métricas finales --
PLOT_BAR_BEST_AUC       = True   # barras de best val AUC por experimento
PLOT_ROC_CURVES         = True   # curvas ROC superpuestas
PLOT_PR_CURVES          = True   # curvas Precision-Recall superpuestas

# -- Resumen global --
PLOT_HEATMAP_SUMMARY    = True   # heatmap hiperparámetros vs métricas