# src/design/config.py
# =============================================================================
# CONFIGURACIÓN CENTRAL — MÓDULO DESIGN
# =============================================================================
# Dos parámetros controlan todo: THRESHOLD_TAG y BATCH_SIZE.
# Cámbialos aquí y todas las rutas se actualizan solas.
#
# Ejemplos de combinaciones:
#   THRESHOLD_TAG = "t050"  BATCH_SIZE = 500   →  t050_bs500
#   THRESHOLD_TAG = "t050"  BATCH_SIZE = 1000  →  t050_bs1000
#   THRESHOLD_TAG = "t050"  BATCH_SIZE = 2000  →  t050_bs2000
#   THRESHOLD_TAG = "t080"  BATCH_SIZE = 2000  →  t080_bs2000
# =============================================================================

# =============================================================================
# CONFIGURACIÓN CENTRAL — MÓDULO DESIGN
# =============================================================================
THRESHOLD_TAG = "t050w15"
BATCH_SIZE    = 2000
INPUT_SHAPE = (2, 750, 1)

# Dropout escalonado:
# - CONV:  capas conv extraen features de bajo nivel → menos restricción
# - DENSE: capas densas combinan features → más propensas a memorizar
DROPOUT_CONV  = 0.3 # drop1, drop2, drop3  (paper original: 0.6)
DROPOUT_DENSE = 0.5   # drop4, drop5         (paper original: 0.6)

# NUEVO: qué variante de X_train usar.
#   None      -> X_train.npy / y_train.npy           (balanceado 1:1, el de siempre)
#   "round1"  -> X_train_round1.npy / y_train_round1.npy  (Fase 6, post-purga)
TRAIN_DATA_VARIANT = None

# NUEVO: qué normalización usó main_imput_trainval_v2.py al generar
# X_train.npy/X_val.npy. Esto NO cambia ninguna ruta ni lógica de carga,
# solo sirve para que el nombre del experimento distinga sin ambigüedad
# qué normalización se usó (Fase 8: percentile-clip 1-99, Nassi et al.)
NORM_TAG = "_pclip"

# NUEVO: sufijo de tag según la variante, para que el nombre del experimento
# distinga de forma inequívoca qué dataset de entrenamiento se usó.
_VARIANT_SUFFIX = f"_boost_{TRAIN_DATA_VARIANT}" if TRAIN_DATA_VARIANT else ""

EXPERIMENT_TAG = f"{THRESHOLD_TAG}_bs{BATCH_SIZE}_d{int(DROPOUT_CONV*10)}{int(DROPOUT_DENSE*10)}_rlrp{_VARIANT_SUFFIX}{NORM_TAG}"

# =============================================================================
# 2) HIPERPARÁMETROS DE ENTRENAMIENTO
# =============================================================================
EPOCHS = 50
LR     = 1e-4
LOG_EVERY_N_BATCHES = 500

# Dropout escalonado:
# - CONV:  capas conv extraen features de bajo nivel → menos restricción
# - DENSE: capas densas combinan features → más propensas a memorizar
DROPOUT_CONV  = 0.3 # drop1, drop2, drop3  (paper original: 0.6)
DROPOUT_DENSE = 0.5   # drop4, drop5         (paper original: 0.6)

# =============================================================================
# 3) RUTAS DE ENTRADA
# =============================================================================
IMPUT_DATA_DIR = f"data/complementation_results/imput_{THRESHOLD_TAG}/"

_train_suffix = f"_{TRAIN_DATA_VARIANT}" if TRAIN_DATA_VARIANT else ""
X_TRAIN_PATH = IMPUT_DATA_DIR + f"X_train{_train_suffix}.npy"
Y_TRAIN_PATH = IMPUT_DATA_DIR + f"y_train{_train_suffix}.npy"

# X_val / y_val NO cambian — mismo set de validación que todas las fases anteriores,
# para que el AUC val siga siendo comparable
X_VAL_PATH   = IMPUT_DATA_DIR + "X_val.npy"
Y_VAL_PATH   = IMPUT_DATA_DIR + "y_val.npy"

# =============================================================================
# 4) RUTAS DE SALIDA
# =============================================================================
TRAIN_RESULTS_DIR = f"data/experiments/train_results/{EXPERIMENT_TAG}/"

MODEL_BEST_PATH  = TRAIN_RESULTS_DIR + "best_model.h5"
MODEL_FINAL_PATH = TRAIN_RESULTS_DIR + "model_final.h5"
HISTORY_CSV_PATH = TRAIN_RESULTS_DIR + "history.csv"

# Evaluación sobre validación  ← NUEVO
EVAL_VAL_DIR     = TRAIN_RESULTS_DIR + "eval_val/"
VAL_METRICS_PATH = EVAL_VAL_DIR + "val_metrics.txt"
VAL_Y_PROB_PATH  = EVAL_VAL_DIR + "val_y_prob.npy"
VAL_WINDOW_PROBS_PATH = EVAL_VAL_DIR + "val_window_probs.csv"   # ← Fase 9

# Evaluación directa Piorecky (sin fine-tuning)
EVAL_DIRECT_DIR     = TRAIN_RESULTS_DIR + "eval_direct/"
DIRECT_METRICS_PATH = EVAL_DIRECT_DIR + "direct_metrics.txt"
DIRECT_Y_PROB_PATH  = EVAL_DIRECT_DIR + "direct_y_prob.npy"

# =============================================================================
# 5) PESOS PIORECKY
# =============================================================================
PIORECKY_WEIGHTS_PATH = "data/piorecky_weights/model_apnoe_best_final.h5"

# =============================================================================
# 6) CALLBACKS
# =============================================================================
USE_REDUCE_LR    = True
USE_EARLY_STOP   = True

REDUCE_LR_FACTOR   = 0.5
REDUCE_LR_PATIENCE = 4 #job 1062274 estaba en 8 
REDUCE_LR_MIN_LR   = 1e-6

EARLY_STOP_PATIENCE = 15

# =============================================================================
# 7) DIAGNÓSTICO RÁPIDO: python -m src.design.config
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    print("=" * 60)
    print(f"  EXPERIMENT_TAG : {EXPERIMENT_TAG}")
    print(f"  THRESHOLD_TAG  : {THRESHOLD_TAG}")
    print(f"  BATCH_SIZE     : {BATCH_SIZE}")
    print(f"  DROPOUT_CONV   : {DROPOUT_CONV}  |  DROPOUT_DENSE: {DROPOUT_DENSE}")
    print(f"  EPOCHS         : {EPOCHS}  |  LR: {LR}")
    print("=" * 60)
    rutas = {
        "IMPUT_DATA_DIR"    : IMPUT_DATA_DIR,
        "X_TRAIN"           : X_TRAIN_PATH,
        "Y_TRAIN"           : Y_TRAIN_PATH,
        "X_VAL"             : X_VAL_PATH,
        "Y_VAL"             : Y_VAL_PATH,
        "TRAIN_RESULTS_DIR" : TRAIN_RESULTS_DIR,
        "MODEL_BEST"        : MODEL_BEST_PATH,
        "MODEL_FINAL"       : MODEL_FINAL_PATH,
        "HISTORY_CSV"       : HISTORY_CSV_PATH,
        "EVAL_VAL_DIR"      : EVAL_VAL_DIR,
        "VAL_METRICS"       : VAL_METRICS_PATH,
        "VAL_Y_PROB"        : VAL_Y_PROB_PATH,
        "EVAL_DIRECT_DIR"   : EVAL_DIRECT_DIR,
        "PIORECKY_WEIGHTS"  : PIORECKY_WEIGHTS_PATH,
    }
    for nombre, ruta in rutas.items():
        existe = "✓" if Path(ruta).exists() else "✗ no existe"
        print(f"  {nombre:<22} {ruta}  [{existe}]")
    print("=" * 60)