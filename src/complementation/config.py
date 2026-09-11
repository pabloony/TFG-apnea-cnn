# config.py  (src/complementation/config.py)


# =============================================================================
# EXPERIMENTO ACTIVO  
# =============================================================================
THRESHOLD_TAG = "t050w15"   # opciones: "t050", "t080"

# =============================================================================

# Cohortes a incluir en el dataset de entrenamiento (filtra por prefijo de ID de sujeto)
# Ejemplos: ["STNF"]  -> solo STNF
#           ["STNF", "STLK"] -> ambas
COHORTS_TO_USE = ["STLK"]
# =============================================================================
# RUTAS BASE
# Resultados fase preprocessing
NORMALISED_RESULTS_DIR = "data/module_results/normalised_data/"

# Resultados fase complementation
JOINED_RESULTS_DIR = "data/complementation_results/joined_data/"
IMPUT_RESULTS_DIR  = f"data/complementation_results/imput_{THRESHOLD_TAG}/"


