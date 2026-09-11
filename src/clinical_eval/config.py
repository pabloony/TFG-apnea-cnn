# src/clinical_eval/config.py
# =============================================================================
# CONFIGURACIÓN — MÓDULO CLINICAL_EVAL (Fase 9: evaluación clínica AHI)
# =============================================================================
# Este config NO redefine el experimento activo. IMPORTA de design/config.py
# el tag y las rutas base (EVAL_VAL_DIR, VAL_WINDOW_PROBS_PATH...), de modo que
# hay UNA sola fuente de verdad sobre qué experimento se evalúa. Aquí solo se
# añade lo específico de la evaluación clínica por paciente.
# =============================================================================

from ..design.config import (
    EVAL_VAL_DIR,            # carpeta de evaluación del experimento activo
    VAL_WINDOW_PROBS_PATH,   # CSV de probabilidades por-ventana por sujeto (STNF)
    EXPERIMENT_TAG,          # solo para trazabilidad/logs
)

# -----------------------------------------------------------------------------
# Ground truth por paciente (TST, AHI, conteos por tipo de evento)
# ÚNICO para TODAS las cohortes: ahi_labels.csv contiene BOGN, STNF, STLK...
# El filtrado por cohorte es automático (se cruzan solo los s_code presentes en
# el CSV de probabilidades de la cohorte activa). NO cambia entre cohortes.
# -----------------------------------------------------------------------------
AHI_GT_CSV = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0/ahi_labels.csv"

# -----------------------------------------------------------------------------
# Carpeta raíz de Fase 9 (cuelga del experimento activo vía EVAL_VAL_DIR)
# -----------------------------------------------------------------------------
FASE9_DIR = EVAL_VAL_DIR + "fase9_ahi/"

# =============================================================================
# COHORTE ACTIVA PARA EVALUACIÓN CLÍNICA  ← interruptor único
# =============================================================================
# Cambia SOLO esta línea para evaluar una cohorte u otra. De aquí derivan:
#   - qué CSV de probabilidades se lee
#   - a qué carpetas de salida se escribe (para no pisar resultados entre cohortes)
#
#   "STNF"  -> cohorte de entrenamiento/validación original
#   "STLK"  -> cohorte cross-sensor (NO vista en training; validación de
#              generalización). Su CSV lo genera export_stlk_probs_per_subject.py
# =============================================================================
CLINICAL_COHORT = "STLK"          # "STNF" o "STLK"

if CLINICAL_COHORT == "STNF":
    CLINICAL_PROBS_CSV = VAL_WINDOW_PROBS_PATH
    _COHORT_SFX = ""              # STNF conserva los nombres originales sin sufijo
elif CLINICAL_COHORT == "STLK":
    CLINICAL_PROBS_CSV = FASE9_DIR + "val_window_probs_STLK.csv"
    _COHORT_SFX = "_STLK"
else:
    raise ValueError(f"CLINICAL_COHORT no reconocida: {CLINICAL_COHORT}")

# Carpetas de salida POR COHORTE (se crean solas en los scripts con mkdir).
# STNF -> stage_a/ , stage_b/   |   STLK -> stage_a_STLK/ , stage_b_STLK/
FASE9_STAGE_A_DIR = FASE9_DIR + f"stage_a{_COHORT_SFX}/"
FASE9_STAGE_B_DIR = FASE9_DIR + f"stage_b{_COHORT_SFX}/"

# Carpeta de CSVs de eventos de origen (para la auditoría de calidad de datos).
# Contiene EDF y CSV mezclados; el audit filtra solo los .csv de la cohorte.
EVENTS_CSV_DIR = (
    "/mnt/home/users/ac_aux/portega/fscratch/stages/original/STAGES PSGs/"
    + CLINICAL_COHORT
)

# -----------------------------------------------------------------------------
# Geometría de ventana (para el contador de eventos)
#   INPUT_SHAPE = (2, 750, 1) -> 750 muestras = 15s a 50 Hz
# -----------------------------------------------------------------------------
WINDOW_S = 15   # ancho de ventana en segundos
STRIDE_S = 1    # stride entre ventanas en segundos

# -----------------------------------------------------------------------------
# Barridos ETAPA A (calibración del contador contra y_true)
# -----------------------------------------------------------------------------
STAGE_A_TOLERANCIAS  = [0, 1, 2, 3, 5]   # segundos de hueco tolerado
STAGE_A_MIN_VENTANAS = [1, 2, 3, 5, 8]   # nº mínimo de ventanas consecutivas

# -----------------------------------------------------------------------------
# Sujetos excluidos de la validación clínica (CALIDAD DE DATOS, no bug del pipeline)
# Detectados por auditoría (audit_event_annotations.py) y diagnóstico de runs.
# (Los STNF solo afectan a la cohorte STNF; para STLK se re-auditará y se
#  añadirán aquí los suyos si los hubiera.)
# -----------------------------------------------------------------------------
EXCLUDED_SUBJECTS = {
    "STNF00489": "scoring respiratorio AUSENTE en CSV de origen (AHI=18 en GT, 0 eventos en CSV -> y_true todo ceros)",
    "STNF00310": "run continuo de ~686s (11 min) en y_true, imposible fisiológicamente; segmento mask_events corrupto (tipo STNF00083)",
}
# Nota: STNF00171 (scoring incompleto pero AHI=1, paciente sano) se MANTIENE;
# excluir sanos sesgaría la muestra hacia graves. Documentado como caso límite.

# Sufijo para los archivos de salida de esta pasada, para NO sobrescribir los
# PNG/CSV de pasadas anteriores (p.ej. la primera Etapa A sin exclusiones).
STAGE_A_OUTPUT_SUFFIX = "_clean"

# -----------------------------------------------------------------------------
# ETAPA B (umbral-para-AHI sobre y_prob, con split calibración/test)
# -----------------------------------------------------------------------------
# Parámetros del contador heredados de Etapa A: min_ventanas fijo = 1.
# La tolerancia se RE-CALIBRA aquí (y_prob tiene huecos que y_true no tenía).
STAGE_B_MIN_VENTANAS = 1

# Barrido 2D: umbral de binarización × tolerancia de huecos
STAGE_B_UMBRALES     = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
STAGE_B_TOLERANCIAS  = [0, 1, 2, 3, 5]

# Sufijo de salida (para no pisar corridas anteriores). Cambia por corrida.
#   ""            -> primera corrida (umbrales 0.30-0.60)
#   "_umbralalto" -> ampliación hacia umbrales altos (0.60-0.95)
STAGE_B_OUTPUT_SUFFIX = "_umbralalto"

# Split calibración/test estratificado por severidad AASM
STAGE_B_SPLIT_SEED   = 42
STAGE_B_CALIB_FRAC   = 0.50   # ~mitad calibración, ~mitad test

# Métrica para elegir el mejor par (umbral, tolerancia): "mae" de AHI
STAGE_B_METRIC = "mae"

# -----------------------------------------------------------------------------
# ETAPA B — MODO DE OPERACIÓN (relevante sobre todo para cross-cohorte STLK)
# -----------------------------------------------------------------------------
#   "recalib" -> split calib/test + barrido umbral×tolerancia en ESTA cohorte
#                (lo que se hizo en STNF). Responde: "¿se puede calibrar aquí?"
#   "fixed"   -> NO calibra: aplica el umbral/tolerancia ya fijados (los de STNF)
#                a TODOS los sujetos y reporta. Responde: "¿generaliza tal cual?"
#
# Para STLK, el experimento cross-cohorte limpio es:
#   1) "fixed"   con el umbral de STNF -> generalización pura
#   2) "recalib" -> ¿basta con reajustar el umbral por cohorte?
# -----------------------------------------------------------------------------
STAGE_B_MODE          = "fixed"   # "recalib" o "fixed"
STAGE_B_FIXED_UMBRAL  = 0.85      # umbral óptimo hallado en STNF (Etapa B)
STAGE_B_FIXED_TOL     = 5         # tolerancia óptima hallada en STNF (segundos)

# -----------------------------------------------------------------------------
# Umbrales de severidad AASM (fijos, clínicos — no dependen del modelo)
#   normal < 5 ; leve 5-15 ; moderado 15-30 ; grave > 30
# -----------------------------------------------------------------------------
AASM_SEVERITY_BINS   = [0, 5, 15, 30, float("inf")]
AASM_SEVERITY_LABELS = ["normal", "leve", "moderado", "grave"]

# =============================================================================
# DIAGNÓSTICO RÁPIDO: python -m src.clinical_eval.config
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    print("=" * 60)
    print(f"  EXPERIMENT_TAG (heredado) : {EXPERIMENT_TAG}")
    print(f"  COHORTE ACTIVA            : {CLINICAL_COHORT}")
    print(f"  ETAPA B MODO              : {STAGE_B_MODE}", end="")
    if STAGE_B_MODE == "fixed":
        print(f"  (umbral={STAGE_B_FIXED_UMBRAL}, tol={STAGE_B_FIXED_TOL}s)")
    else:
        print()
    print("=" * 60)
    rutas = {
        "CLINICAL_PROBS_CSV": CLINICAL_PROBS_CSV,
        "AHI_GT_CSV"        : AHI_GT_CSV,
        "FASE9_STAGE_A_DIR" : FASE9_STAGE_A_DIR,
        "FASE9_STAGE_B_DIR" : FASE9_STAGE_B_DIR,
    }
    for nombre, ruta in rutas.items():
        existe = "✓" if Path(ruta).exists() else "✗ no existe"
        print(f"  {nombre:<20} {ruta}  [{existe}]")
    print(f"  WINDOW_S={WINDOW_S}s  STRIDE_S={STRIDE_S}s")
    print(f"  Barrido tol   : {STAGE_A_TOLERANCIAS}")
    print(f"  Barrido minvt : {STAGE_A_MIN_VENTANAS}")
    print("=" * 60)