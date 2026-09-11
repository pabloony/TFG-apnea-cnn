# src/dta/config.py
from pathlib import Path

# =========================
# 1) Canales candidatos
# =========================
AIRFLOW_CANDIDATES = ["Nasal_Therm"]
SPO2_CANDIDATES = ["SAO2"]

# =========================
# 2) Rutas base
# =========================

# Raíz del proyecto: .../ProyectoPython3.0
PROJECT_ROOT = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0").resolve()

# fscratch (ruta fija y clara)
FSCRATCH_ROOT = Path("/mnt/home/users/ac_aux/portega/fscratch").resolve()

# Carpeta REAL donde están EDF + CSV (mismo sitio)
DATA_ROOT = (FSCRATCH_ROOT / "stages" / "original" / "STAGES PSGs" / "STLK").resolve()

# Alias para tu código (si en otros sitios usas DATA_EDF_DIR)
# Si prefieres, puedes usar DATA_ROOT directamente.
DATA_EDF_DIR = DATA_ROOT

# =========================
# 3) Salidas (fuera de fscratch)
# =========================
# Aquí SÍ debe ser Path, no string
DTA_RESULTS_DIR = (PROJECT_ROOT / "data" / "module_results" / "dta_results").resolve()
DTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DTA_RESULTS_DIR_INSPECTION = (PROJECT_ROOT / "data" / "module_results" / "inspection_results").resolve()
DTA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 4) Extensiones
# =========================
EDF_EXTENSIONS = {".edf"}
CSV_EXTENSIONS = {".csv"}
