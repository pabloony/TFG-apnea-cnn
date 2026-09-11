"""Punto de entrada para el módulo preprocessing."""

"""
main_resample.py

Paso 1 del Preprocessing:
- Lee los archivos DTA (.npz) generados en la fase anterior.
- Reconstruye los diccionarios signals e info.
- Re-muestrea airflow y SpO₂ usando resample_pair().
- Guarda los resultados en data/module_results/preprocessed_data/
  con nombres del tipo: resampled_<paciente>.npz
"""

from pathlib import Path
import numpy as np
from .config import RESAMPLED_RESULTS_DIR,DTA_RESULTS_DIR,FS_TARGET,COHORT_PREFIX
from .resample import resample_pair

def process_file(npz_path : Path, output_dir : Path): #NUEVO, uso de type hints,aclarando tipos de variables y ayudan a sugerir codigo.
    """Procesa un solo archivo DTA: carga → valida → resamplea → guarda."""
    
    print(f"\n[INFO] Procesando: {npz_path.name}")

    # Cargar archivo .npz
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {npz_path.name}: {e}")
        return
    # Validar contenido
    required_keys = {
        "airflow","spo2",
        "fs_airflow","fs_spo2",
        "duration_s","start_time"
    }  
    # Verificar que todas las claves requeridas estén presentes
    for key in required_keys:
        if key not in data:
            print(f"[WARNING] Archivo incompleto ({key} no encontrado). Saltando...")
            return
    
    # Reconstruir diccionarios signals e info
    signals = {
        "airflow": data["airflow"],
        "spo2": data["spo2"]
    }
    info = {
        "fs_airflow": data["fs_airflow"],
        "fs_spo2": data["fs_spo2"],
        "duration_s": data["duration_s"],
        "start_time": data["start_time"]
    }
    # Re-muestrear señales , recuerda que resmaple_pair devuelve dos diccionarios, signals_resampled e info_updated
    signals_resampled,info_updated = resample_pair(signals,info,fs_target=FS_TARGET)

    #"cadena_original".replace("quitar_esto", "poner_esto")
    base = npz_path.stem.replace("_dta_results","")
    out_path = output_dir / f"resampled_{base}.npz"

    # Verificar si el archivo ya existe para evitar sobreescritura, útil si se re-ejecuta el script
    if out_path.exists():
        print(f"[INFO] El archivo {out_path.name} ya existe. (skip)")
        return
    
    try:
        np.savez_compressed( # estructura: np.savez_compressed(<ruta>, <clave>=<valor>, ...)
            out_path,
            airflow=signals_resampled['airflow'],
            spo2=signals_resampled['spo2'],
            fs_airflow=info_updated['fs_airflow'],
            fs_spo2=info_updated['fs_spo2'],
            duration_s=info_updated['duration_s'],
            start_time=str(info_updated["start_time"]),
             n_samples_resampled=info_updated["n_samples_resampled"]
        )
        print(f"[SUCCESS] Guardado: {out_path.name}")
        
    except Exception as e:
        print(f"[ERROR] No se pudo guardar {out_path.name}: {e}")



def main():

    input_dir = Path(DTA_RESULTS_DIR) # Acuérdate de que es un objeto de la clase Path
    output_dir = Path(RESAMPLED_RESULTS_DIR) # Estructura objeto clase Path("carpeta1/carpeta2/archivo.ext")

    print(f"[INFO] Buscando archivos en: {input_dir.resolve()}")

    # Listar todos los archivos .npz en el directorio de entrada
    npz_files = list(input_dir.glob(f"*{COHORT_PREFIX}*_dta_results.npz")) # The glob module finds all the pathnames matching a specified pattern

    if not npz_files:
        print("[WARNING] No se encontraron archivos .npz para procesar.")
        return

    print(f"[INFO] {len(npz_files)} archivos encontrados.")

    # Procesar archivos uno por uno
    for npz_path in npz_files: # Definimos la variable npz_path en cada iteración
        process_file(npz_path, output_dir)

    
if __name__ == "__main__":
    main()
