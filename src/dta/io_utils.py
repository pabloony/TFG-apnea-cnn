"""
io_utils.py
-----------
Lectura y detección de canales desde archivos EDF.

"""

import pyedflib 
from pathlib import Path
import numpy as np
from .config import AIRFLOW_CANDIDATES, SPO2_CANDIDATES

def find_channels(ch_names,candidates):
    """Detecta un canal en una lista de nombres de canales."""
    for ch in ch_names:
        for candidate in candidates:
            if ch.lower() == candidate.lower():
                return ch
    return None

def get_channel_fs(f):
    labels = f.getSignalLabels()
    freqs = f.getSampleFrequencies()
    return dict(zip(labels,freqs))

def read_channels(fpath):
    """Lee los canales de interés desde un archivo EDF."""
    path = Path(fpath)

    if not path.exists():
        return None,{"error":f"File does not exist: {path}"}
    try:
        with pyedflib.EdfReader(str(path)) as f: # Abrir con pyedflib para obtener info de canales
            # no lo uso : n_signals = f.signals_in_file
            chs = f.getSignalLabels() # Lista de nombres de canales

            airflow_ch =find_channels(chs,AIRFLOW_CANDIDATES)
            spo2_ch = find_channels(chs,SPO2_CANDIDATES)

            if airflow_ch is None or spo2_ch is None:
                return None,{"error":f"Required channels not found in {path.name}"}
            
            airflow_idx = chs.index(airflow_ch)
            spo2_idx = chs.index(spo2_ch)

            fs_dict = get_channel_fs(f)

            fs_airflow = float(fs_dict[airflow_ch])
            fs_spo2 = float(fs_dict[spo2_ch])

            airflow_data = f.readSignal(airflow_idx)
            spo2_data = f.readSignal(spo2_idx)

            meas_date = f.getStartdatetime()
            duration_s = f.file_duration
            
    except Exception as e:
        return None,{"error":f"Error reading EDF file {path.name}: {e}"}

    
    info = {
        "airflow_ch": airflow_ch,
        "spo2_ch": spo2_ch,
        "fs_airflow": fs_airflow,
        "fs_spo2": fs_spo2,
        "duration_s": duration_s,
        "start_time": meas_date,
        "n_channels":len(chs)
    }

    signals = {
        "airflow": airflow_data,
        "spo2": spo2_data
    }

    return signals,info

def save_dta_result(edf_path,signals,info,output_dir):
    """Guarda los resultados DTA en un archivo npz."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True) # parents y exist_ok para crear carpetas si no existen

    base_name = edf_path.stem 
    save_path = output_path / f"{base_name}_dta_results.npz"

    np.savez_compressed(
        save_path,
        airflow=signals['airflow'],
        spo2=signals['spo2'],
        fs_airflow=info['fs_airflow'],
        fs_spo2=info['fs_spo2'],
        duration_s=info['duration_s'],
        start_time=str(info['start_time']) #cuidado!, lo guardamos como string
        #airflow_ch=info['airflow_ch'], guarda el nombre del canal, pero no es tan útil
        #spo2_ch=info['spo2_ch']#
    )
    return save_path
    



