#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_joining.py
---------------
Une airflow y SpO2 en bloques conjuntos (2 × window_size) por segmento.

Entrada:
    normalised_<ID>.npz

Salida:
    joined_<ID>.npz
"""

from pathlib import Path
import numpy as np

from .joining import join_signals
from .config import NORMALISED_RESULTS_DIR, JOINED_RESULTS_DIR, COHORTS_TO_USE


def load_normalised(npz_path: Path):
    """Carga un archivo normalised_<ID>.npz."""
    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["airflow_windows", "spo2_windows", "labels", "fs"]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Falta la clave obligatoria '{key}' en {npz_path.name}")

    result = {
        "airflow": data["airflow_windows"],
        "spo2": data["spo2_windows"],
        "labels": data["labels"],
        "fs": int(data["fs"]),
    }

    # win_times es opcional
    if "win_times" in data:
        result["win_times"] = data["win_times"]
    else:
        result["win_times"] = None

    return result


def save_joined(output_path: Path, data_dict: dict):
    """Guarda archivo joined_<ID>.npz."""
    save_dict = {
        "joined_windows": data_dict["joined"],
        "labels": data_dict["labels"],
        "fs": data_dict["fs"],
    }

    if data_dict["win_times"] is not None:
        save_dict["win_times"] = data_dict["win_times"]

    np.savez_compressed(output_path, **save_dict)


def main():
    input_dir = Path(NORMALISED_RESULTS_DIR)
    output_dir = Path(JOINED_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("normalised_*.npz"))
    if COHORTS_TO_USE:
        npz_files = [f for f in npz_files if any(c in f.stem for c in COHORTS_TO_USE)]
        print(f"[INFO] Filtro cohorte {COHORTS_TO_USE}: {len(npz_files)} sujetos incluidos")


    print(f"[INFO] Leyendo desde : {input_dir.resolve()}")
    print(f"[INFO] Guardando en  : {output_dir.resolve()}")
    print(f"[INFO] Encontrados {len(npz_files)} sujetos.")

    for npz_path in npz_files:
        ID = npz_path.stem.replace("normalised_", "")
        print(f"\n[INFO] Procesando joining para: {ID}")

        try:
            subj = load_normalised(npz_path)

            airflow = subj["airflow"]
            spo2 = subj["spo2"]
            labels = subj["labels"]
            win_times = subj["win_times"]

            print(f"     airflow shape: {airflow.shape}")
            print(f"     spo2 shape   : {spo2.shape}")
            print(f"     labels shape : {labels.shape}")

            if airflow.shape != spo2.shape:
                raise ValueError(
                    f"Shapes distintas: airflow={airflow.shape}, spo2={spo2.shape}"
                )

            if airflow.ndim != 2 or spo2.ndim != 2:
                raise ValueError(
                    f"Se esperaban arrays 2D (N, W). "
                    f"Recibido: airflow.ndim={airflow.ndim}, spo2.ndim={spo2.ndim}"
                )

            if airflow.shape[0] != len(labels):
                raise ValueError(
                    f"N segmentos != N labels: "
                    f"segmentos={airflow.shape[0]}, labels={len(labels)}"
                )

            if win_times is not None:
                print(f"     win_times shape: {win_times.shape}")
                if len(win_times) != airflow.shape[0]:
                    raise ValueError(
                        f"N segmentos != N win_times: "
                        f"segmentos={airflow.shape[0]}, win_times={len(win_times)}"
                    )
            else:
                print("     [WARN] No existe 'win_times' en este sujeto.")

            # Unir señales
            joined = join_signals(airflow, spo2)

            # Comprobación final de shape esperada
            if joined.shape[0] != len(labels):
                raise ValueError(
                    f"El joined final no coincide con labels: "
                    f"joined={joined.shape}, labels={len(labels)}"
                )

            if joined.shape[1] != 2:
                raise ValueError(
                    f"La dimensión de canales no es 2: joined.shape={joined.shape}"
                )

            # Guardar
            out_path = output_dir / f"joined_{ID}.npz"

            save_joined(
                out_path,
                {
                    "joined": joined,
                    "labels": labels,
                    "win_times": win_times,
                    "fs": subj["fs"],
                },
            )

            print(f"[OK] {ID} → guardado con forma {joined.shape}")

        except Exception as e:
            print(f"[ERROR] No se pudo procesar {ID}: {e}")


if __name__ == "__main__":
    main()