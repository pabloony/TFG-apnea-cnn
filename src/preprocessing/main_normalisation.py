#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_normalisation.py
---------------------
Lee archivos segments_*.npz y aplica:

 - Filtro NaNs
 - Filtro TST mínimo
 - Filtro % positivos máximo
 - Tukey para limitar positivos
 - Guarda solo sujetos válidos como normalised_<ID>.npz
"""

from pathlib import Path
import numpy as np

from .normalisation import (
    has_nans,
    compute_tst_hours,
    compute_positive_ratio,
    tukey_limit,
    apply_tukey_to_subject,
)

from .config import (
    SEGMENTS_RESULTS_DIR,
    NORMALISED_RESULTS_DIR,
    MIN_TST_HOURS,
    MAX_POSITIVE_RATIO,
)


# ================================================================
# Cargar un archivo segments_<ID>.npz
# ================================================================
def load_segments(npz_path: Path):
    """Carga archivo segments_<ID>.npz como dict con claves coherentes."""

    data = np.load(npz_path, allow_pickle=True)

    return {
        "airflow": data["airflow_windows"],
        "spo2": data["spo2_windows"],
        "labels": data["labels"],
        "win_times": data["win_times"],
        "n_segments_total": int(data["n_segments_total"]),
        "n_segments_pos": int(data["n_segments_pos"]),
        "n_segments_neg": int(data["n_segments_neg"]),
        "pct_positive": float(data["pct_positive"]),
        "fs": int(data["fs"]),
    }


# ================================================================
# Guardar sujeto normalizado
# ================================================================
def save_normalised(output_path: Path, data_dict: dict):
    np.savez_compressed(
        output_path,
        airflow_windows=data_dict["airflow"],
        spo2_windows=data_dict["spo2"],
        labels=data_dict["labels"],
        win_times=data_dict["win_times"],
        n_segments_total=len(data_dict["labels"]),
        n_segments_pos=int(data_dict["labels"].sum()),
        n_segments_neg=int(len(data_dict["labels"]) - data_dict["labels"].sum()),
        pct_positive=float(data_dict["labels"].mean()),
        fs=data_dict["fs"],
    )


# ================================================================
# MAIN
# ================================================================
def main():
    input_dir = Path(SEGMENTS_RESULTS_DIR)
    output_dir = Path(NORMALISED_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("segments_*.npz"))

    print(f"[INFO] Leyendo desde : {input_dir.resolve()}")
    print(f"[INFO] Guardando en  : {output_dir.resolve()}")
    print(f"[INFO] Encontrados {len(npz_files)} sujetos.")

    # -----------------------------------------------------
    # 1) Recopilar número de positivos de TODOS los sujetos
    # -----------------------------------------------------
    pos_counts = []
    all_subjects = {}

    for npz_path in npz_files:
        try:
            subject = load_segments(npz_path)
            all_subjects[npz_path] = subject
            pos_counts.append(subject["n_segments_pos"])
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {npz_path.name}: {e}")

    pos_counts = np.array(pos_counts)

    # -----------------------------------------------------
    # 2) Calcular límite Tukey
    # -----------------------------------------------------
    tukey_lim = int(tukey_limit(pos_counts))
    print(f"[INFO] Límite Tukey para positivos: {tukey_lim}")

    # -----------------------------------------------------
    # 3) Procesar cada sujeto
    # -----------------------------------------------------
    for npz_path, subj in all_subjects.items():
        ID = npz_path.stem.replace("segments_", "")

        airflow = subj["airflow"]
        spo2 = subj["spo2"]
        labels = subj["labels"]
        win_times = subj["win_times"]

        n_total = subj["n_segments_total"]
        n_pos = subj["n_segments_pos"]
        pct_pos = subj["pct_positive"]

        # ---------- A) NaNs ----------
        if has_nans(subj):
            print(f"[DESCARTADO] {ID} → contiene NaNs")
            continue

        # ---------- B) Tiempo de sueño mínimo ----------
        tst_hours = compute_tst_hours(n_total)
        if tst_hours < MIN_TST_HOURS:
            print(f"[DESCARTADO] {ID} → TST={tst_hours:.2f} h (< {MIN_TST_HOURS})")
            continue

        # ---------- C) Ratio positivos excesivo ----------
        if pct_pos > MAX_POSITIVE_RATIO:
            print(f"[DESCARTADO] {ID} → ratio positivos={pct_pos:.2f} (> {MAX_POSITIVE_RATIO})")
            continue

        # ---------- D) Tukey: recortar SOLO positivos ----------
        airflow_new, spo2_new, labels_new = apply_tukey_to_subject(
            airflow, spo2, labels,
            max_pos_allowed=tukey_lim
        )

        # ---------- E) Guardar sujeto válido ----------
        out_path = output_dir / f"normalised_{ID}.npz"

        save_normalised(
            out_path,
            {
                "airflow": airflow_new,
                "spo2": spo2_new,
                "labels": labels_new,
                "win_times": win_times[: len(labels_new)],
                "fs": subj["fs"],
            },
        )

        print(
            f"[OK] {ID} → guardado con {len(labels_new)} segmentos "
            f"({labels_new.sum()} positivos)."
        )


# Ejecutar
if __name__ == "__main__":
    main()
