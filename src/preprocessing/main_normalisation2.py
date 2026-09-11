#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main_normalisation2.py
---------------------
Lee archivos segments_*.npz y aplica:

 - Filtro NaNs
 - Filtro TST mínimo
 - Filtro % positivos máximo
 - Tukey para limitar positivos
 - Guarda solo sujetos válidos como normalised_<ID>.npz

IMPORTANTE:
- NO carga todos los sujetos a RAM.
- Hace 2 pasadas: (1) contar positivos, (2) procesar/guardar uno a uno.
- Mantiene win_times alineado usando índices reales tras Tukey.
"""

from pathlib import Path
import numpy as np

from .normalisation import (
    has_nans,
    compute_tst_hours,
    tukey_limit,
    filter_invalid_spo2_segments,
)

from .config import (
    SEGMENTS_RESULTS_DIR,
    NORMALISED_RESULTS_DIR,
    MIN_TST_HOURS,
    MAX_POSITIVE_RATIO,
    COHORT_PREFIX
)


# ================================================================
# Lectura ligera: SOLO metadatos
# ================================================================
def load_pos_count(npz_path: Path) -> int:
    """Lee n_segments_pos sin cargar arrays grandes."""
    with np.load(npz_path, allow_pickle=False) as data:
        # si existe n_segments_pos como escalar guardado
        if "n_segments_pos" in data:
            return int(data["n_segments_pos"])
        # fallback: si no existe, calcular desde labels (costoso)
        labels = data["labels"]
        return int(labels.sum())


def load_subject_arrays(npz_path: Path) -> dict:
    """Carga arrays necesarios del sujeto (esto sí puede pesar)."""
    with np.load(npz_path, allow_pickle=False) as data:
        airflow = data["airflow_windows"]
        spo2 = data["spo2_windows"]
        labels = data["labels"]
        win_times = data["win_times"]
        fs = int(data["fs"]) if "fs" in data else None

        # si tienes estos campos precomputados los usamos; si no, derivamos
        n_total = int(data["n_segments_total"]) if "n_segments_total" in data else int(labels.shape[0])
        n_pos = int(data["n_segments_pos"]) if "n_segments_pos" in data else int(labels.sum())
        pct_pos = float(data["pct_positive"]) if "pct_positive" in data else float(labels.mean())

    return {
        "airflow": airflow,
        "spo2": spo2,
        "labels": labels,
        "win_times": win_times,
        "n_segments_total": n_total,
        "n_segments_pos": n_pos,
        "pct_positive": pct_pos,
        "fs": fs,
    }


# ================================================================
# Índices Tukey (evita devolver copias múltiples)
# ================================================================
def tukey_indices(labels: np.ndarray, max_pos_allowed: int) -> np.ndarray:
    pos_idx = np.flatnonzero(labels == 1)
    if pos_idx.size <= max_pos_allowed:
        return np.arange(labels.size, dtype=np.int64)

    neg_idx = np.flatnonzero(labels == 0)
    keep_pos = np.random.choice(pos_idx, size=max_pos_allowed, replace=False)

    idx = np.empty(neg_idx.size + keep_pos.size, dtype=np.int64)
    idx[:neg_idx.size] = neg_idx
    idx[neg_idx.size:] = keep_pos
    idx.sort()
    return idx


# ================================================================
# Guardado
# ================================================================
def save_normalised(output_path: Path, airflow, spo2, labels, win_times, fs):
    np.savez_compressed(
        output_path,
        airflow_windows=airflow,
        spo2_windows=spo2,
        labels=labels,
        win_times=win_times,
        n_segments_total=int(labels.size),
        n_segments_pos=int(labels.sum()),
        n_segments_neg=int(labels.size - labels.sum()),
        pct_positive=float(labels.mean()) if labels.size else 0.0,
        fs=int(fs) if fs is not None else -1,
    )


# ================================================================
# MAIN
# ================================================================
def main():
    input_dir = Path(SEGMENTS_RESULTS_DIR)
    output_dir = Path(NORMALISED_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob(f"segments_{COHORT_PREFIX}*.npz"))

    print(f"[INFO] Leyendo desde : {input_dir.resolve()}")
    print(f"[INFO] Guardando en  : {output_dir.resolve()}")
    print(f"[INFO] Encontrados {len(npz_files)} sujetos.")

    # -----------------------------------------------------
    # 1) PASADA 1: contar positivos (ligero)
    # -----------------------------------------------------
    pos_counts = []
    valid_files_for_tukey = []

    for npz_path in npz_files:
        try:
            n_pos = load_pos_count(npz_path)
            pos_counts.append(n_pos)
            valid_files_for_tukey.append(npz_path)
        except Exception as e:
            print(f"[ERROR] No se pudo leer metadatos de {npz_path.name}: {e}")

    if not pos_counts:
        print("[ERROR] No hay sujetos válidos para calcular Tukey.")
        return

    pos_counts = np.asarray(pos_counts, dtype=np.int64)

    # -----------------------------------------------------
    # 2) Límite Tukey global
    # -----------------------------------------------------
    tukey_lim = int(tukey_limit(pos_counts))
    print(f"[INFO] Límite Tukey para positivos: {tukey_lim}")

    # -----------------------------------------------------
    # 3) PASADA 2: procesar 1 a 1 (pesado pero estable)
    # -----------------------------------------------------
    n_errors_processing = 0

    for npz_path in valid_files_for_tukey:
        ID = npz_path.stem.replace("segments_", "")

        try:
            subj = load_subject_arrays(npz_path)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar arrays de {npz_path.name}: {e}")
            continue

        try:
            airflow = subj["airflow"]
            spo2 = subj["spo2"]
            labels = subj["labels"]
            win_times = subj["win_times"]

            n_total = subj["n_segments_total"]
            pct_pos = subj["pct_positive"]

            # ---------- A) NaNs ----------
            if np.isnan(airflow).any() or np.isnan(spo2).any():
                print(f"[DESCARTADO] {ID} → contiene NaNs")
                continue

            # ---------- B) Eliminar segmentos con SpO2 no fisiológica ----------
            airflow, spo2, labels, win_times, n_removed_spo2 = filter_invalid_spo2_segments(
                airflow,
                spo2,
                labels,
                win_times,
                min_valid=50.0,
                max_valid=100.0,
            )

            if n_removed_spo2 > 0:
                print(f"[INFO] {ID} → eliminados {n_removed_spo2} segmentos por SpO2 no fisiológica")

            # Recalcular métricas tras el filtrado
            n_total = int(labels.size)
            n_pos = int(labels.sum())
            pct_pos = float(labels.mean()) if labels.size > 0 else 0.0

            # Si no queda nada, descartar
            if n_total == 0:
                print(f"[DESCARTADO] {ID} → sin segmentos tras filtrar SpO2")
                continue

            # ---------- C) Tiempo de sueño mínimo ----------
            tst_hours = compute_tst_hours(n_total)
            if tst_hours < MIN_TST_HOURS:
                print(f"[DESCARTADO] {ID} → TST={tst_hours:.2f} h (< {MIN_TST_HOURS})")
                continue

            # ---------- D) Ratio positivos excesivo ----------
            if pct_pos > MAX_POSITIVE_RATIO:
                print(f"[DESCARTADO] {ID} → ratio positivos={pct_pos:.2f} (> {MAX_POSITIVE_RATIO})")
                continue

            # ---------- E) Tukey: recortar positivos (con índices reales) ----------
            idx = tukey_indices(labels, max_pos_allowed=tukey_lim)

            airflow_new = airflow[idx]
            spo2_new = spo2[idx]
            labels_new = labels[idx]
            win_times_new = win_times[idx]

            # ---------- F) Guardar ----------
            out_path = output_dir / f"normalised_{ID}.npz"
            save_normalised(out_path, airflow_new, spo2_new, labels_new, win_times_new, subj["fs"])

            print(f"[OK] {ID} → guardado con {labels_new.size} segmentos ({int(labels_new.sum())} positivos).")

        except Exception as e:
            n_errors_processing += 1
            print(f"[ERROR] {ID} → fallo durante procesamiento: {e}")
            continue

    print(f"[INFO] Sujetos con error durante procesamiento: {n_errors_processing}")


if __name__ == "__main__":
    main()