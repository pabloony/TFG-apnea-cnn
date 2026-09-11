#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_event_durations.py
---------------------------
Analiza la duración real (en segundos) de los eventos de apnea/hipopnea
marcados en mask_events, usando los ficheros delay_<ID>.npz ya generados
por el pipeline de preprocessing (src/preprocessing/main_delay_extra.py).

No entrena nada. No toca GPU. Corre en CPU en segundos/minutos.

Qué hace:
1. Recorre todos los delay_<ID>.npz de la cohorte STNF.
2. Para cada sujeto, detecta tramos continuos de mask_events == 1
   (un "evento" = secuencia ininterrumpida de 1s).
3. Convierte cada tramo a duración en segundos usando fs_delay.
4. Agrega todas las duraciones de todos los sujetos.
5. Imprime estadísticos clave y guarda un histograma.
6. Imprime, para varias ventanas candidatas W, qué % de eventos quedarían
   con frac_event máximo < 0.5 (es decir, quedarían excluidos del positivo
   bajo tu threshold actual thr_event=0.5).

Uso:
    python analyze_event_durations.py
"""

from pathlib import Path
import numpy as np

# ============================================================
# RUTAS — ajusta solo si tu config.py ha cambiado estas rutas
# ============================================================
BASE_PROJECT = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0"
DELAY_RESULTS_DIR = Path(f"{BASE_PROJECT}/data/module_results/delay_data/")
COHORT_PREFIX = "STNF"  # análisis centrado en la cohorte base, no STLK

OUTPUT_DIR = Path(f"{BASE_PROJECT}/data/module_results/event_duration_analysis/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ventanas candidatas a evaluar (segundos)
CANDIDATE_WINDOWS = [10, 15, 20, 30]
# Threshold de etiquetado actual (frac_event >= THR -> positivo)
THR_EVENT = 0.5


def find_event_runs(mask: np.ndarray, fs: float):
    """
    Devuelve una lista de duraciones (en segundos) de tramos continuos
    de mask == 1.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return []

    # IMPORTANTE: castear a tipo con signo antes del diff.
    # Si mask es uint8 (sin signo), una transición 1->0 calcula 0-1
    # como 255 (overflow), no -1, y los eventos nunca se cierran.
    diff = np.diff(mask.astype(np.int16))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Casos borde: si la señal empieza o acaba ya en evento
    if mask[0] == 1:
        starts = np.r_[0, starts]
    if mask[-1] == 1:
        ends = np.r_[ends, len(mask)]

    durations_samples = ends - starts
    durations_s = durations_samples / fs
    return durations_s.tolist()


def main():
    npz_files = sorted(DELAY_RESULTS_DIR.glob(f"delay_{COHORT_PREFIX}*.npz"))
    print(f"[INFO] Encontrados {len(npz_files)} ficheros delay_{COHORT_PREFIX}*.npz en {DELAY_RESULTS_DIR}")

    if not npz_files:
        print("[ERROR] No se han encontrado ficheros. Revisa DELAY_RESULTS_DIR / COHORT_PREFIX arriba.")
        return

    all_durations = []
    n_subjects_ok = 0
    n_subjects_fail = 0

    for npz_path in npz_files:
        try:
            d = np.load(npz_path, allow_pickle=True)
            mask_events = d["mask_events"]
            fs = float(d["fs_delay"]) if "fs_delay" in d.files else float(d.get("fs", 50))

            durations = find_event_runs(mask_events, fs)
            all_durations.extend(durations)
            n_subjects_ok += 1
        except Exception as e:
            print(f"[WARNING] Falló {npz_path.name}: {e}")
            n_subjects_fail += 1

    all_durations = np.array(all_durations)
    n_events = len(all_durations)

    print("\n" + "=" * 60)
    print(f"[RESUMEN] Sujetos procesados OK: {n_subjects_ok} | Fallidos: {n_subjects_fail}")
    print(f"[RESUMEN] Total de eventos detectados: {n_events}")

    if n_events == 0:
        print("[ERROR] No se detectaron eventos. Revisa que mask_events no esté vacío.")
        return

    print("\n--- Estadísticos de duración de evento (segundos) ---")
    print(f"  Mínimo   : {all_durations.min():.2f}")
    print(f"  Máximo   : {all_durations.max():.2f}")
    print(f"  Media    : {all_durations.mean():.2f}")
    print(f"  Mediana  : {np.median(all_durations):.2f}")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"  P{p:<3d}     : {np.percentile(all_durations, p):.2f}")

    print("\n--- Impacto de cada ventana candidata bajo thr_event=%.2f ---" % THR_EVENT)
    print("    (un evento de duración E queda EXCLUIDO del positivo en ventana W")
    print("     si frac_max = min(1, E/W) < thr_event, sin importar el solapamiento)")
    for W in CANDIDATE_WINDOWS:
        frac_max = np.minimum(1.0, all_durations / W)
        pct_excluded = 100.0 * np.mean(frac_max < THR_EVENT)
        print(f"  W={W:>3d}s -> {pct_excluded:5.1f}% de eventos NUNCA alcanzan thr_event={THR_EVENT} (excluidos por diseño)")

    # Guardar histograma simple en texto + npy crudo para que lo grafiques tú o te lo pida luego
    np.save(OUTPUT_DIR / f"event_durations_{COHORT_PREFIX}.npy", all_durations)
    print(f"\n[GUARDADO] Duraciones crudas -> {OUTPUT_DIR / f'event_durations_{COHORT_PREFIX}.npy'}")
    print("[INFO] Pégame la salida de este script (los estadísticos de arriba) y seguimos.")


if __name__ == "__main__":
    main()