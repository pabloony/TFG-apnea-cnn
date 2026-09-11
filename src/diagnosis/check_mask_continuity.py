#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_mask_continuity.py
--------------------------
Comprueba si los eventos "cortos" detectados en mask_events son eventos
reales, o son fragmentos de un evento más largo partido por un hueco de
pocas muestras (ej. 1 sample a 0, típico de errores de reconstrucción de
máscara: interpolación, redondeo de índices, etc.).

Para cada evento, mide el hueco (en muestras) hasta el evento siguiente
en el mismo sujeto. Si el hueco es muy pequeño (<= GAP_THRESHOLD_SAMPLES)
Y ambos fragmentos son razonablemente cortos, es candidato a fusión.

No modifica ningún dato. No entrena nada. No toca GPU.

Uso:
    python check_mask_continuity.py                # todos los sujetos STNF
    python check_mask_continuity.py STNF00012       # un sujeto concreto
"""

from pathlib import Path
import numpy as np
import sys

BASE_PROJECT = "/mnt/home/users/ac_aux/portega/ProyectoPython3.0"
DELAY_RESULTS_DIR = Path(f"{BASE_PROJECT}/data/module_results/delay_data/")
COHORT_PREFIX = "STNF"

OUTPUT_DIR = Path(f"{BASE_PROJECT}/data/module_results/mask_continuity_check/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Huecos de hasta N muestras se consideran "sospechosos" de fragmentación
# (a fs=50Hz, 1 muestra=0.02s, 2 muestras=0.04s -- fisiológicamente
# imposible que la respiración se reanude y pare de nuevo en ese tiempo)
GAP_THRESHOLD_SAMPLES = 3

# Si tras fusionar el evento sigue siendo "corto" (<10s AASM), es más
# sospechoso aún de ser parte de un evento real más largo
AASM_MIN_EVENT_S = 10.0


def find_event_runs_idx(mask: np.ndarray):
    """
    Devuelve lista de (start_idx, end_idx) de tramos contiguos mask==1.
    end_idx es exclusivo (como slicing de python).
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return []
    diff = np.diff(mask.astype(np.int16))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0] == 1:
        starts = np.r_[0, starts]
    if mask[-1] == 1:
        ends = np.r_[ends, len(mask)]
    return list(zip(starts.tolist(), ends.tolist()))


def analyze_subject(npz_path: Path, gap_thr=GAP_THRESHOLD_SAMPLES):
    d = np.load(npz_path, allow_pickle=True)
    mask_events = d["mask_events"]
    fs = float(d["fs_delay"]) if "fs_delay" in d.files else float(d.get("fs", 50))

    runs = find_event_runs_idx(mask_events)
    n_events = len(runs)

    fragmented_candidates = []  # (idx_evento_1, idx_evento_2, gap_samples, dur1_s, dur2_s, dur_fusion_s)

    for i in range(n_events - 1):
        s1, e1 = runs[i]
        s2, e2 = runs[i + 1]
        gap = s2 - e1  # muestras entre el final de un evento y el inicio del siguiente

        if gap <= gap_thr:
            dur1_s = (e1 - s1) / fs
            dur2_s = (e2 - s2) / fs
            dur_fusion_s = (e2 - s1) / fs  # si se fusionaran en uno solo
            fragmented_candidates.append({
                "subject": npz_path.stem.replace("delay_", ""),
                "gap_samples": gap,
                "start1_s": s1 / fs,
                "dur1_s": dur1_s,
                "dur2_s": dur2_s,
                "dur_fusion_s": dur_fusion_s,
            })

    return n_events, fragmented_candidates


def main(subject_id=None):
    if subject_id:
        files = list(DELAY_RESULTS_DIR.glob(f"delay_*{subject_id}*.npz"))
        if not files:
            sys.exit(f"[ERROR] No se encontró ningún npz que contenga '{subject_id}'")
    else:
        files = sorted(DELAY_RESULTS_DIR.glob(f"delay_{COHORT_PREFIX}*.npz"))

    if not files:
        sys.exit(f"[ERROR] No hay delay_*.npz en {DELAY_RESULTS_DIR.resolve()}")

    print(f"[INFO] Analizando {len(files)} sujeto(s), gap_threshold={GAP_THRESHOLD_SAMPLES} muestras\n")

    total_events = 0
    all_candidates = []
    n_subjects_with_candidates = 0

    for f in files:
        try:
            n_events, candidates = analyze_subject(f)
            total_events += n_events
            if candidates:
                n_subjects_with_candidates += 1
                all_candidates.extend(candidates)
        except Exception as e:
            print(f"  [WARN] Falló {f.name}: {e}")

    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Total eventos analizados       : {total_events}")
    print(f"Sujetos con huecos sospechosos  : {n_subjects_with_candidates} / {len(files)}")
    print(f"Pares evento-evento con gap<={GAP_THRESHOLD_SAMPLES} muestras: {len(all_candidates)} "
          f"({100*len(all_candidates)/max(total_events,1):.3f}% de los eventos)")

    if not all_candidates:
        print("\n[INFO] No se encontraron huecos sospechosos con este umbral. "
              "Prueba a subir GAP_THRESHOLD_SAMPLES si quieres ser menos estricto.")
        return

    # Cuántos de estos, si se fusionan, dejarían de ser eventos "cortos" (<10s)
    # -- es decir, cuántos eventos cortos son *causados* por la fragmentación
    short_before = sum(1 for c in all_candidates if c["dur1_s"] < AASM_MIN_EVENT_S or c["dur2_s"] < AASM_MIN_EVENT_S)
    short_after_fusion = sum(1 for c in all_candidates if c["dur_fusion_s"] < AASM_MIN_EVENT_S)

    print(f"\nDe estos {len(all_candidates)} pares:")
    print(f"  - Al menos uno de los dos fragmentos es < {AASM_MIN_EVENT_S}s: {short_before}")
    print(f"  - Tras fusionar, el evento combinado seguiría siendo < {AASM_MIN_EVENT_S}s: {short_after_fusion}")
    print(f"  - Tras fusionar, el evento combinado pasaría a ser >= {AASM_MIN_EVENT_S}s: "
          f"{short_before - short_after_fusion} "
          f"(estos son los que probablemente deberían ser 1 evento largo, no 2 cortos)")

    # Guardar detalle completo para inspección manual
    out_path = OUTPUT_DIR / f"fragmentation_candidates_{COHORT_PREFIX}.csv"
    import csv
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_candidates[0].keys()))
        writer.writeheader()
        writer.writerows(all_candidates)

    print(f"\n[GUARDADO] Detalle completo -> {out_path}")
    print(f"[INFO] Top 10 casos con gap más pequeño (los más sospechosos):")
    all_candidates.sort(key=lambda c: c["gap_samples"])
    print(f"{'Sujeto':<15} {'gap(muestras)':>13} {'dur1(s)':>9} {'dur2(s)':>9} {'dur_fusión(s)':>14}")
    for c in all_candidates[:10]:
        print(f"{c['subject']:<15} {c['gap_samples']:>13d} {c['dur1_s']:>9.2f} "
              f"{c['dur2_s']:>9.2f} {c['dur_fusion_s']:>14.2f}")


if __name__ == "__main__":
    subj = sys.argv[1] if len(sys.argv) > 1 else None
    main(subject_id=subj)