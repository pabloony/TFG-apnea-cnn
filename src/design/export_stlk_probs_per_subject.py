#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_stlk_probs_per_subject.py
--------------------------------
VALIDACIÓN CROSS-COHORTE (STLK) — Export de probabilidades por-ventana por sujeto.

Gemelo de export_val_probs_per_subject.py, pero apuntado a la cohorte STLK, que
NO se usó en entrenamiento (Fase 5: STLK baja el AUC por distribution shift de
sensor Nasal_Therm vs NasOr). El objetivo es medir cómo generaliza el estimador
de AHI (calibrado en STNF) a una cohorte con sensor distinto, SIN reentrenar.

IMPORTANTE (coherencia de normalización):
Se usa el MISMO modelo _pclip de STNF y la MISMA función de normalización
(zscore_per_edf_per_channel de main_imput_trainval_v2, que es la versión
percentile-clip). La normalización se aplica en inferencia, NO está horneada en
los joined; por tanto STLK se normaliza igual que STNF y la coherencia
modelo<->normalización se mantiene automáticamente.

Diferencias con el script de STNF:
- Carpeta de joined distinta (Joined_STLK_STNF/joined_data).
- La lista de sujetos NO existe (no hay stlk_ids.txt): se GENERA listando la
  carpeta (joined_STLK*.npz) y cruzando con ahi_labels.csv (solo sujetos con
  ground truth). La lista efectiva se vuelca a stlk_ids_usados.txt.
- Salida a un CSV propio: val_window_probs_STLK.csv (no pisa el de STNF).

Uso
---
    python -m src.design.export_stlk_probs_per_subject
"""

from __future__ import annotations

from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from ..clinical_eval.config import AHI_GT_CSV, FASE9_DIR
from .config import MODEL_BEST_PATH
from ..complementation.main_imput_trainval_v2 import zscore_per_edf_per_channel


# ---------------------------
# Parámetros específicos STLK
# ---------------------------
STLK_JOINED_DIR = Path(
    "/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/complementation_results/joined_data"
)
# Salida: junto al resto de Fase 9, pero con nombre propio de cohorte
OUT_CSV        = Path(FASE9_DIR) / "val_window_probs_STLK.csv"
IDS_USED_TXT   = Path(FASE9_DIR) / "stlk_ids_usados.txt"

PREDICT_BS = 256
STRIDE_S   = 1  # solo informativo


def _discover_stlk_ids(joined_dir: Path, gt_codes: set[str]) -> list[str]:
    """
    Lista joined_STLK*.npz de la carpeta, extrae el ID, y se queda solo con los
    que tienen ground truth en ahi_labels.csv. Devuelve la lista ordenada.
    """
    pat = re.compile(r"joined_(STLK\d+)\.npz$")
    ids = []
    for p in sorted(joined_dir.glob("joined_STLK*.npz")):
        m = pat.search(p.name)
        if not m:
            continue
        sid = m.group(1)
        if sid in gt_codes:
            ids.append(sid)
    return sorted(set(ids))


def _load_gt_codes(csv_path: Path) -> set[str]:
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    return set(gt["s_code"].astype(str))


def _completed_subjects(out_csv: Path) -> set[str]:
    """Sujetos ya escritos y completos (excluye el último, por si quedó a medias)."""
    if not out_csv.exists():
        return set()
    seen_order, seen_set = [], set()
    with out_csv.open("r", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if row and row[0] not in seen_set:
                seen_set.add(row[0]); seen_order.append(row[0])
    if not seen_order:
        return set()
    print(f"[RESUME] {len(seen_order)} sujetos en CSV. Reprocesando el último: {seen_order[-1]}")
    return set(seen_order[:-1])


def _drop_incomplete_last(out_csv: Path, keep: set[str]) -> None:
    tmp = out_csv.with_suffix(".tmp")
    with out_csv.open("r", newline="") as fin, tmp.open("w", newline="") as fout:
        reader = csv.reader(fin); writer = csv.writer(fout)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            if row and row[0] in keep:
                writer.writerow(row)
    tmp.replace(out_csv)
    print(f"[RESUME] CSV limpiado: conservados {len(keep)} sujetos.")


def _load_subject_windows(joined_dir: Path, subj_id: str):
    npz_path = joined_dir / f"joined_{subj_id}.npz"
    if not npz_path.exists():
        print(f"    [WARN] No existe {npz_path.name}; omitido.")
        return None, None
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"    [ERROR] {npz_path.name} corrupto -> {type(e).__name__}: {e}")
        return None, None
    return data["joined_windows"], data["labels"].astype(np.int64)


def main() -> None:
    out_dir = Path(FASE9_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(MODEL_BEST_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    gt_codes = _load_gt_codes(Path(AHI_GT_CSV))
    stlk_ids = _discover_stlk_ids(STLK_JOINED_DIR, gt_codes)
    IDS_USED_TXT.write_text("\n".join(stlk_ids) + "\n")

    done = _completed_subjects(OUT_CSV)
    if done and OUT_CSV.exists():
        _drop_incomplete_last(OUT_CSV, done)
    file_exists = OUT_CSV.exists()
    mode = "a" if file_exists else "w"

    print("=" * 60)
    print("[CROSS-COHORTE STLK] Export probabilidades por-ventana")
    print("=" * 60)
    print(f"[INFO] JOINED dir  : {STLK_JOINED_DIR}")
    print(f"[INFO] Modelo      : {model_path}  (mismo _pclip de STNF)")
    print(f"[INFO] Sujetos STLK: {len(stlk_ids)} (con ground truth)")
    print(f"[INFO] Ya completos: {len(done)} (se saltan)")
    print(f"[INFO] Salida CSV  : {OUT_CSV}")
    print(f"[INFO] IDs usados  : {IDS_USED_TXT}")
    print(f"[INFO] Modo        : {'APPEND (reanudando)' if mode == 'a' else 'NUEVO'}")
    print("-" * 60)

    model = tf.keras.models.load_model(model_path, compile=False)

    n_rows_new = n_subj_ok = 0
    with OUT_CSV.open(mode, newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["subject_id", "win_idx", "y_prob", "y_true"])

        for i, sid in enumerate(stlk_ids, start=1):
            if sid in done:
                print(f"[SKIP] ({i}/{len(stlk_ids)}) {sid} ya completo.")
                continue
            print(f"[INFO] ({i}/{len(stlk_ids)}) {sid}")

            X, y = _load_subject_windows(STLK_JOINED_DIR, sid)
            if X is None:
                continue

            N = len(y)
            print(f"    [INFO] Ventanas: {N} (pos={int((y==1).sum())}, neg={int((y==0).sum())})")

            Xn = zscore_per_edf_per_channel(X, y)
            Xn = Xn[..., np.newaxis].astype(np.float32)
            y_prob = model.predict(Xn, batch_size=PREDICT_BS, verbose=0).reshape(-1)

            for win_idx in range(N):
                writer.writerow([sid, win_idx, f"{float(y_prob[win_idx]):.6f}", int(y[win_idx])])
            fh.flush()

            n_rows_new += N; n_subj_ok += 1
            print(f"    [OK] {N} ventanas escritas.")

    print("-" * 60)
    print(f"[OK] Sujetos nuevos: {n_subj_ok}")
    print(f"[OK] Filas nuevas  : {n_rows_new}")
    print(f"[OK] CSV en        : {OUT_CSV}")
    print("[DONE]")


if __name__ == "__main__":
    main()