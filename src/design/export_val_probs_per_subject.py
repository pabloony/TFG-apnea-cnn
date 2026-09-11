#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_val_probs_per_subject.py
-------------------------------
FASE 9 — Export de probabilidades por-ventana, POR SUJETO, para el set de validación.

Motivación
----------
El pipeline de entrenamiento (main_imput_trainval_v2.py) construye X_val.npy con:
  - balanceo 1:1  -> tira la mayoría de negativos
  - shuffle       -> destruye el orden temporal
  - concatenación -> se pierde a qué sujeto pertenece cada ventana

Eso es correcto para medir AUC por-ventana, pero INUTIL para reconstruir el AHI
por-paciente (necesitamos TODAS las ventanas, en orden, y sabiendo de quién son).

Este script arranca de los joined_<ID>.npz (materia prima antes del balanceo) y,
para cada sujeto de validación, hace inferencia sobre TODAS sus ventanas en orden
temporal, sin balanceo ni shuffle. Vuelca un CSV único listo para el post-proceso
de Fase 9 (umbral -> fusión run-length -> conteo de eventos -> AHI).

Normalización
-------------
Reutiliza EXACTAMENTE la misma función de normalización que el entrenamiento
(zscore_per_edf_per_channel, importada de main_imput_trainval_v2) para que las
probabilidades sean coherentes con el modelo entrenado.

Reanudación (RESUME)
--------------------
Si el CSV ya existe (p.ej. un job anterior murió por TIME LIMIT), el script:
  - detecta qué sujetos ya están escritos,
  - descarta las filas del ULTIMO sujeto escrito (pudo quedar a medias) y lo
    reprocesa por seguridad,
  - abre el CSV en modo APPEND y continúa con los sujetos que faltan.
Así se puede relanzar el mismo comando tantas veces como haga falta sin duplicar
ni corromper datos.

Salida
------
CSV en VAL_WINDOW_PROBS_PATH con columnas:
    subject_id : ID del sujeto (p.ej. STNF00042)
    win_idx    : índice temporal de la ventana (0..N-1). Con stride=1s, la ventana
                 i empieza en el segundo i del registro enmascarado por sueño.
    y_prob     : probabilidad de apnea/hipopnea que da el modelo para esa ventana
    y_true     : etiqueta verdadera de la ventana (0/1), útil para VALIDAR el
                 contador de eventos aplicándolo a las etiquetas reales.

Uso
---
    python -m src.design.export_val_probs_per_subject
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import tensorflow as tf

# Rutas y constantes: fuente de verdad = los config del proyecto
from ..complementation.config import JOINED_RESULTS_DIR, IMPUT_RESULTS_DIR
from .config import MODEL_BEST_PATH, EVAL_VAL_DIR, VAL_WINDOW_PROBS_PATH

# Normalización IDENTICA a la de entrenamiento (no se reimplementa)
from ..complementation.main_imput_trainval_v2 import zscore_per_edf_per_channel


# ---------------------------
# Parámetros
# ---------------------------
STRIDE_S     = 1          # stride de las ventanas deslizantes (segundos) — solo informativo
PREDICT_BS   = 256        # batch size de inferencia (igual que eval_val.py)


def _load_val_ids(imput_dir: Path) -> list[str]:
    """Lee val_ids.txt (los sujetos que caen en validación)."""
    path = imput_dir / "val_ids.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra {path}. Corre antes main_imput_trainval_v2 "
            f"para generar el split y val_ids.txt."
        )
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _completed_subjects(out_csv: Path) -> set[str]:
    """
    Devuelve el conjunto de sujetos YA escritos y COMPLETOS en el CSV.
    Excluye el ultimo sujeto que aparece, porque pudo quedar a medias si el job
    murió mientras lo escribía: ese se reprocesará.
    """
    if not out_csv.exists():
        return set()
    seen_order = []
    seen_set = set()
    with out_csv.open("r", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # saltar cabecera
        for row in reader:
            if not row:
                continue
            sid = row[0]
            if sid not in seen_set:
                seen_set.add(sid)
                seen_order.append(sid)
    if not seen_order:
        return set()
    last = seen_order[-1]
    print(f"[RESUME] {len(seen_order)} sujetos ya en el CSV.")
    print(f"[RESUME] Se reprocesa el ultimo por seguridad (pudo quedar a medias): {last}")
    return set(seen_order[:-1])


def _drop_incomplete_last(out_csv: Path, keep: set[str]) -> None:
    """
    Reescribe el CSV conservando SOLO las filas de los sujetos completos (keep).
    Elimina así las filas del ultimo sujeto (potencialmente incompleto) para que
    al reprocesarlo no queden duplicadas.
    """
    tmp = out_csv.with_suffix(".tmp")
    with out_csv.open("r", newline="") as fin, tmp.open("w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            if row and row[0] in keep:
                writer.writerow(row)
    tmp.replace(out_csv)
    print(f"[RESUME] CSV limpiado: conservados {len(keep)} sujetos completos.")


def _load_subject_windows(joined_dir: Path, subj_id: str):
    """
    Carga joined_<ID>.npz y devuelve (X, y) del sujeto SIN balancear ni barajar:
        X : (N, 2, W)  float
        y : (N,)       int
    Devuelve (None, None) si el archivo no existe o está corrupto.
    """
    npz_path = joined_dir / f"joined_{subj_id}.npz"
    if not npz_path.exists():
        print(f"    [WARN] No existe {npz_path.name}; sujeto omitido.")
        return None, None
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"    [ERROR] {npz_path.name} corrupto -> {type(e).__name__}: {e}")
        return None, None

    X = data["joined_windows"]           # (N, 2, W)
    y = data["labels"].astype(np.int64)  # (N,)
    return X, y


def main() -> None:
    joined_dir = Path(JOINED_RESULTS_DIR)
    imput_dir  = Path(IMPUT_RESULTS_DIR)
    out_dir    = Path(EVAL_VAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv    = Path(VAL_WINDOW_PROBS_PATH)

    model_path = Path(MODEL_BEST_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    val_ids = _load_val_ids(imput_dir)

    # --- REANUDACIÓN ---
    done = _completed_subjects(out_csv)
    if done and out_csv.exists():
        _drop_incomplete_last(out_csv, done)

    file_exists = out_csv.exists()
    mode = "a" if file_exists else "w"

    print("=" * 60)
    print("[FASE 9] EXPORT PROBABILIDADES POR-VENTANA (POR SUJETO)")
    print("=" * 60)
    print(f"[INFO] JOINED dir  : {joined_dir.resolve()}")
    print(f"[INFO] Modelo      : {model_path}")
    print(f"[INFO] Sujetos VAL : {len(val_ids)} -> {val_ids[:5]}{' ...' if len(val_ids) > 5 else ''}")
    print(f"[INFO] Ya completos: {len(done)} (se saltan)")
    print(f"[INFO] Salida CSV  : {out_csv.resolve()}")
    print(f"[INFO] Modo        : {'APPEND (reanudando)' if mode == 'a' else 'NUEVO'}")
    print(f"[INFO] stride      : {STRIDE_S}s (win_idx i -> inicio en segundo i)")
    print("-" * 60)

    # Cargar modelo una sola vez
    model = tf.keras.models.load_model(model_path, compile=False)

    n_rows_new = 0
    n_subj_ok  = 0

    with out_csv.open(mode, newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["subject_id", "win_idx", "y_prob", "y_true"])

        for i, subj_id in enumerate(val_ids, start=1):
            if subj_id in done:
                print(f"[SKIP] ({i}/{len(val_ids)}) {subj_id} ya completo.")
                continue

            print(f"[INFO] ({i}/{len(val_ids)}) {subj_id}")

            X, y = _load_subject_windows(joined_dir, subj_id)
            if X is None:
                continue

            N = len(y)
            n_pos = int(np.sum(y == 1))
            n_neg = int(np.sum(y == 0))
            print(f"    [INFO] Ventanas: {N} (pos={n_pos}, neg={n_neg})")

            # Normalización IDENTICA a entrenamiento. NO se balancea, NO se baraja.
            Xn = zscore_per_edf_per_channel(X, y)          # (N, 2, W) float32
            Xn = Xn[..., np.newaxis].astype(np.float32)    # (N, 2, W, 1)

            # Inferencia en orden temporal natural (win_idx = 0..N-1)
            y_prob = model.predict(Xn, batch_size=PREDICT_BS, verbose=0).reshape(-1)

            for win_idx in range(N):
                writer.writerow([
                    subj_id,
                    win_idx,
                    f"{float(y_prob[win_idx]):.6f}",
                    int(y[win_idx]),
                ])

            # Volcado a disco tras cada sujeto: si el job muere, lo escrito persiste
            fh.flush()

            n_rows_new += N
            n_subj_ok  += 1
            print(f"    [OK] {N} ventanas escritas.")

    print("-" * 60)
    print(f"[OK] Sujetos nuevos procesados: {n_subj_ok}")
    print(f"[OK] Filas nuevas escritas    : {n_rows_new}")
    print(f"[OK] CSV en                   : {out_csv.resolve()}")
    print("[DONE]")


if __name__ == "__main__":
    main()