#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_event_annotations.py
--------------------------
FASE 9 · AUDITORÍA DE CALIDAD DE DATOS

Motivación
----------
STNF00489 tiene AHI=18 y 84 eventos en el ground truth (ahi_labels.csv), pero su
CSV de anotaciones de origen NO contiene ni un solo evento respiratorio (solo el
hipnograma: Wake/Stage1/2/3/REM). Es decir: el scoring respiratorio falta en el
CSV que alimenta el pipeline, aunque el AHI tabulado exista. Su y_true sale todo
a cero -> el modelo nunca pudo aprender/predecir sus eventos.

Este script comprueba, para TODOS los sujetos de validación, si su CSV de origen
contiene el scoring respiratorio que el ground truth promete. Detecta sujetos con
anotación ausente o incompleta, que deben excluirse de la validación clínica.

NO es un bug del pipeline: es un problema de calidad de la base de datos STAGES.
Se documenta como limitación.

Qué hace
--------
Para cada sujeto de val_ids.txt:
  1. Abre su CSV de eventos de origen.
  2. Cuenta filas de eventos respiratorios (n_csv), por los strings exactos que
     usa STAGES: ObstructiveApnea, CentralApnea, MixedApnea, Hypopnea.
  3. Compara con el conteo del ground truth (n_gt = n_obs+n_cen+n_mix+n_hyp).
  4. Marca como sospechoso si n_csv << n_gt (ratio por debajo de un umbral).

Salida
------
  - CSV stage_a_audit_events.csv (sujeto, n_csv, n_gt, ratio, flag, detalle por tipo).
  - Lista por pantalla de sujetos a excluir.

Uso
---
    python -m src.clinical_eval.audit_event_annotations
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AHI_GT_CSV,
    FASE9_STAGE_A_DIR,
    CLINICAL_PROBS_CSV,
    EVENTS_CSV_DIR,
    CLINICAL_COHORT,
)


# ---------------------------
# Parámetros
# ---------------------------
# Carpeta con los CSV de eventos de origen (según cohorte activa del config).
# Contiene CSV y EDF mezclados; solo se usan los .csv.
EVENTS_DIR = Path(EVENTS_CSV_DIR)

# Strings EXACTOS de eventos respiratorios en los CSV de STAGES.
RESP_EVENTS = ["ObstructiveApnea", "CentralApnea", "MixedApnea", "Hypopnea"]

# Columna del CSV de origen que contiene el tipo de evento.
EVENT_COL = "Event"

# Ratio n_csv/n_gt por debajo del cual se marca el sujeto como sospechoso.
SUSPECT_RATIO = 0.50   # el CSV tiene menos de la mitad de eventos que el ground truth
# Umbral mínimo de eventos en el ground truth para que el ratio sea informativo
# (con muy pocos eventos, el ratio es ruidoso; esos se revisan aparte).
MIN_GT_EVENTS = 5


def _load_cohort_ids(probs_csv: Path) -> list[str]:
    """Lee los subject_id únicos del CSV de probabilidades de la cohorte activa."""
    import pandas as pd
    df = pd.read_csv(probs_csv, usecols=["subject_id"], dtype={"subject_id": str})
    return sorted(df["subject_id"].unique().tolist())


def _load_val_ids(imput_dir: Path) -> list[str]:
    path = imput_dir / "val_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}")
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def _load_gt(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    for c in ["n_obs", "n_cen", "n_mix", "n_hyp"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce").fillna(0).astype(int)
    gt["n_gt"] = gt["n_obs"] + gt["n_cen"] + gt["n_mix"] + gt["n_hyp"]
    gt["ahi"] = pd.to_numeric(gt["ahi"], errors="coerce")
    return gt.set_index("s_code")


def _count_csv_events(csv_path: Path) -> dict:
    """
    Cuenta eventos respiratorios en el CSV de origen de un sujeto.
    Devuelve dict con conteo por tipo y total, o None si el CSV no se puede leer.
    """
    try:
        # Igual que el preprocesado (mask_sleep_STLK.py): on_bad_lines="skip"
        # descarta las notas de texto libre del técnico que llevan comas internas
        # (p.ej. "Pt nrem supine with..., SaO2=95%, ..."). Verificado: esas líneas
        # NO contienen eventos respiratorios, así que el conteo no se ve afectado.
        df = pd.read_csv(csv_path, sep=",", on_bad_lines="skip")
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if EVENT_COL not in df.columns:
        # A veces la columna trae espacios o nombre distinto; intentar detectar
        cand = [c for c in df.columns if c.strip().lower() == EVENT_COL.lower()]
        if not cand:
            return {"error": f"No hay columna '{EVENT_COL}'. Columnas: {list(df.columns)}"}
        col = cand[0]
    else:
        col = EVENT_COL

    counts = {}
    ev = df[col].astype(str).str.strip()
    for name in RESP_EVENTS:
        counts[name] = int((ev == name).sum())
    counts["n_csv"] = int(sum(counts[name] for name in RESP_EVENTS))
    return counts


def main() -> None:
    out_dir = Path(FASE9_STAGE_A_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_ids = _load_cohort_ids(Path(CLINICAL_PROBS_CSV))
    gt = _load_gt(Path(AHI_GT_CSV))

    print("=" * 64)
    print("[FASE 9 · AUDITORÍA] Scoring respiratorio en CSV de origen")
    print("=" * 64)
    print(f"[INFO] Cohorte        : {CLINICAL_COHORT}")
    print(f"[INFO] Carpeta eventos: {EVENTS_DIR}")
    print(f"[INFO] Sujetos        : {len(val_ids)}")
    print(f"[INFO] Umbral sospecha: ratio < {SUSPECT_RATIO} (con n_gt >= {MIN_GT_EVENTS})")
    print("-" * 64)

    rows = []
    for sid in val_ids:
        csv_path = EVENTS_DIR / f"{sid}.csv"
        n_gt = int(gt.loc[sid, "n_gt"]) if sid in gt.index else -1
        ahi = float(gt.loc[sid, "ahi"]) if sid in gt.index else np.nan

        if not csv_path.exists():
            rows.append({"subject_id": sid, "n_csv": -1, "n_gt": n_gt, "ahi": ahi,
                         "ratio": np.nan, "flag": "CSV_NO_ENCONTRADO"})
            continue

        c = _count_csv_events(csv_path)
        if "error" in c:
            rows.append({"subject_id": sid, "n_csv": -1, "n_gt": n_gt, "ahi": ahi,
                         "ratio": np.nan, "flag": f"ERROR_LECTURA ({c['error']})"})
            continue

        n_csv = c["n_csv"]
        ratio = (n_csv / n_gt) if n_gt > 0 else np.nan

        # Clasificación
        if n_csv == 0 and n_gt >= MIN_GT_EVENTS:
            flag = "VACIO_RESPIRATORIO"        # como STNF00489
        elif n_gt >= MIN_GT_EVENTS and ratio < SUSPECT_RATIO:
            flag = "INCOMPLETO"
        else:
            flag = "OK"

        rows.append({
            "subject_id": sid, "n_csv": n_csv, "n_gt": n_gt, "ahi": ahi,
            "ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
            "flag": flag,
            "obs": c["ObstructiveApnea"], "cen": c["CentralApnea"],
            "mix": c["MixedApnea"], "hyp": c["Hypopnea"],
        })

    df = pd.DataFrame(rows)
    csv_out = out_dir / "stage_a_audit_events.csv"
    df.sort_values(["flag", "ratio"]).to_csv(csv_out, index=False)

    # Resumen
    n_ok       = (df["flag"] == "OK").sum()
    vacios     = df[df["flag"] == "VACIO_RESPIRATORIO"]["subject_id"].tolist()
    incompl    = df[df["flag"] == "INCOMPLETO"]["subject_id"].tolist()
    no_csv     = df[df["flag"] == "CSV_NO_ENCONTRADO"]["subject_id"].tolist()
    errores    = df[df["flag"].str.startswith("ERROR")]["subject_id"].tolist()

    print(f"[RESULTADO] OK                 : {n_ok}")
    print(f"[RESULTADO] VACIO_RESPIRATORIO : {len(vacios)} -> {vacios}")
    print(f"[RESULTADO] INCOMPLETO (<50%)  : {len(incompl)} -> {incompl}")
    print(f"[RESULTADO] CSV_NO_ENCONTRADO  : {len(no_csv)} -> {no_csv}")
    print(f"[RESULTADO] ERROR_LECTURA      : {len(errores)} -> {errores}")
    print("-" * 64)

    a_excluir = vacios + incompl
    if a_excluir:
        print("[SUJETOS A EXCLUIR de la validación clínica]:")
        print(f"  {a_excluir}")
        print(f"  Total: {len(a_excluir)} de {len(val_ids)}")
    else:
        print("[OK] Ningún sujeto con scoring respiratorio ausente/incompleto.")
    print("-" * 64)

    # Detalle de los flagged para inspección
    flagged = df[df["flag"] != "OK"]
    if len(flagged):
        print("[DETALLE de sujetos marcados]")
        cols = ["subject_id", "n_csv", "n_gt", "ratio", "ahi", "flag"]
        print(flagged[cols].sort_values("ratio").to_string(index=False))

    print("-" * 64)
    print(f"[OK] Auditoría guardada: {csv_out}")
    print("[DONE]")


if __name__ == "__main__":
    main()