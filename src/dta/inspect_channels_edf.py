#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_channels_edf.py
-----------------------
Audita los canales airflow y SpO2 directamente desde los EDF (pyedflib):
- header EDF (dimension, physical/digital min/max, prefilter, etc.)
- stats de la señal leída con readSignal()
- chequeo de plausibilidad para SpO2 (%)

Usa rutas y candidatos desde src/dta/config.py.

Uso:
  python -m src.dta.inspect_channels_edf --n 5 --seed 42
  python -m src.dta.inspect_channels_edf --file STNF00043.edf
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pyedflib

from .config import DATA_EDF_DIR, EDF_EXTENSIONS, AIRFLOW_CANDIDATES, SPO2_CANDIDATES, DTA_RESULTS_DIR_INSPECTION


def find_channel_index(labels, candidates):
    cand = {c.lower() for c in candidates}
    for i, lab in enumerate(labels):
        if lab.lower() in cand:
            return i, lab
    return None, None


def summarize_signal(x: np.ndarray) -> dict:
    x = np.asarray(x)
    p1, p50, p99 = np.percentile(x, [1, 50, 99])
    return {
        "dtype": str(x.dtype),
        "n": int(x.size),
        "min": float(np.min(x)),
        "p1": float(p1),
        "median": float(p50),
        "p99": float(p99),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "zeros_%": float((x == 0).mean() * 100.0),
    }


def header_subset(h: dict) -> dict:
    keys = [
        "label", "dimension", "sample_rate",
        "physical_min", "physical_max",
        "digital_min", "digital_max",
        "transducer", "prefilter"
    ]
    return {k: h.get(k) for k in keys if k in h}


def plausibility_spo2(stats: dict, hdr: dict) -> str:
    dim = (hdr.get("dimension") or "").strip()
    mn, mx = stats["min"], stats["max"]

    if dim == "%":
        if 0 <= mn and mx <= 120:
            return "OK: dimension='%' y rango plausible (0–120)."
        return "WARN: dimension='%' pero rango NO plausible (outliers/escalado)."

    if 0 <= mn and mx <= 120:
        return "OK-ish: rango parece % aunque dimension no sea '%'."
    if abs(mn) > 200 or abs(mx) > 200:
        return "RED FLAG: rango muy grande para SpO2 (%). Probable canal equivocado o escalado."
    return "WARN: no concluyente; revisar header/unidades."


def list_edfs(root: Path):
    files = []
    for ext in EDF_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def inspect_one(edf_path: Path) -> dict:
    with pyedflib.EdfReader(str(edf_path)) as f:
        labels = f.getSignalLabels()

        a_idx, a_lab = find_channel_index(labels, AIRFLOW_CANDIDATES)
        s_idx, s_lab = find_channel_index(labels, SPO2_CANDIDATES)

        if a_idx is None or s_idx is None:
            return {"file": edf_path.name, "error": "No se encontraron los canales candidatos en este EDF."}

        air = f.readSignal(a_idx)
        spo = f.readSignal(s_idx)

        air_hdr = f.getSignalHeader(a_idx)
        spo_hdr = f.getSignalHeader(s_idx)

        air_stats = summarize_signal(air)
        spo_stats = summarize_signal(spo)

        return {
            "file": edf_path.name,
            "start_datetime": str(f.getStartdatetime()),
            "airflow": {
                "label": a_lab,
                "header": header_subset(air_hdr),
                "stats": air_stats,
            },
            "spo2": {
                "label": s_lab,
                "header": header_subset(spo_hdr),
                "stats": spo_stats,
                "plausibility": plausibility_spo2(spo_stats, spo_hdr),
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="N EDF aleatorios a auditar.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--file", type=str, default=None, help="Nombre de EDF dentro de DATA_EDF_DIR (ej: STNF00043.edf)")
    ap.add_argument("--save_json", action="store_true", help="Guardar resultados en JSON en inspection_results.")
    args = ap.parse_args()

    edfs = list_edfs(DATA_EDF_DIR)
    if not edfs:
        raise SystemExit(f"No encuentro EDFs en {DATA_EDF_DIR}")

    if args.file:
        target = DATA_EDF_DIR / args.file
        if not target.exists():
            raise SystemExit(f"No existe: {target}")
        chosen = [target]
    else:
        rng = np.random.default_rng(args.seed)
        k = min(args.n, len(edfs))
        chosen = list(rng.choice(edfs, size=k, replace=False))

    print(f"[INFO] DATA_EDF_DIR: {DATA_EDF_DIR}")
    print(f"[INFO] EDFs totales: {len(edfs)} | seleccionados: {len(chosen)}")

    results = []
    for i, p in enumerate(chosen, 1):
        print("\n" + "-" * 80)
        print(f"[{i}/{len(chosen)}] {p.name}")
        out = inspect_one(p)
        results.append(out)

        if "error" in out:
            print("[ERROR]", out["error"])
            continue

        a = out["airflow"]
        s = out["spo2"]

        print("[AIRFLOW]")
        print("  label:", a["label"])
        print("  header:", a["header"])
        print("  stats :", a["stats"])

        print("[SPO2]")
        print("  label:", s["label"])
        print("  header:", s["header"])
        print("  stats :", s["stats"])
        print("  plausibility:", s["plausibility"])

    if args.save_json:
        DTA_RESULTS_DIR_INSPECTION.mkdir(parents=True, exist_ok=True)
        out_path = DTA_RESULTS_DIR_INSPECTION / "channel_audit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Guardado JSON → {out_path}")


if __name__ == "__main__":
    main()
