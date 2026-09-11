#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_mask_events_STNF00004.py
Comprueba que mask_events (en delay_STNF00004.npz) marca correctamente
los eventos del CSV para STNF00004 y grafica un tramo del primer evento encontrado.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# RUTAS (según tu mensaje)
# =========================
DATA_CSV_DIR = Path("data/CSVs/")
DELAY_RESULTS_DIR = Path("data/module_results/delay_data/")

SUBJECT_ID = "STNF00004"
PAD_SECONDS = 10  # padding alrededor del evento para ver contexto

# =========================
# Eventos a marcar (CORREGIDO: faltaba coma tras "Hypopnea")
# =========================
APNEA_EVENTS = {
    "ObstructiveApnea", "CentralApnea", "MixedApnea",
    "Hypopnea",
    "Arousal w/ Respiratory"   # opcional
}

# =========================
# Utilidades
# =========================
def hms_to_seconds(hms: str) -> int:
    h, m, s = hms.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

def read_csv_simple(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", on_bad_lines="skip")
    df = df.iloc[:, :3].copy()
    df.columns = ["Start Time", "Duration", "Event"]
    df["Event"] = df["Event"].astype(str).str.strip()
    return df

def main():
    delay_path = DELAY_RESULTS_DIR / f"delay_{SUBJECT_ID}.npz"
    csv_path = DATA_CSV_DIR / f"{SUBJECT_ID}.csv"

    print("-" * 70)
    print(f"[INFO] Subject: {SUBJECT_ID}")
    print(f"[INFO] delay npz: {delay_path}")
    print(f"[INFO] csv      : {csv_path}")
    print("-" * 70)

    if not delay_path.exists():
        raise FileNotFoundError(f"No existe: {delay_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe: {csv_path}")

    # 1) Cargar NPZ delay
    d = np.load(delay_path, allow_pickle=True)

    airflow = d["airflow"]
    spo2 = d["spo2"]
    mask_events = d["mask_events"]

    fs = int(d["fs_delay"]) if "fs_delay" in d.files else int(d["fs"])
    start_time = str(d["start_time"]) if "start_time" in d.files else None

    print(f"[INFO] fs = {fs}")
    print(f"[INFO] start_time meta = {start_time}")
    print(f"[INFO] len airflow={len(airflow)} len spo2={len(spo2)} len mask_events={len(mask_events)}")
    print(f"[INFO] mask_events.sum() total = {int(mask_events.sum())}")

    # 2) Leer CSV
    df = read_csv_simple(csv_path)

    # 3) Filtrar eventos de interés (match exacto)
    df_evt = df[df["Event"].isin(APNEA_EVENTS)].copy()
    print(f"[INFO] filas CSV totales: {len(df)}")
    print(f"[INFO] filas CSV que coinciden con APNEA_EVENTS: {len(df_evt)}")

    if df_evt.empty:
        print("[WARNING] No hay coincidencias EXACTAS con APNEA_EVENTS.")
        uniq = sorted(df["Event"].unique())
        print("[DEBUG] Eventos únicos en el CSV (primeros 80):")
        for e in uniq[:80]:
            print("  -", repr(e))
        print("\n[DEBUG] Sugerencia: compara exactamente esos strings con APNEA_EVENTS.")
        return

    # 4) Tomar el primer evento y comprobar su rango en mask_events
    row = df_evt.iloc[0]
    evt_name = row["Event"]
    evt_start_hms = row["Start Time"]
    evt_dur = float(row["Duration"])

    print("-" * 70)
    print("[INFO] Primer evento (match exacto):")
    print(f"  Event     : {evt_name}")
    print(f"  Start Time: {evt_start_hms}")
    print(f"  Duration  : {evt_dur}")
    print("-" * 70)

    if start_time is None:
        print("[ERROR] El NPZ no tiene 'start_time'. No puedo convertir a tiempo relativo.")
        return

    edf_hms = start_time.split(" ")[1].split("+")[0]  # "22:15:00"
    edf_start_s = hms_to_seconds(edf_hms)

    evt_s = hms_to_seconds(evt_start_hms)
    if evt_s < edf_start_s:
        evt_s += 24 * 3600

    rel_s = evt_s - edf_start_s
    i0 = int(rel_s * fs)
    i1 = int((rel_s + evt_dur) * fs)

    i0 = max(i0, 0)
    i1 = min(i1, len(mask_events))

    slice_sum = int(mask_events[i0:i1].sum())
    slice_len = int(i1 - i0)

    print("[INFO] Evento en muestras:")
    print(f"  rel_s = {rel_s:.2f}s")
    print(f"  idx   = {i0}:{i1}  (len={slice_len})")
    print(f"  mask_events sum en rango = {slice_sum} / {slice_len}")

    # 5) Gráfica alrededor del evento
    pad = int(PAD_SECONDS * fs)
    a = max(i0 - pad, 0)
    b = min(i1 + pad, len(airflow))
    t = np.arange(a, b) / fs

    plt.figure(figsize=(12, 5))
    plt.title(f"{SUBJECT_ID} – mask_events alrededor del primer evento ({evt_name})")

    plt.plot(t, airflow[a:b], label="airflow", alpha=0.7)
    plt.plot(t, spo2[a:b], label="spo2", alpha=0.7)

    ymin = min(airflow[a:b].min(), spo2[a:b].min())
    ymax = max(airflow[a:b].max(), spo2[a:b].max())
    plt.fill_between(
        t, ymin, ymax,
        where=(mask_events[a:b] == 1),
        alpha=0.25,
        label="mask_events==1"
    )

    plt.xlabel("Tiempo (s desde inicio EDF)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("[OK] Check completado.")


if __name__ == "__main__":
    main()
