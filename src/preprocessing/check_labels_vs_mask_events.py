#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

# =========================
# AJUSTES
# =========================
SUBJECT_ID = "STNF00004"

DELAY_DIR    = Path("data/module_results/delay_data/")
SEGMENTS_DIR = Path("data/module_results/segmented_data/")  # donde guardas segments_<ID>.npz

# Si tu segmentación es la típica de Piorecky:
FS = 50
WINDOW_S = 10.0
OVERLAP = 0.90
STRIDE_S = WINDOW_S * (1.0 - OVERLAP)  # 1.0 s si overlap=90%

# =========================
# CARGAS
# =========================
delay_path = DELAY_DIR / f"delay_{SUBJECT_ID}.npz"
seg_path   = SEGMENTS_DIR / f"segments_{SUBJECT_ID}.npz"

if not delay_path.exists():
    raise FileNotFoundError(f"No existe {delay_path}")
if not seg_path.exists():
    raise FileNotFoundError(f"No existe {seg_path}")

d = np.load(delay_path, allow_pickle=True)
mask_events = d["mask_events"]
fs_delay = int(d["fs_delay"]) if "fs_delay" in d.files else int(d["fs"])

if fs_delay != FS:
    print(f"[WARNING] FS esperado {FS} pero fs_delay={fs_delay}. Ajusto FS al del archivo.")
    FS = fs_delay

s = np.load(seg_path, allow_pickle=True)

# Intenta detectar cómo se llaman en tu segments_*.npz
# Cambia aquí si tu npz usa otros nombres:
labels = None
for k in ("labels", "y", "y_seg", "segment_labels"):
    if k in s.files:
        labels = s[k]
        labels_key = k
        break
if labels is None:
    raise KeyError(f"No encuentro labels en {seg_path.name}. Keys: {s.files}")

print("-"*70)
print(f"[INFO] Subject: {SUBJECT_ID}")
print(f"[INFO] delay file   : {delay_path.name} (len mask_events={len(mask_events)})")
print(f"[INFO] segments file: {seg_path.name} (labels key='{labels_key}', n_labels={len(labels)})")
print(f"[INFO] FS={FS}, WINDOW_S={WINDOW_S}, STRIDE_S={STRIDE_S}")
print("-"*70)

# =========================
# RECONSTRUCCIÓN DE INICIOS DE VENTANA
# =========================
win = int(WINDOW_S * FS)
stride = int(STRIDE_S * FS)

# Número de ventanas que saldrían si segmentas "a pelo" toda la señal
n_possible = 1 + (len(mask_events) - win) // stride
print(f"[INFO] ventanas posibles (según win/stride): {n_possible}")

# Si tu segmentador además filtra por mask_sleep o recorta, puede no coincidir.
# En ese caso no podemos reconstruir 1:1 solo con win/stride.
# Pero SÍ podemos comprobar coherencia si en tu segments_*.npz guardaste start_idx.
start_idx = None
for k in ("start_idx", "start_indices", "win_starts", "starts"):
    if k in s.files:
        start_idx = s[k].astype(int)
        start_key = k
        break

if start_idx is None:
    print("[WARNING] No encuentro start indices en segments_*.npz.")
    print("          Haré una comprobación limitada asumiendo segmentación completa a pelo (sin filtros).")
    if len(labels) != n_possible:
        print(f"[FAIL] len(labels)={len(labels)} pero n_possible={n_possible}.")
        print("       Esto sugiere que tu segmentación NO es 'a pelo' (hay filtros/recortes),")
        print("       y entonces NECESITAS guardar start_idx para validar alineación.")
        raise SystemExit(1)
    start_idx = np.arange(n_possible) * stride
    start_key = "(reconstruido)"

print(f"[INFO] usando start indices: {start_key} (n={len(start_idx)})")

# =========================
# COMPARAR LABELS vs mask_events EN CADA VENTANA
# =========================
# Regla A: etiqueta positiva si hay al menos un 1 en la ventana
pred_any = np.zeros(len(start_idx), dtype=np.uint8)
# Regla B: positiva si >=50% de muestras de la ventana son 1
pred_half = np.zeros(len(start_idx), dtype=np.uint8)

for i, st in enumerate(start_idx):
    st = int(st)
    en = st + win
    if en > len(mask_events):
        break
    window = mask_events[st:en]
    pred_any[i] = 1 if window.max() == 1 else 0
    pred_half[i] = 1 if window.mean() >= 0.5 else 0

labels = labels.astype(np.uint8)
n = min(len(labels), len(start_idx))

mismatch_any  = int((labels[:n] != pred_any[:n]).sum())
mismatch_half = int((labels[:n] != pred_half[:n]).sum())

print("-"*70)
print("[RESULTADOS]")
print(f"[INFO] mismatches (regla ANY) : {mismatch_any}/{n} = {100*mismatch_any/n:.2f}%")
print(f"[INFO] mismatches (regla 50%) : {mismatch_half}/{n} = {100*mismatch_half/n:.2f}%")
print(f"[INFO] labels pos%            : {labels[:n].mean()*100:.2f}%")
print(f"[INFO] pred_any pos%          : {pred_any[:n].mean()*100:.2f}%")
print(f"[INFO] pred_half pos%         : {pred_half[:n].mean()*100:.2f}%")
print("-"*70)

# Si quieres ver ejemplos concretos de ventanas donde no coincide:
idx_bad = np.where(labels[:n] != pred_any[:n])[0]
print(f"[DEBUG] primeros 10 idx con mismatch ANY: {idx_bad[:10].tolist()}")
