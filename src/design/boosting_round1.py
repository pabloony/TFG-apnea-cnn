#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/design/boosting_round1.py
------------------------------
Fase 6 - Ronda 1: Boosting de negativos dificiles (Nassi et al.)

Mecanismo (fiel al paper):
  1. Se evalua el modelo BASE ya entrenado (config base, solo STNF,
     t=0.50, bs=2000, dropout=0.3/0.5, RLRP=on -> Best Val AUC = 0.7888,
     tag t050_bs2000_d35_rlrp) sobre TODO X_train_full / y_train_full
     (10.11:1 neg:pos, sin balanceo 1:1).
  2. Se calibra un umbral de probabilidad sobre la curva ROC para
     conseguir TPR objetivo (Ronda 1 -> TPR = 0.995).
  3. Se purgan los negativos "faciles" (score < umbral). Se CONSERVAN
     los negativos "duros" (score >= umbral, el modelo los confunde
     con positivos). Los positivos no se tocan.
  4. Sin class_weight - el unico mecanismo de balanceo es esta purga.

IMPORTANTE: el EXPERIMENT_TAG actual de src/design/config.py apunta al
modelo de la Fase 5 (con STLK, _stnf_stlk). Por eso aqui NO se importa
MODEL_BEST_PATH de config.py, se fija explicitamente el tag del modelo
base STNF-only para no depender de lo que cambie config.py mas adelante.

USO (desde ProyectoPython3.0/, en Picasso3):
  python -m src.design.boosting_round1
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

from .cnn_model import build_cnn
from .config import IMPUT_DATA_DIR

# ============================================================
# CONFIG
# ============================================================
TARGET_TPR = 0.985  # Ronda 1. Ronda 2 -> 0.985, Ronda 3 -> 0.975, ...
ROUND_TAG = "round1"

# Modelo BASE (solo STNF, sin STLK) - tag fijo, no depende de config.py
BASE_EXPERIMENT_TAG = "t050_bs2000_d35_rlrp"
MODEL_PATH = Path(f"data/experiments/train_results/{BASE_EXPERIMENT_TAG}/best_model.h5")

DATA_DIR = Path(IMPUT_DATA_DIR)  # data/complementation_results/imput_t050/
X_TRAIN_FULL_PATH = DATA_DIR / "X_train_full.npy"
Y_TRAIN_FULL_PATH = DATA_DIR / "y_train_full.npy"

BATCH_SIZE_PREDICT = 2000  # mismo bs que entrenamiento, por consistencia

# ============================================================
# 1. Configuración de Hardware (Asegurar GPU)

# ============================================================
gpus = tf.config.list_physical_devices("GPU")
print(f"[INFO] GPUs detectadas por TensorFlow: {gpus}")

if gpus:
    try:
        # Habilitar el crecimiento de memoria para evitar que TF reserve toda la VRAM de golpe
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[OK] Configuración de crecimiento de memoria de GPU completada.")
    except RuntimeError as e:
        # El crecimiento de memoria debe configurarse antes de inicializar los dispositivos
        print(f"[PRECAUCIÓN] No se pudo configurar el crecimiento de memoria: {e}")
else:
    print("[X] ¡ATENCIÓN! No se detectó ninguna GPU. El script correrá en CPU de forma lenta.")

# ============================================================
# 1. Cargar datos y modelo base
# ============================================================
print(f"[INFO] Cargando {X_TRAIN_FULL_PATH} / {Y_TRAIN_FULL_PATH}")
X_train_full = np.load(X_TRAIN_FULL_PATH)
y_train_full = np.load(Y_TRAIN_FULL_PATH).astype(np.int64).ravel()

print(f"[INFO] X_train_full: {X_train_full.shape} | y_train_full: {y_train_full.shape}")

gpus = tf.config.list_physical_devices("GPU")
print(f"[INFO] GPUs disponibles: {gpus}")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

print(f"[INFO] Cargando modelo base: {MODEL_PATH}")
model = build_cnn(input_shape=(2, 500, 1))
model.load_weights(str(MODEL_PATH), by_name=False)
print("[OK] Pesos del modelo base cargados.")



# ============================================================
# 2. Scores del modelo sobre TODO el train set (pos + neg)
# ============================================================
scores = model.predict(X_train_full, batch_size=BATCH_SIZE_PREDICT, verbose=1).ravel()

# ============================================================
# 3. Umbral calibrado sobre la curva ROC para TPR objetivo
# ============================================================
fpr, tpr, thresholds = roc_curve(y_train_full, scores)
idx = np.searchsorted(tpr, TARGET_TPR)
idx = min(idx, len(thresholds) - 1)

threshold_tpr = thresholds[idx]
achieved_tpr = tpr[idx]
achieved_fpr = fpr[idx]

print(f"\n[Umbral ROC] TPR objetivo = {TARGET_TPR:.3f}")
print(f"[Umbral ROC] threshold calibrado = {threshold_tpr:.6f}")
print(f"[Umbral ROC] TPR real conseguido  = {achieved_tpr:.4f}")
print(f"[Umbral ROC] FPR asociado         = {achieved_fpr:.4f}")

# ============================================================
# 4. Purga de negativos faciles (score < umbral)
# ============================================================
is_negative = (y_train_full == 0)
is_positive = (y_train_full == 1)

negative_scores = scores[is_negative]
hard_negative_mask = negative_scores >= threshold_tpr  # se CONSERVAN estos

idx_negatives = np.where(is_negative)[0]
idx_negatives_to_keep = idx_negatives[hard_negative_mask]
idx_positives = np.where(is_positive)[0]

idx_final = np.concatenate([idx_negatives_to_keep, idx_positives])
rng = np.random.default_rng(42)
rng.shuffle(idx_final)

X_train_round1 = X_train_full[idx_final]
y_train_round1 = y_train_full[idx_final]

# ============================================================
# 5. Reporte de ratio neg:pos (comparar con tabla de Nassi)
# ============================================================
n_pos = int(is_positive.sum())
n_neg_before = int(is_negative.sum())
n_neg_after = int(hard_negative_mask.sum())

ratio_before = n_neg_before / n_pos
ratio_after = n_neg_after / n_pos

print("\n=== Resultado de la purga (Ronda 1) ===")
print(f"Positivos (sin cambio):      {n_pos}")
print(f"Negativos antes de purgar:   {n_neg_before}")
print(f"Negativos tras purgar:       {n_neg_after}")
print(f"Ratio neg:pos ANTES:         {ratio_before:.2f}:1")
print(f"Ratio neg:pos DESPUES:       {ratio_after:.2f}:1")
print("[Referencia Nassi Ronda 1]:  10.9:1 -> 8.2:1")

# ============================================================
# 6. Guardar dataset purgado (mismo IMPUT_DATA_DIR que el resto del pipeline)
# ============================================================
out_X = DATA_DIR / f"X_train_{ROUND_TAG}.npy"
out_y = DATA_DIR / f"y_train_{ROUND_TAG}.npy"
np.save(out_X, X_train_round1)
np.save(out_y, y_train_round1)

print(f"\n[OK] Guardado: {out_X.name} / {out_y.name} en {DATA_DIR}")
print(f"[OK] Forma final: X={X_train_round1.shape}, y={y_train_round1.shape}")