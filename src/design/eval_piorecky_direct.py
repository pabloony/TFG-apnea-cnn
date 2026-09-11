#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_piorecky_direct.py
-----------------------
Experimento 2a — Evaluación directa.

Carga los pesos de Piorecky (model_apnoe_best_final.h5) sin ningún
fine-tuning y los evalúa directamente sobre X_val / y_val.

Pregunta que responde:
  ¿Cuánto sabe ya el modelo de Piorecky sobre tus datos STNF
   sin haber visto ni una sola ventana tuya?

Salida (en out_dir):
  - direct_metrics.txt   → ROC-AUC, PR-AUC, F1, threshold óptimo
  - direct_y_prob.npy    → probabilidades predichas sobre X_val
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
)

from .cnn_model import build_cnn

# ── Rutas ─────────────────────────────────────────────────────────────
from .config import IMPUT_DATA_DIR, PIORECKY_WEIGHTS_PATH, EVAL_DIRECT_DIR, DIRECT_METRICS_PATH, DIRECT_Y_PROB_PATH
DATA_DIR     = Path(IMPUT_DATA_DIR)
WEIGHTS_PATH = Path(PIORECKY_WEIGHTS_PATH)
OUT_DIR      = Path(EVAL_DIRECT_DIR)
# ── Verificar GPU ──────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
print(f"[INFO] GPUs disponibles: {gpus}")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Cargar X_val / y_val ──────────────────────────────────────────────
print("[INFO] Cargando X_val y y_val...")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy").astype(np.int32)
print(f"[INFO] X_val: {X_val.shape} | y_val: {y_val.shape}")
print(f"[INFO] y_val counts: 0={np.sum(y_val==0)} 1={np.sum(y_val==1)}")

# ── Construir modelo y cargar pesos ───────────────────────────────────
print("[INFO] Construyendo modelo con arquitectura Piorecky...")
model = build_cnn(input_shape=(2, 500, 1))

print(f"[INFO] Cargando pesos desde: {WEIGHTS_PATH}")
model.load_weights(str(WEIGHTS_PATH), by_name=False)
print("[OK] Pesos cargados correctamente.")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.summary()

# ── Predicción ────────────────────────────────────────────────────────
print("\n[INFO] Prediciendo sobre X_val...")
y_prob = model.predict(X_val, batch_size=2000, verbose=1).reshape(-1).astype(np.float64)

# ── Métricas threshold-free ───────────────────────────────────────────
roc_auc = roc_auc_score(y_val, y_prob)
pr_auc  = average_precision_score(y_val, y_prob)

print("\n─────────────────────────────────────────")
print("  MÉTRICAS THRESHOLD-FREE")
print("─────────────────────────────────────────")
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  PR-AUC  : {pr_auc:.4f}")

# ── Threshold óptimo (Youden) ─────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
youden      = tpr - fpr
best_idx    = np.argmax(youden)
best_thr    = float(thresholds[best_idx])
best_youden = float(youden[best_idx])
print(f"\n  Threshold óptimo (Youden): {best_thr:.4f}  (TPR-FPR={best_youden:.4f})")

# ── Métricas a threshold=0.5 y threshold óptimo ───────────────────────
def report_at_threshold(thr: float, name: str):
    y_pred = (y_prob >= thr).astype(np.int32)
    acc  = accuracy_score(y_val, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp + 1e-12)

    print(f"\n─────────────────────────────────────────")
    print(f"  MÉTRICAS @ {name} (thr={thr:.4f})")
    print(f"─────────────────────────────────────────")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall/Sens : {rec:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  F1          : {f1:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    print(classification_report(y_val, y_pred, digits=4, zero_division=0))

report_at_threshold(0.5,     "default (0.5)")
report_at_threshold(best_thr, "youden óptimo")

# ── Guardar resultados ────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUT_DIR / "direct_y_prob.npy", y_prob.astype(np.float32))

(OUT_DIR / "direct_metrics.txt").write_text(
    f"ROC-AUC={roc_auc:.6f}\n"
    f"PR-AUC={pr_auc:.6f}\n"
    f"thr_youden={best_thr:.6f}\n"
    f"youden_score={best_youden:.6f}\n"
)

print(f"\n[OK] Resultados guardados en: {OUT_DIR.resolve()}")
print("[DONE]")


if __name__ == "__main__":
    pass