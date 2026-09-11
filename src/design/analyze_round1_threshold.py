#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/design/analyze_round1_threshold.py
----------------------------------------
Diagnostico del umbral de Ronda 1 (Fase 6 - boosting de negativos dificiles).

Pregunta que responde: ¿el bajo % de purga (10.11:1 -> 9.62:1, vs el ~25%
de Nassi) se debe a que el modelo base discrimina poco en general, o es un
artefacto especifico de exigir un TPR tan alto (0.995)?

Genera:
  1. Histograma de scores del modelo, separado por clase (pos vs neg).
  2. Curva ROC completa + zoom en la zona TPR=[0.90, 1.0] (donde cae tu
     umbral de Ronda 1).
  3. Tabla FPR para varios TPR objetivo (0.995, 0.95, 0.90, 0.70) - si el
     FPR baja mucho al relajar el TPR, el problema es especifico de pedir
     TPR casi perfecto, no que el modelo sea malo en general.
  4. Guarda los scores en .npy para no tener que recalcularlos cada vez
     que quieras mirar otra cosa (otro umbral, otra ronda, etc.).

USO (desde ProyectoPython3.0/, en Picasso3):
  python -m src.design.analyze_round1_threshold
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # sin display en el nodo de computo
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc

from .cnn_model import build_cnn
from .config import IMPUT_DATA_DIR

# ============================================================
# CONFIG
# ============================================================
BASE_EXPERIMENT_TAG = "t050_bs2000_d35_rlrp"
MODEL_PATH = Path(f"data/experiments/train_results/{BASE_EXPERIMENT_TAG}/best_model.h5")

DATA_DIR = Path(IMPUT_DATA_DIR)
X_TRAIN_FULL_PATH = DATA_DIR / "X_train_full.npy"
Y_TRAIN_FULL_PATH = DATA_DIR / "y_train_full.npy"

OUT_DIR = Path("data/experiments/train_results") / BASE_EXPERIMENT_TAG / "boosting_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCORES_CACHE_PATH = OUT_DIR / "scores_train_full.npy"

BATCH_SIZE_PREDICT = 8192
TPR_TARGETS = [0.995, 0.95, 0.90, 0.70]  # comparativa pedida

# ============================================================
# 1. Scores: reusar cache si ya existe, si no calcularlos
# ============================================================
y_train_full = np.load(Y_TRAIN_FULL_PATH).astype(np.int64).ravel()

if SCORES_CACHE_PATH.exists():
    print(f"[INFO] Usando scores cacheados: {SCORES_CACHE_PATH}")
    scores = np.load(SCORES_CACHE_PATH)
    assert len(scores) == len(y_train_full), "Cache de scores no coincide con y_train_full, borrala y relanza."
else:
    print(f"[INFO] No hay cache. Calculando scores con {MODEL_PATH}")
    X_train_full = np.load(X_TRAIN_FULL_PATH, mmap_mode="r")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPUs disponibles: {gpus}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_cnn(input_shape=(2, 500, 1))
    model.load_weights(str(MODEL_PATH), by_name=False)
    print("[OK] Pesos del modelo base cargados.")

    def gen_batches():
        n = X_train_full.shape[0]
        for start in range(0, n, BATCH_SIZE_PREDICT):
            yield np.asarray(X_train_full[start:start + BATCH_SIZE_PREDICT], dtype=np.float32)

    predict_ds = tf.data.Dataset.from_generator(
        gen_batches,
        output_signature=tf.TensorSpec(shape=(None, 2, 500, 1), dtype=tf.float32),
    ).prefetch(tf.data.AUTOTUNE)

    scores_list = []
    n_batches = int(np.ceil(X_train_full.shape[0] / BATCH_SIZE_PREDICT))
    for i, batch in enumerate(predict_ds, start=1):
        scores_list.append(model.predict_on_batch(batch).ravel())
        if i % 50 == 0 or i == n_batches:
            print(f"[INFO] Batch {i}/{n_batches}")

    scores = np.concatenate(scores_list).astype(np.float32)
    np.save(SCORES_CACHE_PATH, scores)
    print(f"[OK] Scores guardados en {SCORES_CACHE_PATH} (reutilizables).")

# ============================================================
# 2. Histograma de scores por clase
# ============================================================
is_pos = (y_train_full == 1)
is_neg = (y_train_full == 0)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 60)
ax.hist(scores[is_neg], bins=bins, alpha=0.6, label=f"Negativos (n={is_neg.sum():,})",
        color="#BA7517", density=True)
ax.hist(scores[is_pos], bins=bins, alpha=0.6, label=f"Positivos (n={is_pos.sum():,})",
        color="#1D9E75", density=True)
ax.axvline(0.0775, color="red", linestyle="--", linewidth=1.2,
           label="Umbral Ronda 1 (TPR=0.995) = 0.0775")
ax.set_xlabel("Score del modelo")
ax.set_ylabel("Densidad")
ax.set_title(f"Distribución de scores por clase — modelo {BASE_EXPERIMENT_TAG}")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
hist_path = OUT_DIR / "scores_histogram_by_class.png"
fig.savefig(hist_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {hist_path}")

# ============================================================
# 3. Curva ROC completa + zoom en TPR=[0.90, 1.0]
# ============================================================
fpr, tpr, thresholds = roc_curve(y_train_full, scores)
roc_auc = sklearn_auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color="#1D9E75", linewidth=1.8, label=f"ROC (AUC={roc_auc:.4f})")
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].set_title("ROC completa (train_full)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3, linewidth=0.5)

mask_zoom = tpr >= 0.85
axes[1].plot(fpr[mask_zoom], tpr[mask_zoom], color="#BA7517", linewidth=1.8)
axes[1].axhline(0.995, color="red", linestyle="--", linewidth=1, label="TPR=0.995 (Ronda 1)")
axes[1].axvline(0.9509, color="red", linestyle=":", linewidth=1, label="FPR=0.9509 obtenido")
axes[1].set_xlabel("FPR")
axes[1].set_ylabel("TPR")
axes[1].set_title("Zoom: TPR ∈ [0.85, 1.0]")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3, linewidth=0.5)

for a in axes:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

fig.tight_layout()
roc_path = OUT_DIR / "roc_curve_zoom.png"
fig.savefig(roc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {roc_path}")

# ============================================================
# 4. Tabla FPR para varios TPR objetivo
# ============================================================
print("\n=== FPR asociado a distintos TPR objetivo ===")
print(f"{'TPR objetivo':<14}{'TPR real':<12}{'FPR':<12}{'Threshold':<12}")
for t in TPR_TARGETS:
    idx = np.searchsorted(tpr, t)
    idx = min(idx, len(thresholds) - 1)
    print(f"{t:<14.3f}{tpr[idx]:<12.4f}{fpr[idx]:<12.4f}{thresholds[idx]:<12.6f}")

print(f"\n[INFO] AUC global (train_full): {roc_auc:.4f}")
print(f"[OK] Resultados en: {OUT_DIR}")