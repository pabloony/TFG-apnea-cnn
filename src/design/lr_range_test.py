#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lr_range_test.py
----------------
LR Range Test (Leslie Smith, 2017) para fine-tuning de la CNN sobre datos STNF.

Referencia: Smith, L.N. "Cyclical Learning Rates for Training Neural Networks"
            WACV 2017. arXiv:1506.01186 — Sección 3.3

Fidelidad al paper (Sección 3.3):
  - Registra ACCURACY además de loss (Smith interpreta el test con accuracy)
  - Detecta base_lr = donde la accuracy empieza a subir de forma sostenida
  - Detecta max_lr  = donde la accuracy se vuelve irregular o cae
  - Calcula optimal_lr = max_lr / 2 (regla de thumb de Smith Sec. 3.3)
  - batch_size = 2000, igual que el entrenamiento real (el LR óptimo depende
    del batch size — usar uno distinto daría resultados no comparables)
  - Modelo de partida: MODEL_BEST_PATH de src.design.config
    (data/experiments/train_results/t050_bs2000/best_model.h5)
  - num_steps = 557, equivalente a 1 época completa con batch=2000
    (1.112.872 / 2000 = 556.4 → redondeado a 557)
  - Rango: 1e-6 a 1e-2 (justificado por ejecución anterior y ReduceLROnPlateau
    que acabó en 2.5e-5, confirmando que la zona útil está en ese rango)
  - Subida logarítmica (justificada: rango > 2 órdenes de magnitud)

Desviación documentada respecto al paper:
  - Smith propone subida LINEAL del LR. Aquí se usa escala LOGARÍTMICA porque
    el rango explorado abarca varios órdenes de magnitud; con subida lineal
    casi todos los steps se concentrarían en la zona alta sin explorar la baja.

Salida:
  - data/LR_Range_Test/lr_loss_log.json
  - data/LR_Range_Test/lr_range_report.txt

USO (desde ProyectoPython3.0/):
  python -m src.design.lr_range_test [--batch_size 2000] [--num_steps 557]
                                     [--lr_min 1e-6] [--lr_max 1e-2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.design.cnn_model import build_cnn
from src.design.config import IMPUT_DATA_DIR, MODEL_BEST_PATH

# ── Rutas ──────────────────────────────────────────────────────────────
DATA_DIR     = Path(IMPUT_DATA_DIR)
WEIGHTS_PATH = Path(MODEL_BEST_PATH)
OUT_DIR      = Path("data/LR_Range_Test")


def smooth(values, beta=0.9):
    """Suavizado exponencial con corrección de bias (igual que Adam internamente)."""
    smoothed, avg = [], 0.0
    for i, v in enumerate(values):
        avg = beta * avg + (1 - beta) * v
        smoothed.append(avg / (1 - beta ** (i + 1)))
    return smoothed


def find_boundaries(lrs, accs_smooth):
    """
    Detecta base_lr y max_lr según criterio de Smith (Sección 3.3):
      - base_lr: LR donde la accuracy empieza a subir de forma sostenida
                 (primer punto donde la derivada es positiva y se mantiene)
      - max_lr:  LR donde la accuracy se vuelve irregular o empieza a caer
                 (primer punto donde la derivada se vuelve negativa de forma sostenida)
    """
    lrs  = np.array(lrs)
    accs = np.array(accs_smooth)
    grad = np.gradient(accs)

    # base_lr: primer idx donde el gradiente es positivo y los 3 siguientes también
    base_idx = None
    for i in range(len(grad) - 3):
        if all(grad[i:i+4] > 0):
            base_idx = i
            break

    # max_lr: después del base_lr, primer idx donde el gradiente es negativo
    # y los 3 siguientes también (accuracy empieza a caer sostenidamente)
    max_idx = None
    if base_idx is not None:
        for i in range(base_idx + 5, len(grad) - 3):
            if all(grad[i:i+4] < 0):
                max_idx = i
                break

    base_lr = float(lrs[base_idx]) if base_idx is not None else float(lrs[0])
    max_lr  = float(lrs[max_idx])  if max_idx  is not None else float(lrs[-1])

    return base_lr, max_lr, base_idx, max_idx


def main():
    parser = argparse.ArgumentParser(description="LR Range Test — Smith 2017 Sec. 3.3")
    parser.add_argument("--batch_size", type=int,   default=2000,
                        help="Igual que el entrenamiento real (LR óptimo depende del batch size)")
    parser.add_argument("--num_steps",  type=int,   default=557,
                        help="1 época completa con batch=2000 (1.112.872 / 2000 = 557)")
    parser.add_argument("--lr_min",     type=float, default=1e-6,
                        help="LR mínimo — por debajo es zona plana según ejecución anterior")
    parser.add_argument("--lr_max",     type=float, default=1e-2,
                        help="LR máximo — por encima diverge según ejecución anterior")
    parser.add_argument("--beta",       type=float, default=0.9,
                        help="Factor de suavizado EMA")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── GPU ───────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[INFO] GPUs disponibles: {gpus}")
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # ── Cargar datos ──────────────────────────────────────────────────
    print("[INFO] Cargando X_train / y_train...")
    X = np.load(DATA_DIR / "X_train.npy", mmap_mode="r")
    y = np.load(DATA_DIR / "y_train.npy")
    y = (y > 0.5).astype(np.float32)

    print(f"[INFO] X_train: {X.shape}  y_train: {y.shape}")
    print(f"[INFO] Positivos: {int(y.sum())}  Negativos: {int((1-y).sum())}")

    # ── Subsample balanceado ──────────────────────────────────────────
    n_needed = min(args.num_steps * args.batch_size, len(y))
    rng      = np.random.default_rng(42)
    pos_idx  = np.where(y == 1)[0]
    neg_idx  = np.where(y == 0)[0]
    n_each   = min(n_needed // 2, len(pos_idx), len(neg_idx))
    sel      = np.concatenate([
        rng.choice(pos_idx, n_each, replace=False),
        rng.choice(neg_idx, n_each, replace=False)
    ])
    rng.shuffle(sel)

    X_sub = np.asarray(X[sel]).astype(np.float32)
    y_sub = y[sel]
    print(f"[INFO] Subconjunto para LR test: {X_sub.shape}")

    # ── Construir modelo y cargar best_model (batch=2000) ─────────────
    print("[INFO] Construyendo modelo y cargando best_model batch=2000...")
    model = build_cnn(input_shape=(2, 500, 1))
    model.load_weights(str(WEIGHTS_PATH), by_name=False)
    print("[OK] Pesos cargados.")

    # ── Preparar dataset TF ───────────────────────────────────────────
    dataset = (
        tf.data.Dataset.from_tensor_slices((X_sub, y_sub))
        .shuffle(buffer_size=len(y_sub), seed=42)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── LR Range Test ─────────────────────────────────────────────────
    lr_min    = args.lr_min
    lr_max    = args.lr_max
    num_steps = args.num_steps
    lr_factor = (lr_max / lr_min) ** (1.0 / num_steps)

    loss_fn   = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_min)

    lrs, losses, accs = [], [], []
    current_lr = lr_min
    best_loss  = float("inf")
    step       = 0

    print(f"\n[INFO] Iniciando LR Range Test (Smith 2017, Sec. 3.3):")
    print(f"       LR:    {lr_min:.2e} → {lr_max:.2e}  (escala logarítmica)")
    print(f"       Steps: {num_steps}  |  batch_size: {args.batch_size}")
    print("-" * 60)

    for X_batch, y_batch in dataset:
        if step >= num_steps:
            break

        optimizer.learning_rate.assign(current_lr)

        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss   = loss_fn(y_batch, y_pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_val = float(loss.numpy())

        # Accuracy del batch (threshold=0.5, igual que Smith)
        acc_val = float(
            tf.reduce_mean(
                tf.cast(tf.equal(y_batch, tf.round(y_pred)), tf.float32)
            ).numpy()
        )

        lrs.append(current_lr)
        losses.append(loss_val)
        accs.append(acc_val)

        if loss_val < best_loss:
            best_loss = loss_val

        # Parada por divergencia
        if loss_val > 4 * best_loss and step > 10:
            print(f"[STOP] Loss divergió en step {step} (lr={current_lr:.2e})")
            break

        if step % 20 == 0:
            print(f"  step {step:4d} | lr={current_lr:.2e} | "
                  f"loss={loss_val:.4f} | acc={acc_val:.4f}")

        current_lr *= lr_factor
        step += 1

    # ── Suavizar ──────────────────────────────────────────────────────
    losses_smooth = smooth(losses, beta=args.beta)
    accs_smooth   = smooth(accs,   beta=args.beta)

    # ── Detectar base_lr y max_lr (criterio Smith Sec. 3.3) ──────────
    base_lr, max_lr, base_idx, max_idx = find_boundaries(lrs, accs_smooth)

    # LR óptimo según regla de Smith Sec. 3.3:
    # "the optimum learning rate is usually within a factor of two
    #  of the largest one that converges"
    optimal_lr = max_lr / 2.0

    print("\n" + "=" * 60)
    print("  LR RANGE TEST — RESULTADO (Smith 2017, Sec. 3.3)")
    print("=" * 60)
    print(f"  Steps ejecutados : {step}")
    print(f"  Loss mínima      : {min(losses_smooth):.4f}")
    print(f"  Acc. máxima      : {max(accs_smooth):.4f}")
    print(f"  base_lr          : {base_lr:.2e}  (límite inferior útil, zona plana por debajo)")
    print(f"  max_lr           : {max_lr:.2e}  (accuracy empieza a caer)")
    print(f"  optimal_lr       : {optimal_lr:.2e}  (= max_lr / 2, regla Smith Sec. 3.3)")
    print(f"  → Usar optimal_lr = {optimal_lr:.2e} como LR inicial de Adam")
    print(f"  → ReduceLROnPlateau bajará adaptativamente hasta ~{base_lr:.2e}")
    print("=" * 60)

    # ── Guardar resultados ────────────────────────────────────────────
    log = {
        "lr_min_sweep":   lr_min,
        "lr_max_sweep":   lr_max,
        "batch_size":     args.batch_size,
        "num_steps":      step,
        "base_lr":        base_lr,
        "max_lr":         max_lr,
        "optimal_lr":     optimal_lr,
        "lrs":            [float(x) for x in lrs],
        "losses_raw":     [float(x) for x in losses],
        "losses_smooth":  [float(x) for x in losses_smooth],
        "accs_raw":       [float(x) for x in accs],
        "accs_smooth":    [float(x) for x in accs_smooth],
        "base_lr_idx":    int(base_idx) if base_idx is not None else None,
        "max_lr_idx":     int(max_idx)  if max_idx  is not None else None,
    }

    log_path    = OUT_DIR / "lr_loss_log.json"
    report_path = OUT_DIR / "lr_range_report.txt"

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    report_lines = [
        "LR RANGE TEST REPORT — Smith 2017, Sec. 3.3",
        "=" * 45,
        f"LR barrido:      {lr_min:.2e} → {lr_max:.2e}",
        f"Batch size:      {args.batch_size}  (igual que entrenamiento real)",
        f"Steps:           {step}  (≈ 1 época completa)",
        f"Loss mínima:     {min(losses_smooth):.4f}",
        f"Acc. máxima:     {max(accs_smooth):.4f}",
        "",
        "LÍMITES DETECTADOS (criterio Smith Sec. 3.3):",
        f"  base_lr    : {base_lr:.2e}  (accuracy empieza a subir)",
        f"  max_lr     : {max_lr:.2e}  (accuracy empieza a caer)",
        f"  optimal_lr : {optimal_lr:.2e}  (= max_lr / 2, regla Smith Sec. 3.3)",
        "",
        "INTERPRETACIÓN:",
        f"  Usar optimal_lr = {optimal_lr:.2e} como LR inicial de Adam",
        f"  ReduceLROnPlateau bajará adaptativamente desde {optimal_lr:.2e} hasta ~{base_lr:.2e}",
        f"  Por debajo de base_lr = {base_lr:.2e} el modelo entra en zona plana",
        f"  Para CLR triangular: oscilar entre {base_lr:.2e} y {max_lr:.2e}",
        "",
        "REFERENCIA:",
        "  Smith, L.N. 'Cyclical Learning Rates for Training Neural Networks'",
        "  WACV 2017. arXiv:1506.01186",
    ]
    report_path.write_text("\n".join(report_lines))

    print(f"\n[OK] Log guardado en:     {log_path}")
    print(f"[OK] Reporte guardado en: {report_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()