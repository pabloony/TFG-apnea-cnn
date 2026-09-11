#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_zscore.py
--------------
Verifica que la normalización z-score no borra la separación entre clases.
Carga un joined_*.npz, normaliza, y reporta:
  1. μ y σ de cada canal POR CLASE (antes y después de normalizar)
  2. Separabilidad por MEDIA (Cohen's d sobre media de ventana)
  3. Separabilidad por AMPLITUD (Cohen's d sobre std de ventana)
  4. Guarda gráficas: histogramas de media y amplitud por canal y clase
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

JOINED_DIR = Path("data/complementation_results/joined_data")
OUT_DIR    = Path("data/experiments/diag_zscore")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS     = 1e-6
MIN_NEG = 10


# --------------------------------------------------
# Normalización (idéntica a tu pipeline)
# --------------------------------------------------
def zscore(X, y, eps=EPS, min_neg=MIN_NEG):
    Xn = X.astype(np.float32, copy=True)
    neg_mask = (y == 0)
    use_neg  = int(neg_mask.sum()) >= min_neg
    for ch in range(Xn.shape[1]):
        ref = Xn[neg_mask, ch, :] if use_neg else Xn[:, ch, :]
        mu  = ref.mean(dtype=np.float32)
        sd  = ref.std(dtype=np.float32)
        if sd < eps:
            sd = 1.0
        Xn[:, ch, :] = (Xn[:, ch, :] - mu) / (sd + eps)
    return Xn


# --------------------------------------------------
# Métricas de separabilidad
# --------------------------------------------------
def cohens_d_mean(pos_windows, neg_windows):
    """Cohen's d sobre la MEDIA de cada ventana (detecta desplazamiento DC)."""
    a = pos_windows.mean(axis=1)
    b = neg_windows.mean(axis=1)
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2 + EPS)
    return abs(a.mean() - b.mean()) / pooled


def cohens_d_std(pos_windows, neg_windows):
    """Cohen's d sobre la STD de cada ventana (detecta cambios de amplitud)."""
    a = pos_windows.std(axis=1)
    b = neg_windows.std(axis=1)
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2 + EPS)
    return abs(a.mean() - b.mean()) / pooled


# --------------------------------------------------
# Reporte + gráfica
# --------------------------------------------------
def report_and_plot(X, y, label, out_prefix):
    ch_names = ["airflow", "spo2"]

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    for ch, name in enumerate(ch_names):
        pos = X[y == 1, ch, :]
        neg = X[y == 0, ch, :]

        d_mean = cohens_d_mean(pos, neg)
        d_amp  = cohens_d_std(pos, neg)

        print(f"  [{name}]")
        print(f"    μ_pos={pos.mean():.4f}  σ_pos={pos.std():.4f}  |  "
              f"μ_neg={neg.mean():.4f}  σ_neg={neg.std():.4f}")
        print(f"    Cohen's d (media)    = {d_mean:.4f}")
        print(f"    Cohen's d (amplitud) = {d_amp:.4f}  ← cambio de amplitud por apnea")

    # Gráfica 2x2: fila 0 = medias de ventana, fila 1 = stds de ventana
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{out_prefix}  |  {label}", fontsize=12)

    for ch, name in enumerate(ch_names):
        pos = X[y == 1, ch, :]
        neg = X[y == 0, ch, :]

        # --- Fila 0: distribución de MEDIAS ---
        ax = axes[0][ch]
        window_means_pos = pos.mean(axis=1)
        window_means_neg = neg.mean(axis=1)
        d_mean = cohens_d_mean(pos, neg)
        ax.hist(window_means_neg, bins=80, alpha=0.6,
                label=f"clase 0 (normal) n={len(neg)}", color="steelblue")
        ax.hist(window_means_pos, bins=80, alpha=0.6,
                label=f"clase 1 (apnea)  n={len(pos)}", color="tomato")
        ax.set_title(f"{name} — media de ventana   Cohen's d={d_mean:.3f}")
        ax.set_xlabel("media de ventana")
        ax.set_ylabel("nº ventanas")
        ax.legend(fontsize=8)

        # --- Fila 1: distribución de AMPLITUD (std) ---
        ax = axes[1][ch]
        window_stds_pos = pos.std(axis=1)
        window_stds_neg = neg.std(axis=1)
        d_amp = cohens_d_std(pos, neg)
        ax.hist(window_stds_neg, bins=80, alpha=0.6,
                label=f"clase 0 (normal) n={len(neg)}", color="steelblue")
        ax.hist(window_stds_pos, bins=80, alpha=0.6,
                label=f"clase 1 (apnea)  n={len(pos)}", color="tomato")
        ax.set_title(f"{name} — amplitud (std por ventana)   Cohen's d={d_amp:.3f}")
        ax.set_xlabel("std de ventana")
        ax.set_ylabel("nº ventanas")
        ax.legend(fontsize=8)

    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("/", "-")
    out_path = OUT_DIR / f"{out_prefix}_{safe_label}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  → Guardada: {out_path.name}")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(npz_name=None):
    files = sorted(JOINED_DIR.glob("joined_*.npz"))
    if not files:
        sys.exit(f"[ERROR] No hay joined_*.npz en {JOINED_DIR.resolve()}")

    if npz_name:
        path = JOINED_DIR / npz_name
        if not path.exists():
            sys.exit(f"[ERROR] No se encuentra: {path.resolve()}")
    else:
        path = files[0]

    print(f"[INFO] Analizando: {path.name}")
    data  = np.load(path)
    X_raw = data["joined_windows"].astype(np.float32)
    y     = data["labels"].astype(int)

    print(f"[INFO] Shape: {X_raw.shape} | "
          f"pos={y.sum()} neg={(y==0).sum()}")

    prefix = path.stem  # e.g. joined_STNF00490

    # Antes de normalizar
    report_and_plot(X_raw, y, "ANTES de z-score", prefix)

    # Después de normalizar
    X_norm = zscore(X_raw, y)
    report_and_plot(X_norm, y, "DESPUES de z-score", prefix)

    print(f"\n[OK] Diagnóstico completado. Imágenes en: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)