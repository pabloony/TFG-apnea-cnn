#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_zscore_percentile.py
------------------------------
Corrige el diagnóstico anterior: Cohen's d es invariante a cualquier
normalización lineal (std o clip), así que nunca podía detectar diferencia
entre métodos -- eso fue un error de diseño de la prueba anterior, no un
resultado real.

Aquí medimos lo que sí puede diferir entre métodos:
  1. Cuánto "aplasta" el método estándar la señal normal en sujetos con
     outliers, comparado con el método percentile-clip.
  2. Cuántos valores normalizados con std caen en un rango extremo
     (|z| > 5) solo por el efecto del outlier arrastrando la sigma.
  3. Gráfica de la señal normalizada por ambos métodos, para los sujetos
     con sigma_ratio más alto (los candidatos reales).

No entrena nada. No toca GPU.

Uso:
    python diag_zscore_percentile_v2.py                # top 15 sujetos
    python diag_zscore_percentile_v2.py STNF00012       # un sujeto concreto
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

JOINED_DIR = Path("data/complementation_results/joined_data")
OUT_DIR    = Path("data/experiments/diag_zscore_percentile")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS     = 1e-6
MIN_NEG = 10
P_LOW, P_HIGH = 1, 99


def zscore_standard(X, y, eps=EPS, min_neg=MIN_NEG):
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


def zscore_percentile_clip(X, y, eps=EPS, min_neg=MIN_NEG, p_low=P_LOW, p_high=P_HIGH):
    Xn = X.astype(np.float32, copy=True)
    neg_mask = (y == 0)
    use_neg  = int(neg_mask.sum()) >= min_neg
    for ch in range(Xn.shape[1]):
        ref = Xn[neg_mask, ch, :] if use_neg else Xn[:, ch, :]
        ref_flat = ref.ravel()
        lo, hi = np.percentile(ref_flat, [p_low, p_high])
        clipped = np.clip(ref_flat, lo, hi)
        mu = clipped.mean(dtype=np.float32)
        sd = clipped.std(dtype=np.float32)
        if sd < eps:
            sd = 1.0
        Xn[:, ch, :] = (Xn[:, ch, :] - mu) / (sd + eps)
    return Xn


# --------------------------------------------------
# Métrica 1: rango dinámico útil preservado
# Se mide como el percentil 5-95 de TODAS las ventanas normales
# (clase 0) tras normalizar -- cuanto más "aplastado" por un outlier,
# más estrecho será ese rango.
# --------------------------------------------------
def dynamic_range_p5_95(X_norm, y, ch):
    neg = X_norm[y == 0, ch, :].ravel()
    p5, p95 = np.percentile(neg, [5, 95])
    return p95 - p5


# --------------------------------------------------
# Métrica 2: cuántos valores "saturan" a |z| > 5 solo por el outlier
# --------------------------------------------------
def pct_extreme(X_norm, y, ch, z_thr=5.0):
    neg = X_norm[y == 0, ch, :].ravel()
    return 100.0 * np.mean(np.abs(neg) > z_thr)


def compare_subject(path, plot=False):
    data  = np.load(path)
    X_raw = data["joined_windows"].astype(np.float32)
    y     = data["labels"].astype(int)

    ch_names = ["airflow", "spo2"]
    results = {"id": path.stem}

    X_std  = zscore_standard(X_raw, y)
    X_clip = zscore_percentile_clip(X_raw, y)

    for ch, name in enumerate(ch_names):
        rng_std  = dynamic_range_p5_95(X_std, y, ch)
        rng_clip = dynamic_range_p5_95(X_clip, y, ch)
        ext_std  = pct_extreme(X_std, y, ch)
        ext_clip = pct_extreme(X_clip, y, ch)

        results[f"{name}_range_std"]  = rng_std
        results[f"{name}_range_clip"] = rng_clip
        results[f"{name}_range_gain"] = (rng_clip / rng_std - 1) * 100 if rng_std > 0 else np.nan
        results[f"{name}_ext_std"]  = ext_std
        results[f"{name}_ext_clip"] = ext_clip

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6))
        for ch, name in enumerate(ch_names):
            ax = axes[ch]
            neg_std  = X_std[y == 0, ch, :].ravel()
            neg_clip = X_clip[y == 0, ch, :].ravel()
            ax.hist(neg_std, bins=150, alpha=0.6, label="z-score estándar", color="#E24B4A", density=True)
            ax.hist(neg_clip, bins=150, alpha=0.6, label="z-score percentil 1-99", color="#1D9E75", density=True)
            ax.set_title(f"{name} — distribución normalizada (clase normal), {path.stem}")
            ax.set_xlabel("valor normalizado")
            ax.legend(fontsize=8)
        plt.tight_layout()
        out_path = OUT_DIR / f"{path.stem}_comparison.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  → Gráfica guardada: {out_path.name}")

    return results


def main(n_subjects=15, npz_name=None):
    if npz_name:
        candidates = list(JOINED_DIR.glob(f"joined_*{npz_name}*.npz"))
        if not candidates:
            sys.exit(f"[ERROR] No se encontró ningún npz que contenga '{npz_name}'")
        files = candidates
    else:
        files = sorted(JOINED_DIR.glob("joined_*.npz"))[:n_subjects]

    if not files:
        sys.exit(f"[ERROR] No hay joined_*.npz en {JOINED_DIR.resolve()}")

    print(f"[INFO] Analizando {len(files)} sujeto(s)\n")

    all_results = []
    # graficar solo si es 1 sujeto específico, o los 3 con más sigma_ratio
    plot_this_one = (npz_name is not None)

    for f in files:
        try:
            r = compare_subject(f, plot=plot_this_one)
            all_results.append(r)
            print(f"  {r['id']:<25s} "
                  f"rango airflow: std={r['airflow_range_std']:.2f} clip={r['airflow_range_clip']:.2f} "
                  f"(ganancia {r['airflow_range_gain']:+.1f}%)  |  "
                  f"rango spo2: std={r['spo2_range_std']:.2f} clip={r['spo2_range_clip']:.2f} "
                  f"(ganancia {r['spo2_range_gain']:+.1f}%)")
        except Exception as e:
            print(f"  [WARN] Falló {f.name}: {e}")

    if not all_results:
        sys.exit("[ERROR] Ningún sujeto procesado correctamente.")

    print(f"\n{'='*80}")
    print("RESUMEN AGREGADO")
    print(f"{'='*80}")
    for ch in ["airflow", "spo2"]:
        gains = [r[f"{ch}_range_gain"] for r in all_results]
        ext_std  = [r[f"{ch}_ext_std"]  for r in all_results]
        ext_clip = [r[f"{ch}_ext_clip"] for r in all_results]
        print(f"\n  [{ch}]")
        print(f"    Ganancia media de rango dinámico (clip vs std): {np.mean(gains):+.1f}%  "
              f"(máx: {np.max(gains):+.1f}%, sujeto más beneficiado)")
        print(f"    % valores extremos (|z|>5) -> std={np.mean(ext_std):.3f}%  clip={np.mean(ext_clip):.3f}%")

    print(f"\n[INFO] 'Ganancia de rango dinámico' > 0 significa que percentile-clip preserva "
          f"MÁS variación útil en la señal normal que el z-score estándar -- es la métrica "
          f"correcta para justificar (o descartar) el cambio, a diferencia de Cohen's d.")
    print(f"[INFO] Para ver la distribución completa de un sujeto concreto: "
          f"python {Path(__file__).name} STNF00012")


if __name__ == "__main__":
    n = 15
    name = None
    if len(sys.argv) > 1:
        name = sys.argv[1]
    main(n_subjects=n, npz_name=name)