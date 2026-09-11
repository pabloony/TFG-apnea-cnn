# src/graphs/plot_example_airflow.py
# =============================================================================
# Figura de ejemplo: un tramo de airflow crudo de un sujeto STNF y uno STLK,
# lado a lado, para MOSTRAR que las magnitudes crudas son distintas.
#
# No calcula estadísticos ni CSV: solo dibuja. Se ejecuta desde la raíz:
#   python -m src.graphs.plot_example_airflow --stnf STNF00046 --stlk STLK00014
#   python -m src.graphs.plot_example_airflow --stnf STNF00046 --stlk STLK00014 --seconds 90
# =============================================================================

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla (cluster)
import matplotlib.pyplot as plt

try:
    from .config import GRAPHS_OUTPUT_DIR, DPI, DTA_RESULTS_DIR
except Exception:
    GRAPHS_OUTPUT_DIR = "src/graphs/results/"
    DPI = 150
    DTA_RESULTS_DIR = "data/module_results/dta_results/"

COLOR_STNF = "#1D9E75"
COLOR_STLK = "#D85A30"


def load_airflow(results_dir: Path, subject: str):
    """Carga airflow y fs de un sujeto desde su .npz."""
    p = results_dir / f"{subject}_dta_results.npz"
    if not p.exists():
        raise SystemExit(f"No existe: {p}")
    d = np.load(p, allow_pickle=True)
    air = np.asarray(d["airflow"], dtype=float)
    fs = float(d["fs_airflow"]) if "fs_airflow" in d else 32.0
    return air, fs


def pick_clean_segment(air, fs, seconds, seed=42, n_try=40):
    """
    Elige un tramo 'limpio' dentro del tercio central de la señal.
    Prueba n_try posiciones aleatorias y se queda con la que tenga menos
    muestras cercanas a cero (evita huecos de sensor / zonas planas).
    """
    n = int(seconds * fs)
    if len(air) <= n:                     # señal más corta que el tramo pedido
        seg = air
        return np.arange(len(seg)) / fs, seg

    # Ventana de búsqueda: tercio central de la noche
    lo = len(air) // 3
    hi = max(lo + 1, 2 * len(air) // 3 - n)

    rng = np.random.default_rng(seed)
    best_start, best_score = lo, np.inf
    for _ in range(n_try):
        start = int(rng.integers(lo, hi)) if hi > lo else lo
        seg = air[start:start + n]
        # "score" = fracción de muestras casi nulas + penalización si es casi plano
        near_zero = float(np.mean(np.abs(seg) < 1e-6))
        flat = 0.0 if np.std(seg) > 1e-6 else 1.0
        score = near_zero + flat
        if score < best_score:
            best_score, best_start = score, start
        if best_score == 0.0:             # tramo perfecto, no hace falta seguir
            break

    seg = air[best_start:best_start + n]
    t = np.arange(len(seg)) / fs
    return t, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stnf", required=True, help="ID STNF, ej. STNF00046")
    ap.add_argument("--stlk", required=True, help="ID STLK, ej. STLK00014")
    ap.add_argument("--seconds", type=float, default=60.0, help="Duración del tramo (s)")
    ap.add_argument("--results-dir", default=DTA_RESULTS_DIR)
    ap.add_argument("--out-dir", default=GRAPHS_OUTPUT_DIR)
    ap.add_argument("--shared-y", action="store_true",
                    help="Usa el MISMO eje Y en ambos (para ver la diferencia real de escala)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Cambia la semilla para obtener otro tramo distinto")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    air_stnf, fs_stnf = load_airflow(results_dir, args.stnf)
    air_stlk, fs_stlk = load_airflow(results_dir, args.stlk)

    t1, s1 = pick_clean_segment(air_stnf, fs_stnf, args.seconds, seed=args.seed)
    t2, s2 = pick_clean_segment(air_stlk, fs_stlk, args.seconds, seed=args.seed)

    sharey = args.shared_y
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=sharey)

    axes[0].plot(t1, s1, color=COLOR_STNF, lw=0.9)
    axes[0].set_title(f"STNF · {args.stnf} · NasOr (fs={fs_stnf:.0f} Hz)")
    axes[0].set_ylabel("Airflow (u.a.)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t2, s2, color=COLOR_STLK, lw=0.9)
    axes[1].set_title(f"STLK · {args.stlk} · Nasal_Therm (fs={fs_stlk:.0f} Hz)")
    axes[1].set_ylabel("Airflow (u.a.)")
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].grid(True, alpha=0.3)

    tag = "sharedY" if sharey else "indepY"
    suptitle = "Airflow crudo (sin normalizar)"
    suptitle += " · mismo eje Y" if sharey else " · ejes Y independientes"
    fig.suptitle(suptitle)
    fig.tight_layout()

    out = out_dir / f"airflow_example_{args.stnf}_vs_{args.stlk}_{tag}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[OK] Figura → {out.resolve()}")


if __name__ == "__main__":
    main()