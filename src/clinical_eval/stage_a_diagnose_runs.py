#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_a_diagnose_runs.py
------------------------
FASE 9 · ETAPA A · DIAGNÓSTICO — ¿Por qué el contador infracuenta?

Contexto
--------
La validación del contador sobre y_true (stage_a_validate.py) mostró una
infracuenta sistemática (~-12 eventos/sujeto) que EMPEORA con la severidad:
pacientes con muchos eventos reales se quedan muy cortos. Incluso con el
contador más permisivo (tol=0, min_vent=1).

Hipótesis
---------
El nivel de ventana (W=15s) no tiene resolución para separar eventos muy
seguidos: dos apneas reales próximas generan tiradas de ventanas positivas que
se TOCAN (sin ningún 0 entre medias), y el contador las ve como UN solo run.
Si es así, veríamos:
  - runs anormalmente largos en y_true (40, 60, 100+ ventanas),
  - nº de runs << nº de eventos reales,
  - la diferencia concentrada en sujetos con runs largos / muchos eventos.

Este script NO calibra nada: solo describe la estructura de los runs de y_true
para confirmar o descartar la hipótesis antes de decidir cómo seguir.

Qué produce
-----------
  - Histograma global de longitudes de run (en ventanas y en segundos).
  - Scatter: nº de runs (tol=0,minvt=1) vs nº eventos reales, por sujeto.
  - Scatter: longitud media/máxima de run vs infracuenta, por sujeto.
  - CSV por sujeto con: n_runs, n_real, deficit, run_len_mean/max, etc.
  - Resumen por pantalla.

Uso
---
    python -m src.clinical_eval.stage_a_diagnose_runs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    CLINICAL_PROBS_CSV,
    AHI_GT_CSV,
    FASE9_STAGE_A_DIR,
    WINDOW_S,
    STRIDE_S,
)
from .event_counter import _find_runs, event_durations_s


def _load_predictions(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        usecols=["subject_id", "win_idx", "y_true"],
        dtype={"subject_id": str, "win_idx": np.int32, "y_true": np.int8},
    )


def _load_ground_truth(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    for c in ["n_obs", "n_cen", "n_mix", "n_hyp"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce").fillna(0).astype(int)
    gt["target_mask_events"] = gt["n_obs"] + gt["n_cen"] + gt["n_mix"] + gt["n_hyp"]
    return gt.set_index("s_code")


def main() -> None:
    out_dir = Path(FASE9_STAGE_A_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[FASE 9 · ETAPA A · DIAG] Estructura de runs en y_true")
    print("=" * 60)

    pred = _load_predictions(Path(CLINICAL_PROBS_CSV))
    gt = _load_ground_truth(Path(AHI_GT_CSV))

    all_run_lengths = []          # longitudes de run globales (en ventanas)
    rows = []

    for sid, g in pred.groupby("subject_id"):
        if sid not in gt.index:
            continue
        y = g.sort_values("win_idx")["y_true"].to_numpy(dtype=np.int8)
        runs = _find_runs(y)                      # runs crudos (tol=0, sin filtro)
        run_lens = [e - s + 1 for (s, e) in runs]  # longitud en ventanas
        all_run_lengths.extend(run_lens)

        n_runs = len(runs)
        n_real = int(gt.loc[sid, "target_mask_events"])
        durs = event_durations_s(runs, WINDOW_S, STRIDE_S)

        rows.append({
            "subject_id": sid,
            "n_runs": n_runs,
            "n_real": n_real,
            "deficit": n_real - n_runs,           # >0 = infracuenta
            "run_len_mean": float(np.mean(run_lens)) if run_lens else 0.0,
            "run_len_max": int(np.max(run_lens)) if run_lens else 0,
            "dur_max_s": float(np.max(durs)) if durs else 0.0,
            "pos_windows": int(y.sum()),
        })

    df = pd.DataFrame(rows).sort_values("deficit", ascending=False)
    csv_path = out_dir / "stage_a_diag_runs.csv"
    df.to_csv(csv_path, index=False)

    all_run_lengths = np.array(all_run_lengths)

    # --- Gráfica 1: histograma de longitudes de run ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(all_run_lengths, bins=60, color="#3B7A9E", edgecolor="k", linewidth=0.3)
    ax[0].set_xlabel("Longitud de run (nº ventanas)")
    ax[0].set_ylabel("Frecuencia")
    ax[0].set_title("Longitudes de run en y_true (global)")
    ax[0].axvline(np.median(all_run_lengths), color="r", ls="--",
                  label=f"mediana={np.median(all_run_lengths):.0f}")
    ax[0].legend()

    dur_all = (all_run_lengths - 1) * STRIDE_S + WINDOW_S
    ax[1].hist(dur_all, bins=60, color="#9E6B3B", edgecolor="k", linewidth=0.3)
    ax[1].set_xlabel("Duración de run (s)")
    ax[1].set_ylabel("Frecuencia")
    ax[1].set_title("Duración de run en y_true (global)")
    ax[1].axvline(10, color="g", ls="--", label="10s (mín AASM)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stage_a_diag_run_hist.png", dpi=130)
    plt.close(fig)

    # --- Gráfica 2: n_runs vs n_real ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["n_real"], df["n_runs"], alpha=0.6, edgecolor="k", linewidth=0.3)
    lim = max(df["n_real"].max(), df["n_runs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="identidad")
    ax.set_xlabel("Eventos reales (4 tipos)")
    ax.set_ylabel("nº de runs en y_true (tol=0, sin filtro)")
    ax.set_title("¿Cuántos eventos fusiona la ventana?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stage_a_diag_runs_vs_real.png", dpi=130)
    plt.close(fig)

    # --- Gráfica 3: longitud máxima de run vs déficit ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(df["run_len_max"], df["deficit"], c=df["n_real"],
                    cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Longitud máxima de run (ventanas)")
    ax.set_ylabel("Déficit (n_real - n_runs)")
    ax.set_title("Runs largos → más infracuenta")
    fig.colorbar(sc, ax=ax, label="eventos reales")
    fig.tight_layout()
    fig.savefig(out_dir / "stage_a_diag_deficit.png", dpi=130)
    plt.close(fig)

    # --- Resumen por pantalla ---
    print(f"[INFO] Sujetos analizados       : {len(df)}")
    print(f"[INFO] Runs totales             : {len(all_run_lengths)}")
    print(f"[INFO] Long. run (ventanas)     : "
          f"mediana={np.median(all_run_lengths):.0f}, "
          f"p90={np.percentile(all_run_lengths,90):.0f}, "
          f"max={all_run_lengths.max()}")
    print(f"[INFO] Duración run (s)         : "
          f"mediana={np.median(dur_all):.0f}, "
          f"p90={np.percentile(dur_all,90):.0f}, "
          f"max={dur_all.max():.0f}")
    print(f"[INFO] Déficit total            : "
          f"{int(df['deficit'].sum())} eventos (real - runs)")
    print(f"[INFO] Corr(run_len_max, deficit): "
          f"{df['run_len_max'].corr(df['deficit']):.3f}")
    print("-" * 60)
    print("[TOP 8 sujetos con más déficit]")
    print(df.head(8).to_string(index=False))
    print("-" * 60)
    print(f"[OK] CSV       : {csv_path}")
    print(f"[OK] Histograma: {out_dir / 'stage_a_diag_run_hist.png'}")
    print(f"[OK] Runs vs real: {out_dir / 'stage_a_diag_runs_vs_real.png'}")
    print(f"[OK] Déficit   : {out_dir / 'stage_a_diag_deficit.png'}")
    print("[DONE]")


if __name__ == "__main__":
    main()