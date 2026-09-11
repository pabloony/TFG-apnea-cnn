#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/design/monitor_training.py
--------------------------------
Monitor en vivo de un entrenamiento en curso. Lee history.csv (lo escribe
CSVLogger al terminar cada epoca) y muestra una tabla que se refresca sola
en terminal + regenera una grafica PNG cada vez que hay una epoca nueva.

No reemplaza el log de SLURM (ahi ves el progreso DENTRO de una epoca,
batch a batch) — esto resume el progreso ENTRE epocas, que es lo que se
pierde haciendo scroll hacia abajo en un log de 50 epocas.

USO (en una sesion de terminal SEPARADA de la que lanza el job, mientras
el entrenamiento corre en background vía SLURM):

  python -m src.design.monitor_training

Se actualiza cada 5s. Ctrl+C para salir (no afecta al training, solo lee
el csv).
"""

import time
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import HISTORY_CSV_PATH, TRAIN_RESULTS_DIR, EXPERIMENT_TAG, EPOCHS

HISTORY_PATH = Path(HISTORY_CSV_PATH)
PLOT_PATH = Path(TRAIN_RESULTS_DIR) / "live_training_progress.png"
REFRESH_SECONDS = 5


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def load_history():
    if not HISTORY_PATH.exists():
        return None
    try:
        df = pd.read_csv(HISTORY_PATH)
        if df.empty:
            return None
        return df
    except Exception:
        # CSVLogger puede estar a mitad de escribir la fila justo en este instante
        return None


def fmt(x, nd=4):
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def render_table(df: pd.DataFrame, last_refresh_dt: float):
    n_epochs_done = len(df)
    last = df.iloc[-1]

    best_idx = df["val_auc"].idxmax()
    best_row = df.loc[best_idx]

    avg_epoch_time = last_refresh_dt  # aproximado, ver nota abajo
    remaining_epochs = max(EPOCHS - n_epochs_done, 0)

    print("=" * 78)
    print(f"  MONITOR EN VIVO — {EXPERIMENT_TAG}")
    print("=" * 78)
    print(f"  Epoca actual completada : {n_epochs_done} / {EPOCHS}")
    print(f"  Restantes               : {remaining_epochs}")
    print("-" * 78)
    print(f"  {'':<14}{'train':<14}{'val':<14}")
    print(f"  {'loss':<14}{fmt(last.get('loss')):<14}{fmt(last.get('val_loss')):<14}")
    print(f"  {'accuracy':<14}{fmt(last.get('accuracy')):<14}{fmt(last.get('val_accuracy')):<14}")
    print(f"  {'auc':<14}{fmt(last.get('auc')):<14}{fmt(last.get('val_auc')):<14}")
    print("-" * 78)
    print(f"  MEJOR val_auc hasta ahora: {fmt(best_row['val_auc'])}  (época {int(best_row['epoch']) + 1})")
    print(f"  Referencia (modelo base STNF, t050_bs2000_d35_rlrp): 0.7888")
    delta = best_row["val_auc"] - 0.7888
    signo = "+" if delta >= 0 else ""
    print(f"  Delta vs. referencia: {signo}{delta:.4f}")
    print("-" * 78)
    print(f"  Última actualización del csv hace ~{last_refresh_dt:.0f}s  (refresco cada {REFRESH_SECONDS}s)")
    print("=" * 78)
    print("\n  Últimas 5 épocas:")
    cols = [c for c in ["epoch", "loss", "auc", "val_loss", "val_auc", "learning_rate"] if c in df.columns]
    print(df[cols].tail(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Gráfica actualizada en: {PLOT_PATH}")
    print("  (Ctrl+C para salir — no afecta al training)")


def update_plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(df["epoch"], df["auc"], label="Train AUC", linestyle="--", color="#BA7517")
    axes[0].plot(df["epoch"], df["val_auc"], label="Val AUC", color="#1D9E75")
    axes[0].axhline(0.7888, color="gray", linestyle=":", linewidth=1, label="Referencia base (0.7888)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("AUC — en vivo")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, linewidth=0.5)

    axes[1].plot(df["epoch"], df["loss"], label="Train loss", linestyle="--", color="#BA7517")
    axes[1].plot(df["epoch"], df["val_loss"], label="Val loss", color="#D85A30")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss — en vivo")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, linewidth=0.5)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.suptitle(f"{EXPERIMENT_TAG} — última época: {int(df['epoch'].iloc[-1]) + 1}/{EPOCHS}")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[INFO] Esperando a que aparezca history.csv...")
    last_n_rows = -1
    t_last_change = time.time()

    while True:
        df = load_history()
        if df is not None:
            if len(df) != last_n_rows:
                # hay una epoca nueva (o es la primera lectura) -> refrescar plot
                update_plot(df)
                last_n_rows = len(df)
                t_last_change = time.time()

            clear_screen()
            render_table(df, time.time() - t_last_change)

            if last_n_rows >= EPOCHS:
                print("\n[OK] Entrenamiento completado (o EarlyStopping activado).")
                break
        else:
            clear_screen()
            print("[INFO] history.csv no existe todavía o está vacío. Esperando primera época...")

        try:
            time.sleep(REFRESH_SECONDS)
        except KeyboardInterrupt:
            print("\n[INFO] Monitor detenido por el usuario (el training sigue corriendo).")
            sys.exit(0)


if __name__ == "__main__":
    main()