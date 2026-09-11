#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_experiments.py
-----------------------
Genera gráficas comparativas entre experimentos a partir de history.csv
y val_y_prob.npy / y_val.npy.

Gráficas generadas (controladas desde config.py):
  - val_auc_curves.png       : val AUC superpuesto por experimento
  - val_loss_curves.png      : val loss superpuesto por experimento
  - train_val_auc.png        : train vs val AUC por experimento (subplots)
  - train_val_loss.png       : train vs val loss por experimento (subplots)
  - bar_best_auc.png         : barras best val AUC + época de convergencia
  - roc_curves.png           : curvas ROC superpuestas
  - pr_curves.png            : curvas Precision-Recall superpuestas
  - heatmap_summary.png      : heatmap hiperparámetros vs métricas

Uso:
    python -m src.graphs.compare_experiments --phase 1
    python -m src.graphs.compare_experiments --phase 2
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd

from src.graphs.config import (
    TRAIN_RESULTS_DIR,
    GRAPHS_OUTPUT_DIR,
    BASELINE_EXPERIMENT,
    PHASE1_EXPERIMENTS,
    PHASE2_EXPERIMENTS,
    PHASE3_EXPERIMENTS,
    PHASE4_EXPERIMENTS,
    PHASE5_EXPERIMENTS,
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    DPI,
    PLOT_VAL_AUC_CURVES,
    PLOT_VAL_LOSS_CURVES,
    PLOT_TRAIN_VAL_AUC,
    PLOT_TRAIN_VAL_LOSS,
    PLOT_BAR_BEST_AUC,
    PLOT_ROC_CURVES,
    PLOT_PR_CURVES,
    PLOT_HEATMAP_SUMMARY,
)

PHASE_MAP = {
    1: PHASE1_EXPERIMENTS,
    2: PHASE2_EXPERIMENTS,
    3: PHASE3_EXPERIMENTS,
    4: PHASE4_EXPERIMENTS,
    5: PHASE5_EXPERIMENTS,
}


# ================================================================
# Carga de datos
# ================================================================
def load_history(exp_tag: str, results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / exp_tag / "history.csv"
    if not path.exists():
        print(f"[WARN] No existe: {path}")
        return None
    return pd.read_csv(path)


def load_roc_data(exp_tag: str, results_dir: Path):
    """Carga y_prob y y_true para ROC/PR. Devuelve (y_true, y_prob) o (None, None)."""
    prob_path  = results_dir / exp_tag / "eval_val" / "val_y_prob.npy"
    threshold_tag = exp_tag.split("_")[0]  # "t050_bs2000" → "t050"
    ytrue_path = Path("data/complementation_results") / f"imput_{threshold_tag}" / "y_val.npy"

    if not prob_path.exists():
        print(f"[WARN] No existe val_y_prob.npy para {exp_tag}")
        return None, None
    if not ytrue_path.exists():
        print(f"[WARN] No existe y_val.npy para threshold={threshold_tag} en {ytrue_path}")
        return None, None

    y_prob = np.load(prob_path)
    y_true = np.load(ytrue_path)
    return y_true, y_prob

def get_label(exp_tag: str) -> str:
    return EXPERIMENT_LABELS.get(exp_tag, exp_tag)


def get_color(exp_tag: str) -> str:
    return EXPERIMENT_COLORS.get(exp_tag, "#888888")


# ================================================================
# Gráfica: val AUC superpuesto
# ================================================================
def plot_val_auc_curves(experiments: list, results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))

    for exp in experiments:
        hist = load_history(exp, results_dir)
        if hist is None:
            continue
        col = "val_auc" if "val_auc" in hist.columns else "val_auroc"
        if col not in hist.columns:
            print(f"[WARN] No se encuentra columna AUC en history de {exp}")
            continue
        ax.plot(hist[col].values, label=get_label(exp),
                color=get_color(exp), linewidth=1.8)

    ax.set_xlabel("Época", fontsize=10)
    ax.set_ylabel("Val AUC", fontsize=10)
    ax.set_title("Val AUC por época — comparativa", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "val_auc_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: val loss superpuesto
# ================================================================
def plot_val_loss_curves(experiments: list, results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))

    for exp in experiments:
        hist = load_history(exp, results_dir)
        if hist is None:
            continue
        if "val_loss" not in hist.columns:
            continue
        ax.plot(hist["val_loss"].values, label=get_label(exp),
                color=get_color(exp), linewidth=1.8)

    ax.set_xlabel("Época", fontsize=10)
    ax.set_ylabel("Val Loss", fontsize=10)
    ax.set_title("Val Loss por época — comparativa", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "val_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: train vs val AUC por experimento (subplots)
# ================================================================
def plot_train_val_auc(experiments: list, results_dir: Path, output_dir: Path) -> None:
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        hist = load_history(exp, results_dir)
        if hist is None:
            ax.set_title(f"{get_label(exp)}\n(sin datos)", fontsize=10)
            continue

        col_val   = "val_auc" if "val_auc" in hist.columns else "val_auroc"
        col_train = "auc" if "auc" in hist.columns else "auroc"

        if col_val in hist.columns:
            ax.plot(hist[col_val].values, color=get_color(exp),
                    linewidth=1.8, label="Val AUC")
        if col_train in hist.columns:
            ax.plot(hist[col_train].values, color=get_color(exp),
                    linewidth=1.8, linestyle="--", alpha=0.6, label="Train AUC")

        best_val = hist[col_val].max() if col_val in hist.columns else float("nan")
        best_ep  = hist[col_val].idxmax() if col_val in hist.columns else -1
        ax.set_title(f"{get_label(exp)}\nbest val AUC={best_val:.4f} (ep {best_ep})",
                     fontsize=9)
        ax.set_xlabel("Época", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("AUC", fontsize=10)
    fig.suptitle("Train vs Val AUC por experimento", fontsize=11)
    out = output_dir / "train_val_auc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: train vs val loss por experimento (subplots)
# ================================================================
def plot_train_val_loss(experiments: list, results_dir: Path, output_dir: Path) -> None:
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        hist = load_history(exp, results_dir)
        if hist is None:
            continue
        if "val_loss" in hist.columns:
            ax.plot(hist["val_loss"].values, color=get_color(exp),
                    linewidth=1.8, label="Val loss")
        if "loss" in hist.columns:
            ax.plot(hist["loss"].values, color=get_color(exp),
                    linewidth=1.8, linestyle="--", alpha=0.6, label="Train loss")

        ax.set_title(get_label(exp), fontsize=9)
        ax.set_xlabel("Época", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Loss", fontsize=10)
    fig.suptitle("Train vs Val Loss por experimento", fontsize=11)
    out = output_dir / "train_val_loss.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: barras best val AUC
# ================================================================
def plot_bar_best_auc(experiments: list, results_dir: Path, output_dir: Path) -> None:
    tags, best_aucs, best_eps = [], [], []

    for exp in experiments:
        hist = load_history(exp, results_dir)
        if hist is None:
            continue
        col = "val_auc" if "val_auc" in hist.columns else "val_auroc"
        if col not in hist.columns:
            continue
        tags.append(get_label(exp))
        best_aucs.append(hist[col].max())
        best_eps.append(int(hist[col].idxmax()))

    if not tags:
        print("[WARN] Sin datos para barras AUC")
        return

    x = np.arange(len(tags))
    colors = [get_color(exp) for exp in experiments if load_history(exp, results_dir) is not None]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(x, best_aucs, color=colors, width=0.5, alpha=0.9, edgecolor="white")

    for bar, auc_val, ep in zip(bars, best_aucs, best_eps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{auc_val:.4f}\n(ep {ep})",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel("Best Val AUC", fontsize=10)
    ax.set_title("Best Val AUC por experimento", fontsize=11)
    ax.set_ylim(min(best_aucs) * 0.98, max(best_aucs) * 1.03)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "bar_best_auc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: curvas ROC superpuestas
# ================================================================
def plot_roc_curves(experiments: list, results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)

    any_data = False
    for exp in experiments:
        y_true, y_prob = load_roc_data(exp, results_dir)
        if y_true is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=get_color(exp), linewidth=1.8,
                label=f"{get_label(exp)} (AUC={roc_auc:.4f})")
        any_data = True

    if not any_data:
        print("[WARN] Sin val_y_prob.npy disponibles para ROC. Genera primero eval_val.")
        plt.close(fig)
        return

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Curvas ROC — comparativa", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: curvas Precision-Recall superpuestas
# ================================================================
def plot_pr_curves(experiments: list, results_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    any_data = False
    for exp in experiments:
        y_true, y_prob = load_roc_data(exp, results_dir)
        if y_true is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=get_color(exp), linewidth=1.8,
                label=f"{get_label(exp)} (PR-AUC={pr_auc:.4f})")
        any_data = True

    if not any_data:
        print("[WARN] Sin val_y_prob.npy disponibles para PR. Genera primero eval_val.")
        plt.close(fig)
        return

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Curvas Precision-Recall — comparativa", fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.7)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = output_dir / "pr_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# Gráfica: heatmap resumen
# ================================================================
def plot_heatmap_summary(all_experiments: list, results_dir: Path, output_dir: Path) -> None:
    rows = []
    for exp in all_experiments:
        hist = load_history(exp, results_dir)
        if hist is None:
            continue
        col = "val_auc" if "val_auc" in hist.columns else "val_auroc"
        if col not in hist.columns:
            continue
        best_auc = hist[col].max()
        best_ep  = int(hist[col].idxmax())
        rows.append({
            "experimento": get_label(exp),
            "best_val_AUC": round(best_auc, 4),
            "época_convergencia": best_ep,
        })

    if not rows:
        print("[WARN] Sin datos para heatmap")
        return

    df = pd.DataFrame(rows).set_index("experimento")

    fig, ax = plt.subplots(figsize=(6, max(3, len(rows) * 0.6 + 1)))
    im = ax.imshow(df[["best_val_AUC"]].values, cmap="YlGn",
                   aspect="auto", vmin=0.70, vmax=0.85)

    ax.set_xticks([0])
    ax.set_xticklabels(["Best Val AUC"], fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index.tolist(), fontsize=9)

    for i, (auc_val, ep) in enumerate(zip(df["best_val_AUC"], df["época_convergencia"])):
        ax.text(0, i, f"{auc_val:.4f}\n(ep {ep})",
                ha="center", va="center", fontsize=9, color="black")

    plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    ax.set_title("Resumen de experimentos", fontsize=11)

    out = output_dir / "heatmap_summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1,
                        help="Fase a comparar: 1, 2, 3 o 4")
    args = parser.parse_args()

    experiments = PHASE_MAP.get(args.phase, [])
    if not experiments:
        print(f"[ERROR] No hay experimentos definidos para fase {args.phase}.")
        return

    results_dir = Path(TRAIN_RESULTS_DIR)
    output_dir  = Path(GRAPHS_OUTPUT_DIR) / f"phase{args.phase}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Fase {args.phase} — {len(experiments)} experimentos")
    print(f"[INFO] Baseline: {BASELINE_EXPERIMENT}")
    print(f"[INFO] Output: {output_dir.resolve()}\n")

    if PLOT_VAL_AUC_CURVES:
        plot_val_auc_curves(experiments, results_dir, output_dir)
    if PLOT_VAL_LOSS_CURVES:
        plot_val_loss_curves(experiments, results_dir, output_dir)
    if PLOT_TRAIN_VAL_AUC:
        plot_train_val_auc(experiments, results_dir, output_dir)
    if PLOT_TRAIN_VAL_LOSS:
        plot_train_val_loss(experiments, results_dir, output_dir)
    if PLOT_BAR_BEST_AUC:
        plot_bar_best_auc(experiments, results_dir, output_dir)
    if PLOT_ROC_CURVES:
        plot_roc_curves(experiments, results_dir, output_dir)
    if PLOT_PR_CURVES:
        plot_pr_curves(experiments, results_dir, output_dir)
    if PLOT_HEATMAP_SUMMARY:
        # heatmap usa todos los experimentos de todas las fases disponibles
        all_exps = [e for phase in PHASE_MAP.values() for e in phase]
        plot_heatmap_summary(all_exps, results_dir, output_dir)

    print("\n[DONE] Comparativa completada.")


if __name__ == "__main__":
    main()