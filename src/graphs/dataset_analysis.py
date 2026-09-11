#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_analysis.py
--------------------
Genera gráficas de análisis del dataset.

Gráficas generadas:
  - grafica_11_ratio_por_sujeto_STNF+STLK.png   : ratio positivos por sujeto, con razón de descarte
  - grafica_12_distribucion_ratio_STNF+STLK.png : histograma comparativo antes/después normalización

Uso:
    python -m src.graphs.dataset_analysis

Requiere:
  - segments_*.npz  (antes de normalización)
  - normalised_*.npz (después de normalización)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.graphs.config import (
    NORMALISED_DATA_DIR,
    SEGMENTS_DATA_DIR,
    GRAPHS_OUTPUT_DIR,
    MAX_POSITIVE_RATIO,
    MIN_TST_HOURS,
    DPI,
)


# ================================================================
# Carga rápida: solo campos escalares, sin airflow/spo2
# ================================================================
def load_stats_from_npz(npz_path: Path) -> dict | None:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            n_total = int(data["n_segments_total"]) if "n_segments_total" in data \
                      else int(data["labels"].size)
            n_pos   = int(data["n_segments_pos"]) if "n_segments_pos" in data \
                      else int(data["labels"].sum())
            pct     = float(data["pct_positive"]) if "pct_positive" in data \
                      else (n_pos / n_total if n_total > 0 else 0.0)
        return {
            "id": npz_path.stem,
            "n_total": n_total,
            "n_pos": n_pos,
            "pct_positive": pct,
        }
    except Exception as e:
        print(f"[WARN] No se pudo leer {npz_path.name}: {e}")
        return None


# ================================================================
# Clasificar razón de descarte
# ================================================================
def clasificar_descarte(stats: dict) -> str:
    tst_hours = (stats["n_total"] * 1.0) / 3600.0
    if tst_hours < MIN_TST_HOURS:
        return "TST insuficiente"
    if stats["pct_positive"] > MAX_POSITIVE_RATIO:
        return "Ratio positivos excesivo"
    return "valido"


# ================================================================
# Gráfica 11 — ratio por sujeto con razón de descarte
# ================================================================
def plot_ratio_por_sujeto(stats_before: list, output_dir: Path) -> None:
    datos = [(s["pct_positive"], clasificar_descarte(s)) for s in stats_before]
    datos.sort(key=lambda x: x[0])

    ratios  = np.array([d[0] for d in datos])
    razones = [d[1] for d in datos]
    x = np.arange(len(ratios))

    color_map = {
        "valido":                   "#378ADD",
        "Ratio positivos excesivo": "#E24B4A",
        "TST insuficiente":         "#BA7517",
    }
    colores = [color_map[r] for r in razones]

    n_validos = razones.count("valido")
    n_ratio   = razones.count("Ratio positivos excesivo")
    n_tst     = razones.count("TST insuficiente")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(x, ratios, c=colores, s=12, alpha=0.85, linewidths=0)
    ax.axhline(MAX_POSITIVE_RATIO, color="#444", linewidth=1.2, linestyle="--")

    patches = [
        mpatches.Patch(color="#378ADD", label=f"Válidos ({n_validos})"),
        mpatches.Patch(color="#E24B4A", label=f"Descartados: ratio excesivo ({n_ratio})"),
        mpatches.Patch(color="#BA7517", label=f"Descartados: TST insuficiente ({n_tst})"),
        plt.Line2D([0], [0], color="#444", linestyle="--",
                   label=f"Límite ratio = {MAX_POSITIVE_RATIO}"),
    ]
    ax.legend(handles=patches, fontsize=8.5, framealpha=0.7)
    ax.set_xlabel("Sujeto (ordenado por ratio ascendente)", fontsize=10)
    ax.set_ylabel("Ratio de ventanas positivas", fontsize=10)
    ax.set_title(
        f"Distribución del ratio de positivos por sujeto\n"
        f"(total={len(ratios)}, válidos={n_validos}, "
        f"descartados={n_ratio + n_tst})",
        fontsize=11
    )
    ax.set_xlim(-1, len(ratios) + 1)
    ax.set_ylim(0, min(ratios.max() * 1.15, 1.05))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = output_dir / "grafica_11_ratio_por_sujeto.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardada: {out_path}")


# ================================================================
# Gráfica 12 — histograma antes/después
# ================================================================
def plot_distribucion_ratio(stats_before: list, stats_after: list,
                             output_dir: Path) -> None:
    ratios_before = np.array([s["pct_positive"] for s in stats_before])
    ratios_after  = np.array([s["pct_positive"] for s in stats_after])

    bin_edges = np.linspace(0, max(ratios_before.max(), 0.7), 20)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, ratios, color, titulo in [
        (axes[0], ratios_before, "#85B7EB", "Antes de normalización"),
        (axes[1], ratios_after,  "#1D9E75", "Después de normalización"),
    ]:
        ax.hist(ratios, bins=bin_edges, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.9)
        ax.axvline(MAX_POSITIVE_RATIO, color="#BA7517", linewidth=1.2,
                   linestyle="--", label=f"Límite = {MAX_POSITIVE_RATIO}")
        media = ratios.mean()
        ax.axvline(media, color="#444", linewidth=1.0, linestyle=":",
                   label=f"Media = {media:.3f}")
        ax.set_title(f"{titulo}\n(n = {len(ratios)} sujetos)", fontsize=10)
        ax.set_xlabel("Ratio de ventanas positivas", fontsize=9)
        ax.set_ylabel("Número de sujetos", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Efecto de la normalización sobre el ratio de positivos", fontsize=11)
    out_path = output_dir / "grafica_12_distribucion_ratio.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardada: {out_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    normalised_dir = Path(NORMALISED_DATA_DIR)
    segments_dir   = Path(SEGMENTS_DATA_DIR)
    output_dir     = Path(GRAPHS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- segments (antes) ---
    segments_files = sorted(segments_dir.glob("segments_*.npz"))
    if not segments_files:
        raise FileNotFoundError(f"No hay segments_*.npz en {segments_dir.resolve()}")

    print(f"[INFO] Leyendo {len(segments_files)} sujetos pre-normalización...")
    stats_before = [load_stats_from_npz(f) for f in segments_files]
    stats_before = [s for s in stats_before if s is not None]

    # --- normalised (después) ---
    normalised_files = sorted(normalised_dir.glob("normalised_*.npz"))
    if not normalised_files:
        raise FileNotFoundError(f"No hay normalised_*.npz en {normalised_dir.resolve()}")

    print(f"[INFO] Leyendo {len(normalised_files)} sujetos normalizados...")
    stats_after = [load_stats_from_npz(f) for f in normalised_files]
    stats_after = [s for s in stats_after if s is not None]

    print(f"\n[INFO] Sujetos antes  : {len(stats_before)}")
    print(f"[INFO] Sujetos después: {len(stats_after)}")
    print(f"[INFO] Descartados    : {len(stats_before) - len(stats_after)}")

    plot_ratio_por_sujeto(stats_before, output_dir)
    plot_distribucion_ratio(stats_before, stats_after, output_dir)

    print("\n[DONE] Análisis del dataset completado.")


if __name__ == "__main__":
    main()