#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_a_validate.py
-------------------
FASE 9 · ETAPA A — Validar el contador de eventos SIN el modelo de por medio.

Idea
----
La columna y_true del CSV son las etiquetas VERDADERAS por-ventana (0/1),
derivadas de mask_events. Si aplicamos el contador a y_true, deberíamos
reproducir el número real de eventos de cada sujeto (que conocemos del CSV de
ground truth). Esto AÍSLA el error del contador del error del modelo: si aquí
no cuadra, el problema es la lógica de fusión/filtro, no la CNN.

Contra qué se compara (OJO, punto clave)
----------------------------------------
mask_events marca CUATRO tipos: obstructivas + centrales + mixtas + hipopneas.
Por tanto y_true contiene esos cuatro tipos. El objetivo correcto de validación
del contador es el conteo de LOS CUATRO tipos:

    target_mask_events = n_obs + n_cen + n_mix + n_hyp

NOTA: esto NO es lo mismo que el AHI del dataset, que EXCLUYE las centrales
(AHI = n_obs + n_mix + n_hyp). La exclusión de centrales solo importa en la
Etapa B (AHI estimado vs AHI real). En la Etapa A validamos que el contador
reconstruye lo que hay DENTRO de mask_events, y ahí las centrales SÍ están.
Se calculan ambos objetivos para diagnóstico, pero el primario es el de 4 tipos.

Qué hace
--------
1. Lee val_window_probs.csv (subject_id, win_idx, y_prob, y_true).
2. Lee ahi_labels.csv (ground truth; sep=';', decimal=',').
3. Para cada sujeto de validación, reconstruye la secuencia y_true ordenada por
   win_idx y aplica el contador barriendo (tolerancia × min_ventanas).
4. Compara el conteo del contador con el conteo real (4 tipos) por sujeto.
5. Genera:
   - Heatmap de MAE (error absoluto medio de nº de eventos) sobre la rejilla.
   - Heatmap de sesgo medio (con signo) sobre la rejilla.
   - Scatter del mejor combo: eventos contados vs reales, por sujeto.
   - CSV resumen del barrido.

Uso
---
    python -m src.clinical_eval.stage_a_validate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Config propio de clinical_eval (que a su vez hereda rutas de design/config)
from .config import (
    CLINICAL_PROBS_CSV,
    AHI_GT_CSV,
    FASE9_STAGE_A_DIR,
    STAGE_A_TOLERANCIAS,
    STAGE_A_MIN_VENTANAS,
    WINDOW_S,
    STRIDE_S,
    EXCLUDED_SUBJECTS,
    STAGE_A_OUTPUT_SUFFIX,
)

from .event_counter import count_events


# ----------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------
def _load_predictions(csv_path: Path) -> pd.DataFrame:
    """Lee val_window_probs.csv. Solo necesitamos subject_id, win_idx, y_true."""
    df = pd.read_csv(
        csv_path,
        usecols=["subject_id", "win_idx", "y_true"],
        dtype={"subject_id": str, "win_idx": np.int32, "y_true": np.int8},
    )
    return df


def _load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """
    Lee ahi_labels.csv (formato europeo: sep=';', decimal=',').
    Devuelve un DataFrame indexado por s_code con los conteos y el TST.
    """
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    for c in ["n_obs", "n_cen", "n_mix", "n_hyp"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce").fillna(0).astype(int)
    gt["sleep_time"] = pd.to_numeric(gt["sleep_time"], errors="coerce")
    gt["ahi"] = pd.to_numeric(gt["ahi"], errors="coerce")

    # Objetivo PRIMARIO: lo que mask_events contiene (4 tipos)
    gt["target_mask_events"] = gt["n_obs"] + gt["n_cen"] + gt["n_mix"] + gt["n_hyp"]
    # Objetivo secundario (referencia): definición de AHI del dataset (sin centrales)
    gt["target_ahi_def"] = gt["n_obs"] + gt["n_mix"] + gt["n_hyp"]

    gt = gt.set_index("s_code")
    return gt


# ----------------------------------------------------------------------
# Núcleo Etapa A
# ----------------------------------------------------------------------
def _subject_sequences(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Devuelve {subject_id: array y_true ordenado por win_idx}.
    Excluye los sujetos de EXCLUDED_SUBJECTS (calidad de datos).
    """
    seqs = {}
    n_excl = 0
    for sid, g in pred.groupby("subject_id"):
        if sid in EXCLUDED_SUBJECTS:
            n_excl += 1
            continue
        g = g.sort_values("win_idx")
        seqs[sid] = g["y_true"].to_numpy(dtype=np.int8)
    if n_excl:
        print(f"[EXCL] {n_excl} sujetos excluidos por calidad de datos: "
              f"{list(EXCLUDED_SUBJECTS.keys())}")
    return seqs


def _sweep(seqs: dict[str, np.ndarray],
           gt: pd.DataFrame,
           tolerancias: list[int],
           min_ventanas: list[int]) -> pd.DataFrame:
    """
    Barre (tolerancia × min_ventanas). Para cada combo calcula, sujeto a sujeto,
    el nº de eventos del contador sobre y_true y lo compara con el conteo real
    (4 tipos). Devuelve un DataFrame con una fila por combo y métricas agregadas.
    """
    rows = []
    subjects = [s for s in seqs.keys() if s in gt.index]
    missing = [s for s in seqs.keys() if s not in gt.index]
    if missing:
        print(f"[WARN] {len(missing)} sujetos sin ground truth (ignorados): {missing[:5]}...")

    real = np.array([gt.loc[s, "target_mask_events"] for s in subjects], dtype=float)

    for tol in tolerancias:
        for mw in min_ventanas:
            est = np.array(
                [count_events(seqs[s], tol, mw) for s in subjects],
                dtype=float,
            )
            err = est - real                     # con signo (sesgo)
            abs_err = np.abs(err)
            rows.append({
                "tolerancia_s": tol,
                "min_ventanas": mw,
                "MAE": abs_err.mean(),
                "sesgo_medio": err.mean(),        # >0 sobrecuenta, <0 infracuenta
                "MAE_rel_%": 100.0 * abs_err.sum() / real.sum() if real.sum() else np.nan,
                "corr": np.corrcoef(est, real)[0, 1] if len(est) > 1 else np.nan,
            })
    return pd.DataFrame(rows), subjects, real


# ----------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------
def _heatmap(sweep_df: pd.DataFrame, value: str, title: str,
             out_path: Path, cmap: str, center_zero: bool = False) -> None:
    piv = sweep_df.pivot(index="min_ventanas", columns="tolerancia_s", values=value)
    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = np.nanmax(np.abs(piv.values)) if center_zero else None
    vmin = -vmax if center_zero else None
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Tolerancia de hueco (s)")
    ax.set_ylabel("Mínimo de ventanas consecutivas")
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9,
                    color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _scatter_best(seqs, gt, subjects, real, best_tol, best_mw, out_path: Path) -> None:
    est = np.array([count_events(seqs[s], best_tol, best_mw) for s in subjects],
                   dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(real, est, alpha=0.6, edgecolor="k", linewidth=0.3)
    lim = max(real.max(), est.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="identidad (y=x)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Eventos reales (obs+cen+mix+hyp)")
    ax.set_ylabel("Eventos contados sobre y_true")
    ax.set_title(f"Etapa A — mejor combo: tol={best_tol}s, min_vent={best_mw}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    out_dir = Path(FASE9_STAGE_A_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[FASE 9 · ETAPA A] Validación del contador sobre y_true")
    print("=" * 60)
    print(f"[INFO] Predicciones : {CLINICAL_PROBS_CSV}")
    print(f"[INFO] Ground truth : {AHI_GT_CSV}")
    print(f"[INFO] Salida       : {out_dir.resolve()}")
    print(f"[INFO] W={WINDOW_S}s, stride={STRIDE_S}s")
    print(f"[INFO] Barrido tol   : {STAGE_A_TOLERANCIAS}")
    print(f"[INFO] Barrido minvt : {STAGE_A_MIN_VENTANAS}")
    print("-" * 60)

    pred = _load_predictions(Path(CLINICAL_PROBS_CSV))
    gt = _load_ground_truth(Path(AHI_GT_CSV))
    seqs = _subject_sequences(pred)
    print(f"[INFO] Sujetos en predicciones: {len(seqs)}")

    sweep_df, subjects, real = _sweep(seqs, gt, STAGE_A_TOLERANCIAS, STAGE_A_MIN_VENTANAS)
    print(f"[INFO] Sujetos validados con ground truth: {len(subjects)}")
    print(f"[INFO] Total eventos reales (4 tipos): {int(real.sum())}")
    print("-" * 60)

    # Mejor combo = menor MAE
    best = sweep_df.loc[sweep_df["MAE"].idxmin()]
    best_tol = int(best["tolerancia_s"])
    best_mw = int(best["min_ventanas"])

    # Sufijo para no sobrescribir los outputs de pasadas anteriores
    sfx = STAGE_A_OUTPUT_SUFFIX

    # Guardar resumen del barrido
    sweep_csv = out_dir / f"stage_a_sweep{sfx}.csv"
    sweep_df.sort_values("MAE").to_csv(sweep_csv, index=False)

    # Gráficas
    heatmap_mae_png   = out_dir / f"stage_a_heatmap_mae{sfx}.png"
    heatmap_sesgo_png = out_dir / f"stage_a_heatmap_sesgo{sfx}.png"
    scatter_png       = out_dir / f"stage_a_scatter_best{sfx}.png"

    _heatmap(sweep_df, "MAE",
             "Etapa A (limpia) — MAE (nº eventos) por combo",
             heatmap_mae_png, cmap="viridis_r")
    _heatmap(sweep_df, "sesgo_medio",
             "Etapa A (limpia) — sesgo medio (con signo)",
             heatmap_sesgo_png, cmap="RdBu_r", center_zero=True)
    _scatter_best(seqs, gt, subjects, real, best_tol, best_mw, scatter_png)

    # Resumen por pantalla
    print("[RESULTADO] Top 5 combos por MAE:")
    print(sweep_df.sort_values("MAE").head(5).to_string(index=False))
    print("-" * 60)
    print(f"[MEJOR] tol={best_tol}s, min_ventanas={best_mw}")
    print(f"        MAE={best['MAE']:.2f} eventos/sujeto | "
          f"sesgo={best['sesgo_medio']:+.2f} | "
          f"MAE_rel={best['MAE_rel_%']:.1f}% | corr={best['corr']:.4f}")
    print("-" * 60)
    print(f"[OK] Resumen barrido : {sweep_csv}")
    print(f"[OK] Heatmap MAE     : {heatmap_mae_png}")
    print(f"[OK] Heatmap sesgo   : {heatmap_sesgo_png}")
    print(f"[OK] Scatter mejor   : {scatter_png}")
    print("[DONE]")


if __name__ == "__main__":
    main()