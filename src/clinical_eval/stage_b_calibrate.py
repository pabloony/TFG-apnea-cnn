#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stage_b_calibrate.py
--------------------
FASE 9 · ETAPA B — Umbral-para-AHI sobre y_prob, con split calibración/test.

Idea
----
Con el contador ya validado en Etapa A (min_ventanas=1 fijo), ahora aplicamos el
contador a las PREDICCIONES del modelo (y_prob). Como y_prob es continua, hay que
binarizarla con un UMBRAL — y como esa binarización crea huecos que y_true no
tenía, la TOLERANCIA se re-calibra aquí junto con el umbral (barrido 2D).

Para evitar optimismo de calibración (calibrar y reportar sobre los mismos
sujetos), se hace un SPLIT estratificado por severidad AASM:
  - CALIBRACIÓN (~50%): se elige el mejor par (umbral, tolerancia) minimizando
    el error de AHI vs el AHI real.
  - TEST (~50%): con el par ya fijado, se reporta Bland-Altman + matriz de
    severidad AASM sobre sujetos que NO participaron en la calibración.

Denominador del AHI
-------------------
AHI_estimado = n_eventos / (sleep_time / 3600), usando el sleep_time REAL del
ground truth (TST verdadero). Esto aísla el error del CONTEO del error del TST.
(Pendiente futuro: comparar el TST de mask_sleep contra el real.)

Sesgo conocido (de Etapa A): infracuenta estructural en apnea severa por fusión
de eventos contiguos (resolución de ventana W=15s). Ningún umbral lo corrige del
todo; la Etapa B lo CUANTIFICA limpiamente sobre el test set.

Uso
---
    python -m src.clinical_eval.stage_b_calibrate
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
    FASE9_STAGE_B_DIR,
    WINDOW_S,
    STRIDE_S,
    EXCLUDED_SUBJECTS,
    STAGE_B_MIN_VENTANAS,
    STAGE_B_UMBRALES,
    STAGE_B_TOLERANCIAS,
    STAGE_B_SPLIT_SEED,
    STAGE_B_CALIB_FRAC,
    STAGE_B_OUTPUT_SUFFIX,
    STAGE_B_MODE,
    STAGE_B_FIXED_UMBRAL,
    STAGE_B_FIXED_TOL,
    CLINICAL_COHORT,
    AASM_SEVERITY_BINS,
    AASM_SEVERITY_LABELS,
)
from .event_counter import count_events


# ----------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------
def _load_predictions(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        usecols=["subject_id", "win_idx", "y_prob", "y_true"],
        dtype={"subject_id": str, "win_idx": np.int32,
               "y_prob": np.float32, "y_true": np.int8},
    )


def _load_ground_truth(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    for c in ["n_obs", "n_cen", "n_mix", "n_hyp"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce").fillna(0).astype(int)
    gt["sleep_time"] = pd.to_numeric(gt["sleep_time"], errors="coerce")
    gt["ahi"] = pd.to_numeric(gt["ahi"], errors="coerce")
    return gt.set_index("s_code")


def _severity(ahi: float) -> str:
    for i in range(len(AASM_SEVERITY_LABELS)):
        if AASM_SEVERITY_BINS[i] <= ahi < AASM_SEVERITY_BINS[i + 1]:
            return AASM_SEVERITY_LABELS[i]
    return AASM_SEVERITY_LABELS[-1]


def _prob_sequences(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    """{subject_id: array y_prob ordenado por win_idx}, excluyendo EXCLUDED_SUBJECTS."""
    seqs = {}
    for sid, g in pred.groupby("subject_id"):
        if sid in EXCLUDED_SUBJECTS:
            continue
        seqs[sid] = g.sort_values("win_idx")["y_prob"].to_numpy(dtype=np.float32)
    return seqs


# ----------------------------------------------------------------------
# Split estratificado por severidad
# ----------------------------------------------------------------------
def _stratified_split(subjects: list[str], gt: pd.DataFrame,
                      calib_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """
    Reparte sujetos en (calibración, test) equilibrando categorías AASM.
    Dentro de cada categoría, baraja con semilla fija y asigna calib_frac a calib.
    """
    rng = np.random.default_rng(seed)
    by_sev = {lab: [] for lab in AASM_SEVERITY_LABELS}
    for s in subjects:
        by_sev[_severity(gt.loc[s, "ahi"])].append(s)

    calib, test = [], []
    for lab, group in by_sev.items():
        group = list(group)
        rng.shuffle(group)
        n_calib = round(len(group) * calib_frac)
        calib.extend(group[:n_calib])
        test.extend(group[n_calib:])
    return sorted(calib), sorted(test)


# ----------------------------------------------------------------------
# AHI estimado para un sujeto dado (umbral, tolerancia)
# ----------------------------------------------------------------------
def _estimate_ahi(prob_seq: np.ndarray, tst_h: float,
                  umbral: float, tol: int, min_vent: int) -> float:
    binary = (prob_seq >= umbral).astype(np.int8)
    n_events = count_events(binary, tol, min_vent)
    return n_events / tst_h if tst_h > 0 else np.nan


# ----------------------------------------------------------------------
# Barrido de calibración
# ----------------------------------------------------------------------
def _calibrate(seqs, gt, calib_ids, umbrales, tolerancias, min_vent):
    """
    Barre (umbral × tolerancia) sobre los sujetos de calibración.
    Devuelve DataFrame con métricas por par y el mejor par (min MAE de AHI).
    """
    tst_h = {s: gt.loc[s, "sleep_time"] / 3600.0 for s in calib_ids}
    ahi_real = np.array([gt.loc[s, "ahi"] for s in calib_ids], dtype=float)

    rows = []
    for u in umbrales:
        for tol in tolerancias:
            ahi_est = np.array(
                [_estimate_ahi(seqs[s], tst_h[s], u, tol, min_vent) for s in calib_ids],
                dtype=float,
            )
            err = ahi_est - ahi_real
            rows.append({
                "umbral": u, "tolerancia": tol,
                "MAE": np.abs(err).mean(),
                "sesgo": err.mean(),
                "corr": np.corrcoef(ahi_est, ahi_real)[0, 1] if len(err) > 1 else np.nan,
            })
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["MAE"].idxmin()]
    return sweep, float(best["umbral"]), int(best["tolerancia"])


# ----------------------------------------------------------------------
# Reporte sobre test
# ----------------------------------------------------------------------
def _evaluate_test(seqs, gt, test_ids, umbral, tol, min_vent):
    rows = []
    for s in test_ids:
        tst_h = gt.loc[s, "sleep_time"] / 3600.0
        ahi_r = gt.loc[s, "ahi"]
        ahi_e = _estimate_ahi(seqs[s], tst_h, umbral, tol, min_vent)
        rows.append({
            "subject_id": s,
            "ahi_real": ahi_r, "ahi_est": ahi_e,
            "sev_real": _severity(ahi_r), "sev_est": _severity(ahi_e),
            "diff": ahi_e - ahi_r,
        })
    return pd.DataFrame(rows)


def _plot_bland_altman(df, out_path, umbral, tol):
    mean = (df["ahi_real"] + df["ahi_est"]) / 2
    diff = df["ahi_est"] - df["ahi_real"]
    bias = diff.mean()
    sd = diff.std()
    loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mean, diff, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(bias, color="b", ls="-", lw=1.2, label=f"sesgo={bias:+.2f}")
    ax.axhline(loa_hi, color="r", ls="--", lw=1, label=f"+1.96 SD={loa_hi:+.2f}")
    ax.axhline(loa_lo, color="r", ls="--", lw=1, label=f"-1.96 SD={loa_lo:+.2f}")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Media (AHI real, AHI estimado)")
    ax.set_ylabel("AHI estimado − AHI real")
    ax.set_title(f"Bland-Altman (TEST) — umbral={umbral}, tol={tol}s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return bias, sd, loa_lo, loa_hi


def _plot_confusion(df, out_path, umbral, tol):
    labels = AASM_SEVERITY_LABELS
    idx = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for _, r in df.iterrows():
        M[idx[r["sev_real"]], idx[r["sev_est"]]] += 1

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Severidad estimada")
    ax.set_ylabel("Severidad real")
    ax.set_title(f"Matriz severidad AASM (TEST) — umbral={umbral}, tol={tol}s")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    acc = np.trace(M) / M.sum() if M.sum() else np.nan
    return M, acc


def _plot_calib_heatmap(sweep, out_path):
    piv = sweep.pivot(index="tolerancia", columns="umbral", values="MAE")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("Umbral"); ax.set_ylabel("Tolerancia (s)")
    ax.set_title("Calibración — MAE de AHI por par (menor = mejor)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i,j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    out_dir = Path(FASE9_STAGE_B_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("[FASE 9 · ETAPA B] Umbral-para-AHI + reporte clínico")
    print("=" * 64)

    print(f"[INFO] Cohorte: {CLINICAL_COHORT} | Modo Etapa B: {STAGE_B_MODE}")

    pred = _load_predictions(Path(CLINICAL_PROBS_CSV))
    gt = _load_ground_truth(Path(AHI_GT_CSV))
    seqs = _prob_sequences(pred)

    subjects = [s for s in seqs if s in gt.index and not np.isnan(gt.loc[s, "ahi"])]
    print(f"[INFO] Sujetos válidos: {len(subjects)} (excluidos: {list(EXCLUDED_SUBJECTS)})")

    sfx = STAGE_B_OUTPUT_SUFFIX

    if STAGE_B_MODE == "fixed":
        # ---- MODO FIXED: sin split, sin barrido. Umbral/tol ya fijados (de STNF).
        # Responde: ¿generaliza el estimador tal cual a esta cohorte?
        best_u, best_tol = STAGE_B_FIXED_UMBRAL, STAGE_B_FIXED_TOL
        test_ids = sorted(subjects)          # TODOS los sujetos son "test"
        print(f"[FIXED] Umbral={best_u}, tol={best_tol}s aplicados a TODOS ({len(test_ids)} sujetos).")
        dist = pd.Series([_severity(gt.loc[s, "ahi"]) for s in test_ids]).value_counts()
        print("  Severidad: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
        print("-" * 64)

    elif STAGE_B_MODE == "recalib":
        # ---- MODO RECALIB: split estratificado + barrido (como en STNF).
        calib_ids, test_ids = _stratified_split(
            subjects, gt, STAGE_B_CALIB_FRAC, STAGE_B_SPLIT_SEED)
        print(f"[INFO] Calibración: {len(calib_ids)} | Test: {len(test_ids)}")
        for name, ids in [("CALIB", calib_ids), ("TEST", test_ids)]:
            dist = pd.Series([_severity(gt.loc[s, "ahi"]) for s in ids]).value_counts()
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in dist.items()))
        print("-" * 64)

        sweep, best_u, best_tol = _calibrate(
            seqs, gt, calib_ids, STAGE_B_UMBRALES, STAGE_B_TOLERANCIAS, STAGE_B_MIN_VENTANAS)
        sweep.sort_values("MAE").to_csv(out_dir / f"stage_b_calib_sweep{sfx}.csv", index=False)
        _plot_calib_heatmap(sweep, out_dir / f"stage_b_calib_heatmap{sfx}.png")

        print(f"[CALIB] Mejor par: umbral={best_u}, tolerancia={best_tol}s")
        print(f"        (MAE calib={sweep['MAE'].min():.2f})")
        print("[CALIB] Top 5 pares:")
        print(sweep.sort_values("MAE").head(5).to_string(index=False))
        print("-" * 64)
    else:
        raise ValueError(f"STAGE_B_MODE no reconocido: {STAGE_B_MODE}")

    # Test (común a ambos modos)
    test_df = _evaluate_test(seqs, gt, test_ids, best_u, best_tol, STAGE_B_MIN_VENTANAS)
    test_df.to_csv(out_dir / f"stage_b_test_per_subject{sfx}.csv", index=False)

    bias, sd, loa_lo, loa_hi = _plot_bland_altman(
        test_df, out_dir / f"stage_b_bland_altman{sfx}.png", best_u, best_tol)
    M, acc = _plot_confusion(
        test_df, out_dir / f"stage_b_confusion_severity{sfx}.png", best_u, best_tol)

    test_mae = test_df["diff"].abs().mean()
    test_corr = test_df["ahi_est"].corr(test_df["ahi_real"])

    print("[TEST] Resultados sobre sujetos no vistos en calibración:")
    print(f"       MAE AHI      : {test_mae:.2f}")
    print(f"       Sesgo (B-A)  : {bias:+.2f}  (SD={sd:.2f})")
    print(f"       LoA          : [{loa_lo:+.2f}, {loa_hi:+.2f}]")
    print(f"       Corr AHI     : {test_corr:.4f}")
    print(f"       Acc severidad: {acc*100:.1f}%")
    print(f"       Matriz severidad (filas=real, cols=est):")
    print(f"       {AASM_SEVERITY_LABELS}")
    print(M)
    print("-" * 64)
    print(f"[OK] Heatmap calib  : {out_dir / f'stage_b_calib_heatmap{sfx}.png'}")
    print(f"[OK] Bland-Altman   : {out_dir / f'stage_b_bland_altman{sfx}.png'}")
    print(f"[OK] Matriz severidad: {out_dir / f'stage_b_confusion_severity{sfx}.png'}")
    print(f"[OK] CSV por sujeto : {out_dir / f'stage_b_test_per_subject{sfx}.csv'}")
    print("[DONE]")


if __name__ == "__main__":
    main()