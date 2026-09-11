#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_b_validate_full.py

FASE 9 · ETAPA B (variante) — Igual que stage_b_calibrate.py, pero el
GROUND TRUTH no es la columna `ahi` del CSV, sino el AHI recalculado "a mano":

    AHI_full = (n_obs + n_cen + n_mix + n_hyp) / (sleep_time / 3600)

Motivo
------
La columna `ahi` del dataset sigue la definición clínica que EXCLUYE las apneas
centrales (n_obs + n_mix + n_hyp). Sin embargo, `mask_events` — y por tanto las
etiquetas con las que se entrenó la CNN — SÍ incluye las centrales. Comparar el
AHI estimado contra la columna `ahi` mezcla dos fuentes de error:
    (a) error del modelo,
    (b) desajuste de definición (el modelo puede detectar eventos que el AHI
        tabulado no cuenta).
Este script aísla (a) usando el objetivo coherente con lo que el modelo aprendió.

Se reportan AMBOS AHI por sujeto (ahi_full y ahi_csv) para poder cuantificar
cuánto del sesgo venía de la definición y cuánto del modelo.

Uso
---
    python -m src.clinical_eval.stage_b_validate_full

Respeta la config vigente: cohorte activa, STAGE_B_MODE ("fixed"/"recalib"),
umbral/tolerancia fijos, exclusiones y sufijos de salida.
"""

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
    EXCLUDED_SUBJECTS,
    STAGE_B_MIN_VENTANAS,
    STAGE_B_UMBRALES,
    STAGE_B_TOLERANCIAS,
    STAGE_B_SPLIT_SEED,
    STAGE_B_CALIB_FRAC,
    STAGE_B_MODE,
    STAGE_B_FIXED_UMBRAL,
    STAGE_B_FIXED_TOL,
    CLINICAL_COHORT,
    AASM_SEVERITY_BINS,
    AASM_SEVERITY_LABELS,
)
from .event_counter import count_events

# Sufijo propio para NO pisar las salidas de stage_b_calibrate.py
OUTPUT_SUFFIX = "_ahifull"


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
    """
    Lee ahi_labels.csv (sep=';', decimal=',') y añade:
        n_eventos_full : n_obs + n_cen + n_mix + n_hyp   (lo que hay en mask_events)
        n_eventos_csv  : n_obs + n_mix + n_hyp           (definición del dataset)
        ahi_full       : n_eventos_full / (sleep_time/3600)   <- objetivo PRIMARIO aquí
        ahi_csv        : columna `ahi` tal cual (referencia)
    """
    gt = pd.read_csv(csv_path, sep=";", decimal=",")
    for c in ["n_obs", "n_cen", "n_mix", "n_hyp"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce").fillna(0).astype(int)
    gt["sleep_time"] = pd.to_numeric(gt["sleep_time"], errors="coerce")
    gt["ahi_csv"] = pd.to_numeric(gt["ahi"], errors="coerce")

    gt["n_eventos_full"] = gt["n_obs"] + gt["n_cen"] + gt["n_mix"] + gt["n_hyp"]
    gt["n_eventos_csv"] = gt["n_obs"] + gt["n_mix"] + gt["n_hyp"]

    tst_h = gt["sleep_time"] / 3600.0
    gt["ahi_full"] = np.where(tst_h > 0, gt["n_eventos_full"] / tst_h, np.nan)

    return gt.set_index("s_code")


def _severity(ahi: float) -> str:
    for i in range(len(AASM_SEVERITY_LABELS)):
        if AASM_SEVERITY_BINS[i] <= ahi < AASM_SEVERITY_BINS[i + 1]:
            return AASM_SEVERITY_LABELS[i]
    return AASM_SEVERITY_LABELS[-1]


def _prob_sequences(pred: pd.DataFrame) -> dict[str, np.ndarray]:
    seqs = {}
    for sid, g in pred.groupby("subject_id"):
        if sid in EXCLUDED_SUBJECTS:
            continue
        seqs[sid] = g.sort_values("win_idx")["y_prob"].to_numpy(dtype=np.float32)
    return seqs


# ----------------------------------------------------------------------
# Split estratificado por severidad (según ahi_full)
# ----------------------------------------------------------------------
def _stratified_split(subjects, gt, calib_frac, seed):
    rng = np.random.default_rng(seed)
    by_sev = {lab: [] for lab in AASM_SEVERITY_LABELS}
    for s in subjects:
        by_sev[_severity(gt.loc[s, "ahi_full"])].append(s)

    calib, test = [], []
    for lab, group in by_sev.items():
        group = list(group)
        rng.shuffle(group)
        n_calib = round(len(group) * calib_frac)
        calib.extend(group[:n_calib])
        test.extend(group[n_calib:])
    return sorted(calib), sorted(test)


# ----------------------------------------------------------------------
# Estimación
# ----------------------------------------------------------------------
def _estimate(prob_seq, tst_h, umbral, tol, min_vent):
    """Devuelve (n_eventos_estimados, ahi_estimado)."""
    binary = (prob_seq >= umbral).astype(np.int8)
    n_events = count_events(binary, tol, min_vent)
    ahi = n_events / tst_h if tst_h > 0 else np.nan
    return n_events, ahi


def _calibrate(seqs, gt, calib_ids, umbrales, tolerancias, min_vent):
    """Barrido umbral × tolerancia minimizando MAE contra ahi_full."""
    tst_h = {s: gt.loc[s, "sleep_time"] / 3600.0 for s in calib_ids}
    ahi_real = np.array([gt.loc[s, "ahi_full"] for s in calib_ids], dtype=float)

    rows = []
    for u in umbrales:
        for tol in tolerancias:
            ahi_est = np.array(
                [_estimate(seqs[s], tst_h[s], u, tol, min_vent)[1] for s in calib_ids],
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


def _evaluate_test(seqs, gt, test_ids, umbral, tol, min_vent):
    rows = []
    for s in test_ids:
        tst_h = gt.loc[s, "sleep_time"] / 3600.0
        ahi_full = gt.loc[s, "ahi_full"]
        ahi_csv = gt.loc[s, "ahi_csv"]
        n_est, ahi_est = _estimate(seqs[s], tst_h, umbral, tol, min_vent)
        rows.append({
            "subject_id": s,
            "tst_h": round(tst_h, 3),
            "n_eventos_real_full": int(gt.loc[s, "n_eventos_full"]),
            "n_eventos_real_csv": int(gt.loc[s, "n_eventos_csv"]),
            "n_cen": int(gt.loc[s, "n_cen"]),
            "n_eventos_est": int(n_est),
            "ahi_real": ahi_full,          # objetivo primario (con centrales)
            "ahi_csv": ahi_csv,            # referencia (sin centrales)
            "ahi_est": ahi_est,
            "sev_real": _severity(ahi_full),
            "sev_est": _severity(ahi_est),
            "diff": ahi_est - ahi_full,
            "diff_vs_csv": ahi_est - ahi_csv,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def _plot_bland_altman(df, out_path, umbral, tol, col_real="ahi_real",
                       titulo="AHI con centrales"):
    mean = (df[col_real] + df["ahi_est"]) / 2
    diff = df["ahi_est"] - df[col_real]
    bias, sd = diff.mean(), diff.std()
    loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mean, diff, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax.axhline(bias, color="b", ls="-", lw=1.2, label=f"sesgo={bias:+.2f}")
    ax.axhline(loa_hi, color="r", ls="--", lw=1, label=f"+1.96 SD={loa_hi:+.2f}")
    ax.axhline(loa_lo, color="r", ls="--", lw=1, label=f"-1.96 SD={loa_lo:+.2f}")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Media (AHI real, AHI estimado)")
    ax.set_ylabel("AHI estimado − AHI real")
    ax.set_title(f"Bland-Altman — {titulo} (umbral={umbral}, tol={tol}s)")
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
    ax.set_ylabel("Severidad real (AHI con centrales)")
    ax.set_title(f"Matriz severidad AASM — umbral={umbral}, tol={tol}s")
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
    ax.set_title("Calibración vs AHI con centrales — MAE (menor = mejor)")
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
    sfx = OUTPUT_SUFFIX

    print("=" * 64)
    print("[FASE 9 · ETAPA B — variante] AHI recalculado CON apneas centrales")
    print("=" * 64)
    print(f"[INFO] Cohorte: {CLINICAL_COHORT} | Modo: {STAGE_B_MODE}")
    print("[INFO] Objetivo = (n_obs+n_cen+n_mix+n_hyp) / (sleep_time/3600)")

    pred = _load_predictions(Path(CLINICAL_PROBS_CSV))
    gt = _load_ground_truth(Path(AHI_GT_CSV))
    seqs = _prob_sequences(pred)

    subjects = [s for s in seqs
                if s in gt.index and not np.isnan(gt.loc[s, "ahi_full"])]
    print(f"[INFO] Sujetos válidos: {len(subjects)} (excluidos: {list(EXCLUDED_SUBJECTS)})")

    # Cuánto se separan las dos definiciones en esta cohorte
    d = pd.DataFrame({
        "full": [gt.loc[s, "ahi_full"] for s in subjects],
        "csv": [gt.loc[s, "ahi_csv"] for s in subjects],
        "n_cen": [gt.loc[s, "n_cen"] for s in subjects],
    })
    print(f"[INFO] AHI_full − AHI_csv : media={(d['full']-d['csv']).mean():+.2f}, "
          f"max={(d['full']-d['csv']).max():+.2f}")
    print(f"[INFO] Centrales/sujeto   : media={d['n_cen'].mean():.1f}, "
          f"max={d['n_cen'].max()}, sujetos con 0 centrales={(d['n_cen']==0).sum()}")
    print("-" * 64)

    if STAGE_B_MODE == "fixed":
        best_u, best_tol = STAGE_B_FIXED_UMBRAL, STAGE_B_FIXED_TOL
        test_ids = sorted(subjects)
        print(f"[FIXED] umbral={best_u}, tol={best_tol}s sobre {len(test_ids)} sujetos.")

    elif STAGE_B_MODE == "recalib":
        calib_ids, test_ids = _stratified_split(
            subjects, gt, STAGE_B_CALIB_FRAC, STAGE_B_SPLIT_SEED)
        print(f"[INFO] Calibración: {len(calib_ids)} | Test: {len(test_ids)}")
        sweep, best_u, best_tol = _calibrate(
            seqs, gt, calib_ids, STAGE_B_UMBRALES, STAGE_B_TOLERANCIAS,
            STAGE_B_MIN_VENTANAS)
        sweep.sort_values("MAE").to_csv(out_dir / f"stage_b_calib_sweep{sfx}.csv",
                                        index=False)
        _plot_calib_heatmap(sweep, out_dir / f"stage_b_calib_heatmap{sfx}.png")
        print(f"[CALIB] Mejor par: umbral={best_u}, tol={best_tol}s "
              f"(MAE={sweep['MAE'].min():.2f})")
        print(sweep.sort_values("MAE").head(5).to_string(index=False))
    else:
        raise ValueError(f"STAGE_B_MODE no reconocido: {STAGE_B_MODE}")

    print("-" * 64)

    test_df = _evaluate_test(seqs, gt, test_ids, best_u, best_tol, STAGE_B_MIN_VENTANAS)
    test_df.to_csv(out_dir / f"stage_b_test_per_subject{sfx}.csv", index=False)

    bias, sd, loa_lo, loa_hi = _plot_bland_altman(
        test_df, out_dir / f"stage_b_bland_altman{sfx}.png", best_u, best_tol,
        col_real="ahi_real", titulo="AHI con centrales")
    M, acc = _plot_confusion(
        test_df, out_dir / f"stage_b_confusion_severity{sfx}.png", best_u, best_tol)

    mae = test_df["diff"].abs().mean()
    corr = test_df["ahi_est"].corr(test_df["ahi_real"])

    # Referencia: mismas predicciones contra la definición del dataset
    mae_csv = test_df["diff_vs_csv"].abs().mean()
    bias_csv = test_df["diff_vs_csv"].mean()
    corr_csv = test_df["ahi_est"].corr(test_df["ahi_csv"])

    print("[RESULTADO] Contra AHI recalculado CON centrales (objetivo primario):")
    print(f"       MAE AHI      : {mae:.2f}")
    print(f"       Sesgo (B-A)  : {bias:+.2f}  (SD={sd:.2f})")
    print(f"       LoA          : [{loa_lo:+.2f}, {loa_hi:+.2f}]")
    print(f"       Corr AHI     : {corr:.4f}")
    print(f"       Acc severidad: {acc*100:.1f}%")
    print(f"       {AASM_SEVERITY_LABELS}")
    print(M)
    print("-" * 64)
    print("[REFERENCIA] Mismas predicciones contra AHI del CSV (sin centrales):")
    print(f"       MAE={mae_csv:.2f} | sesgo={bias_csv:+.2f} | corr={corr_csv:.4f}")
    print(f"       Δ MAE (csv − full) = {mae_csv - mae:+.2f}")
    print("-" * 64)
    print(f"[OK] CSV por sujeto  : {out_dir / f'stage_b_test_per_subject{sfx}.csv'}")
    print(f"[OK] Bland-Altman    : {out_dir / f'stage_b_bland_altman{sfx}.png'}")
    print(f"[OK] Matriz severidad: {out_dir / f'stage_b_confusion_severity{sfx}.png'}")
    print("[DONE]")


if __name__ == "__main__":
    main()