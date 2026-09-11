#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask_on_signal.py
------------------
Pinta la señal completa de una noche (airflow o SpO2) coloreada según en qué
máscara cae cada muestra, para ver de un vistazo qué tramos del registro se
descartan (fuera de mask_sleep) o se marcan como evento respiratorio
(mask_events dentro de mask_sleep).

Fuente de datos: mask_events_data/events_<ID>.npz, que ya trae airflow, spo2,
mask_sleep y mask_events alineados muestra a muestra (fs común, la de
resampled_data/). dta_results/ y mask_data/ NO se usan para pintar -- solo
para comprobar que el sujeto elegido superó las tres fases del pipeline
(dta -> mask_data -> mask_events_data) antes de fiarnos de sus máscaras.

Categorías (misma prioridad que segment_signals(), que descarta la ventana
si mask_sleep tiene algún 0, sin mirar mask_events):
  0 = fuera de mask_sleep (descartado, tenga o no evento marcado)
  1 = dentro de mask_sleep, sin evento
  2 = dentro de mask_sleep, con evento respiratorio (mask_events=1)

Genera 2 figuras (una por canal), cada una con 2 paneles apilados
(STNF arriba, STLK abajo):
    Fig_mask_airflow.png
    Fig_mask_spo2.png

Uso:
    python mask_on_signal.py
    python mask_on_signal.py --stnf STNF00136 --stlk STLK00012
    python mask_on_signal.py --seed 3 --decimate-airflow 10 --decimate-spo2 100

La SpO2 se suaviza (media móvil) antes de decimar y con una decimación más
agresiva que el airflow: a resolución completa, el jitter muestra-a-muestra
(±0.2-0.5%) hace que la traza de la noche entera se vea como ruido denso, y
lo que interesa a esa escala es la tendencia, no la muestra individual. El
airflow NO se suaviza porque a esa escala interesa seguir viendo que hay
respiración (aunque sea como una banda densa).
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from scipy.ndimage import uniform_filter1d

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DTA_RESULTS_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/dta_results")
MASK_DATA_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/mask_data")
MASK_EVENTS_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/mask_events_data")
OUTPUT_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/src/graphs/masks+signals")

DPI = 200
COHORTS = ("STNF", "STLK")
EVENTS_RE = re.compile(r"^events_(STNF|STLK)(\d+)\.npz$")

# Decimación por defecto para el render de la noche completa (~2M muestras a
# 50Hz): pintar 1 de cada N puntos. La estructura de tramos de la máscara
# (minutos) es muchísimo más gruesa que esto, así que no se pierde nada
# relevante y el PNG se genera en segundos, no en minutos.
# El airflow necesita más resolución para que se note que hay respiración;
# la SpO2 varía lento, así que se puede decimar mucho más agresivo -- y se
# suaviza antes (ver SPO2_SMOOTH_S) para que no se vea como ruido denso.
DEFAULT_DECIMATE_AIRFLOW = 10
DEFAULT_DECIMATE_SPO2 = 250
SPO2_SMOOTH_S = 30.0  # ventana de la media móvil aplicada a la SpO2, en segundos

COL_OUT = "#bbbbbb"     # gris: fuera de mask_sleep (descartado)
COL_SLEEP = "#1f77b4"   # azul: dentro de sueño, sin evento
COL_EVENT = "#d62728"   # rojo: dentro de sueño, con evento respiratorio

# Variante para SpO2: colores más saturados/oscuros y línea más gruesa,
# porque a la decimación agresiva (250) + suavizado, la línea fina se ve
# débil/pequeña en la figura de la noche completa.
COL_OUT_SPO2 = "#999999"
COL_SLEEP_SPO2 = "#0a4f8c"
COL_EVENT_SPO2 = "#a80000"
LW_SPO2 = 1.3

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# 1) Localizar sujetos válidos (presentes en las 3 fases del pipeline)
# =============================================================================
def find_valid_subjects() -> dict[str, list[str]]:
    """Devuelve {"STNF": [ids...], "STLK": [ids...]} solo con IDs que tienen
    archivo en dta_results/, mask_data/ Y mask_events_data/."""
    for d in (DTA_RESULTS_DIR, MASK_DATA_DIR, MASK_EVENTS_DIR):
        if not d.exists():
            raise FileNotFoundError(f"No existe la carpeta: {d}")

    subjects: dict[str, list[str]] = {c: [] for c in COHORTS}
    for f in sorted(MASK_EVENTS_DIR.glob("events_*.npz")):
        m = EVENTS_RE.match(f.name)
        if not m:
            continue
        cohort, num = m.groups()
        subj_id = f"{cohort}{num}"
        has_dta = (DTA_RESULTS_DIR / f"{subj_id}_dta_results.npz").exists()
        has_mask = (MASK_DATA_DIR / f"masked_resampled_{subj_id}.npz").exists()
        if has_dta and has_mask:
            subjects[cohort].append(subj_id)

    for c in COHORTS:
        print(f"[INFO] {c}: {len(subjects[c])} sujetos completos (dta + mask_data + mask_events_data).")
    return subjects


def pick_subject(subjects: dict[str, list[str]], cohort: str, forced_id: str | None, rng: random.Random) -> str:
    if forced_id:
        if forced_id not in subjects[cohort]:
            raise ValueError(f"{forced_id} no está completo en las 3 carpetas.")
        return forced_id
    if not subjects[cohort]:
        raise RuntimeError(f"No hay ningún sujeto {cohort} completo.")
    chosen = rng.choice(subjects[cohort])
    print(f"[INFO] Sujeto {cohort} elegido al azar: {chosen} (de {len(subjects[cohort])} candidatos)")
    return chosen


def load_events_npz(subj_id: str) -> dict:
    d = np.load(MASK_EVENTS_DIR / f"events_{subj_id}.npz", allow_pickle=True)
    required = {"airflow", "spo2", "mask_sleep", "mask_events", "fs"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"events_{subj_id}.npz no tiene las claves {missing}")

    L = min(len(d["airflow"]), len(d["spo2"]), len(d["mask_sleep"]), len(d["mask_events"]))
    return {
        "id": subj_id,
        "fs": float(d["fs"]),
        "airflow": d["airflow"][:L],
        "spo2": d["spo2"][:L],
        "mask_sleep": d["mask_sleep"][:L].astype(bool),
        "mask_events": d["mask_events"][:L].astype(bool),
    }


# =============================================================================
# 2) Categorización y ploteo
# =============================================================================
def categorize(mask_sleep: np.ndarray, mask_events: np.ndarray) -> np.ndarray:
    """0 = fuera de sueño, 1 = sueño sin evento, 2 = sueño con evento."""
    cat = np.ones(len(mask_sleep), dtype=np.uint8)
    cat[~mask_sleep] = 0
    cat[mask_sleep & mask_events] = 2
    return cat


def plot_masked_channel(ax, signal: np.ndarray, cat: np.ndarray, fs: float,
                         decimate: int, ylabel: str, smooth_s: float = 0.0,
                         col_out: str = COL_OUT, col_sleep: str = COL_SLEEP,
                         col_event: str = COL_EVENT, lw: float = 0.5,
                         lw_event: float | None = None) -> None:
    if smooth_s > 0:
        win = max(1, int(round(smooth_s * fs)))
        signal = uniform_filter1d(signal.astype(float), size=win, mode="nearest")

    if lw_event is None:
        lw_event = lw + 0.1

    sig = signal[::decimate]
    c = cat[::decimate]
    t_h = np.arange(len(sig)) * decimate / fs / 3600.0  # tiempo en horas

    # Una serie por categoría, con NaN fuera de ella: matplotlib corta la
    # línea automáticamente en los huecos, así que no hay que trocear a mano.
    s_out = np.where(c == 0, sig, np.nan)
    s_sleep = np.where(c == 1, sig, np.nan)
    s_event = np.where(c == 2, sig, np.nan)

    ax.plot(t_h, s_out, color=col_out, lw=lw, rasterized=True,
            label="Fuera de mask_sleep (descartado)")
    ax.plot(t_h, s_sleep, color=col_sleep, lw=lw, rasterized=True,
            label="Sueño, sin evento")
    ax.plot(t_h, s_event, color=col_event, lw=lw_event, rasterized=True,
            label="Sueño, con evento (mask_events)")
    ax.set_ylabel(ylabel)

    n = len(cat)
    pct_out = 100 * np.sum(cat == 0) / n
    pct_event = 100 * np.sum(cat == 2) / n
    ax.set_title(f"descartado: {pct_out:.1f}%   ·   con evento: {pct_event:.1f}%", fontsize=9)


def fig_mask_on_channel(channel: str, ylabel: str, stnf: dict, stlk: dict, out_dir: Path,
                         decimate: int, smooth_s: float = 0.0) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)

    if channel == "spo2":
        plot_kwargs = dict(col_out=COL_OUT_SPO2, col_sleep=COL_SLEEP_SPO2,
                            col_event=COL_EVENT_SPO2, lw=LW_SPO2)
    else:
        plot_kwargs = {}

    for ax, pair, cohort in ((axes[0], stnf, "STNF"), (axes[1], stlk, "STLK")):
        cat = categorize(pair["mask_sleep"], pair["mask_events"])
        plot_masked_channel(ax, pair[channel], cat, pair["fs"], decimate, ylabel, smooth_s, **plot_kwargs)
        ax.set_title(f"{cohort} {pair['id']} — {ax.get_title()}")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=1.0, ncol=3)

    axes[1].set_xlabel("Tiempo (h)")
    fig.suptitle(f"Máscaras sobre la señal completa — {channel}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = out_dir / f"Fig_mask_{channel}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura -> {out}")


# =============================================================================
# 3) main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stnf", default=None, help="ID STNF forzado (ej. STNF00136). Si se omite, al azar.")
    parser.add_argument("--stlk", default=None, help="ID STLK forzado (ej. STLK00012). Si se omite, al azar.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para el sorteo (reproducibilidad).")
    parser.add_argument("--decimate-airflow", type=int, default=DEFAULT_DECIMATE_AIRFLOW,
                         help=f"Airflow: pintar 1 de cada N muestras (por defecto {DEFAULT_DECIMATE_AIRFLOW}).")
    parser.add_argument("--decimate-spo2", type=int, default=DEFAULT_DECIMATE_SPO2,
                         help=f"SpO2: pintar 1 de cada N muestras (por defecto {DEFAULT_DECIMATE_SPO2}).")
    parser.add_argument("--spo2-smooth-s", type=float, default=SPO2_SMOOTH_S,
                         help=f"Ventana (s) de la media móvil sobre SpO2 antes de decimar "
                              f"(por defecto {SPO2_SMOOTH_S}; 0 para desactivar).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    subjects = find_valid_subjects()
    stnf_id = pick_subject(subjects, "STNF", args.stnf, rng)
    stlk_id = pick_subject(subjects, "STLK", args.stlk, rng)

    stnf = load_events_npz(stnf_id)
    stlk = load_events_npz(stlk_id)

    fig_mask_on_channel("airflow", "Airflow (u.a.)", stnf, stlk, OUTPUT_DIR, args.decimate_airflow)
    fig_mask_on_channel("spo2", "SpO$_2$ (%)", stnf, stlk, OUTPUT_DIR, args.decimate_spo2,
                         smooth_s=args.spo2_smooth_s)

    print(f"\n[DONE] Figuras guardadas en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()