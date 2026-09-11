#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_resampling_figures.py
---------------------------
Genera las figuras de la memoria que justifican las decisiones de remuestreo
del Paso 1 del preprocesado (ver src/preprocessing/resample.py):

  Fig1_airflow_<COHORT>_<ID>.png
      Conserva la morfología con resample_poly (nativo vs remuestreado).
      Se genera una para STNF (upsampling 32->50 Hz) y otra para STLK
      (downsampling 500->50 Hz).

  Fig2_spo2_ringing_<ID>.png
      Argumento clave: zero-order hold (método real, tal y como sale de
      resampled_data/) NO produce ringing de Gibbs, a diferencia de un
      remuestreo filtrado (aquí simulado con scipy.signal.resample solo
      para ilustrar la alternativa que NO se usa). Panel (a) con un escalón
      real de la grabación elegida; panel (b) con un escalón sintético
      100%->88% para que el overshoot por encima del 100% fisiológico se
      vea con claridad (se etiqueta explícitamente como no real).

  Fig3_stlk_downsampling_validacion_<ID>.png
      Solo STLK: SpO2 nativa (500 Hz) vs remuestreada real (50 Hz, zero-order
      hold ya aplicado en resampled_data/) superpuestas sobre un evento de
      desaturación real. Verifica visualmente que el downsampling sin filtro
      no distorsiona el evento.

  Fig4_spo2_spectrum_STLK_<ID>.png
      Espectro de potencia (Welch) de la SpO2 nativa de STLK, marcando la
      nueva Nyquist tras bajar a 50 Hz (25 Hz), para justificar que el
      aliasing por no filtrar es despreciable.

Uso (desde la raíz del proyecto, con el venv activado):
    python -m src.graphs.make_resampling_figures
    python -m src.graphs.make_resampling_figures --stnf STNF00136 --stlk STLK00012
    python -m src.graphs.make_resampling_figures --seed 7   # reproducible

Requiere pares completos (mismo ID en dta_results/ y resampled_data/).
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample as fft_resample, welch

# =============================================================================
# CONFIGURACIÓN — ajusta aquí si tus rutas cambian
# (candidatos a mover a src/graphs/config.py: RESAMPLED_DATA_DIR no existe
#  ahí todavía, solo DTA_RESULTS_DIR)
# =============================================================================
DTA_RESULTS_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/dta_results")
RESAMPLED_DATA_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/data/module_results/resampled_data")
OUTPUT_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0/src/graphs/resampling_figures")

DPI = 300
COHORTS = ("STNF", "STLK")
FNAME_RE = re.compile(r"^(STNF|STLK)(\d+)_dta_results\.npz$")
INVALID_SPO2 = (0, 127)  # códigos de error observados en el canal SpO2

COL_NATIVE = "black"
COL_REAL = "#2ca02c"   # verde: método real (zero-order hold / resample_poly, ya en tus .npz)
COL_RING = "#d62728"   # rojo: alternativa NO usada (remuestreo filtrado/FFT), solo ilustrativa

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# 1) Localizar pares nativo + remuestreado válidos
# =============================================================================
def find_valid_pairs(dta_dir: Path, resampled_dir: Path) -> dict[str, list[str]]:
    """Empareja *_dta_results.npz con su resampled_<ID>.npz correspondiente.
    Devuelve {"STNF": [ids...], "STLK": [ids...]} solo con IDs presentes en
    ambas carpetas (mismo criterio que usa main_resample.py para nombrar)."""
    pairs: dict[str, list[str]] = {c: [] for c in COHORTS}
    if not dta_dir.exists():
        raise FileNotFoundError(f"No existe DTA_RESULTS_DIR: {dta_dir}")
    if not resampled_dir.exists():
        raise FileNotFoundError(f"No existe RESAMPLED_DATA_DIR: {resampled_dir}")

    for f in sorted(dta_dir.glob("*_dta_results.npz")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        cohort, num = m.groups()
        subj_id = f"{cohort}{num}"
        if (resampled_dir / f"resampled_{subj_id}.npz").exists():
            pairs[cohort].append(subj_id)

    for c in COHORTS:
        print(f"[INFO] {c}: {len(pairs[c])} pares nativo+remuestreado válidos.")
    return pairs


def pick_subject(pairs: dict[str, list[str]], cohort: str, forced_id: str | None, rng: random.Random) -> str:
    if forced_id:
        if forced_id not in pairs[cohort]:
            raise ValueError(f"{forced_id} no tiene par nativo+remuestreado válido en disco.")
        return forced_id
    if not pairs[cohort]:
        raise RuntimeError(f"No se encontró ningún par válido para {cohort}.")
    chosen = rng.choice(pairs[cohort])
    print(f"[INFO] Sujeto {cohort} elegido al azar: {chosen} (de {len(pairs[cohort])} candidatos)")
    return chosen


def load_pair(dta_dir: Path, resampled_dir: Path, subj_id: str) -> dict:
    native = np.load(dta_dir / f"{subj_id}_dta_results.npz", allow_pickle=True)
    resamp = np.load(resampled_dir / f"resampled_{subj_id}.npz", allow_pickle=True)
    return {
        "id": subj_id,
        "af_n": native["airflow"], "fs_afn": float(native["fs_airflow"]),
        "sp_n": native["spo2"],    "fs_spn": float(native["fs_spo2"]),
        "af_r": resamp["airflow"], "fs_afr": float(resamp["fs_airflow"]),
        "sp_r": resamp["spo2"],    "fs_spr": float(resamp["fs_spo2"]),
    }


# =============================================================================
# 2) Localizar tramos "buenos" automáticamente (sin depender de un ID fijo)
# =============================================================================
def _dominant_period_s(seg: np.ndarray, fs: float) -> float | None:
    """Periodo (s) de la componente de mayor energía en el espectro de seg
    (tras quitar la media). None si el segmento es demasiado corto o plano."""
    seg = seg - np.mean(seg)
    if len(seg) < 8 or np.std(seg) < 1e-9:
        return None
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
    spec[0] = 0.0  # ignora la componente DC
    if not np.any(spec > 0):
        return None
    f_peak = freqs[int(np.argmax(spec))]
    return (1.0 / f_peak) if f_peak > 0 else None


def find_clean_airflow_window(af_native: np.ndarray, fs_native: float,
                               af_ref: np.ndarray, fs_ref: float,
                               win_s: float = 20.0,
                               search_frac: tuple[float, float] = (0.3, 0.7)) -> float:
    """Devuelve el instante (s) de inicio de una ventana de win_s segundos con
    modulación respiratoria clara, sin depender de una escala de amplitud fija
    (STNF y STLK difieren ~10-20x en amplitud de airflow, así que un umbral de
    std absoluto no generaliza entre cohortes -- bug detectado y corregido).

    Busca la modulación sobre af_ref (la señal YA remuestreada/filtrada por
    resample_poly): en STLK esto separa la respiración real del ruido de alta
    frecuencia propio del termistor (Nasal_Therm); en STNF (upsampling) af_ref
    conserva la misma información que la nativa, así que el criterio vale
    igual para ambas. Se queda con el 30% de ventanas más variables (el resto
    es previsiblemente ruido de fondo) y las recorre de mayor a menor
    variabilidad, descartando: (1) ventanas dominadas por una TENDENCIA
    (ajuste lineal con R² alto) -- una respiración real oscila en torno a una
    media estable, mientras que una deriva/recuperación de sensor (ej. una
    caída o subida sostenida de cientos de u.a. en segundos) también tiene
    std alto pero NO es fisiológica; sin este filtro el buscador prefería
    justo esos artefactos por tener la variabilidad más alta (bug detectado
    y corregido); y (2) ventanas cuyo tramo NATIVO correspondiente tenga
    valores extremos atípicos (posible saturación), con un criterio robusto
    (MAD) en vez de un percentil superior fijo -- un techo de percentil
    (ej. <p95) descartaba precisamente los episodios de respiración más
    claros cuando eran minoritarios en el registro (bug previo, también
    corregido)."""
    n_ref = len(af_ref)
    lo, hi = int(n_ref * search_frac[0]), int(n_ref * search_frac[1])
    win_ref = max(1, int(win_s * fs_ref))
    starts_ref = np.arange(lo, max(lo, hi - win_ref), win_ref)
    if len(starts_ref) == 0:
        return lo / fs_ref

    stds = np.array([np.std(af_ref[s:s + win_ref]) for s in starts_ref])
    # Suavizado sobre 3 bloques consecutivos: prima la modulación SOSTENIDA
    # (varios bloques seguidos con variabilidad clara, ej. respiración
    # regular) frente a un único bloque con un pico aislado (un movimiento
    # puntual, no representativo de "un tramo normal"), sin dejar de
    # encontrar el evento si es minoritario en el registro (caso STLK).
    kernel = np.ones(3) / 3
    stds_smooth = np.convolve(stds, kernel, mode="same") if len(stds) >= 3 else stds
    p70 = np.percentile(stds_smooth, 70)
    mask = stds_smooth >= p70
    candidates = starts_ref[mask][np.argsort(stds_smooth[mask])[::-1]]  # de mayor a menor variabilidad sostenida

    win_native = max(1, int(win_s * fs_native))
    for s in candidates:
        seg_ref = af_ref[s:s + win_ref]
        if len(seg_ref) >= 8 and np.std(seg_ref) > 1e-9:
            r2 = float(np.corrcoef(np.arange(len(seg_ref)), seg_ref)[0, 1]) ** 2
            if r2 > 0.5:
                continue  # dominado por una tendencia sostenida, no oscilación (posible artefacto/deriva)

            period_s = _dominant_period_s(seg_ref, fs_ref)
            if period_s is None or not (1.5 <= period_s <= 10.0):
                continue  # la oscilación dominante no está en el rango fisiológico de respiración
                # (12-40 resp/min ~ periodo 1.5-5s; margen ampliado a 10s por
                # posibles bradipneas. Sin este filtro, la ventana de mayor
                # variabilidad podía ser una oscilación rápida y regular
                # -no respiratoria- con envolvente decreciente, p.ej. un
                # artefacto mecánico/de sensor. Bug detectado y corregido.)

        t0 = s / fs_ref
        i0 = int(t0 * fs_native)
        seg_n = af_native[i0:i0 + win_native]
        if len(seg_n) == 0 or np.any(np.isnan(seg_n)):
            continue
        mad = float(np.median(np.abs(seg_n - np.median(seg_n)))) + 1e-9
        if np.any(np.abs(seg_n - np.median(seg_n)) > 20 * mad):
            continue  # posible saturación/artefacto en la señal nativa
        return t0

    print("[WARN] No se encontró ventana con modulación clara y sin posible saturación; "
          "se usa el mejor candidato disponible (revisar a mano).")
    return candidates[0] / fs_ref


def find_spo2_step(sp: np.ndarray, fs: float, win_s: float = 20.0,
                    search_frac: tuple[float, float] = (0.15, 0.85)) -> tuple[float, float] | None:
    """Busca un escalón único y aislado (subida o bajada, con margen plano a
    ambos lados) en la SpO2 nativa. Prioriza escalones grandes y cercanos al
    techo fisiológico (más ilustrativos). Devuelve (t_inicio_ventana_s, step_size)
    o None si no se encuentra ninguno."""
    n = len(sp)
    lo, hi = int(n * search_frac[0]), int(n * search_frac[1])
    win = max(1, int(win_s * fs))
    grid = max(1, int(4 * fs))
    margin = max(1, int(1.0 * fs))

    candidates = []
    for start in range(lo, max(lo, hi - win), grid):
        seg = sp[start:start + win]
        if np.any(np.isin(seg, INVALID_SPO2)) or np.any(np.isnan(seg)):
            continue
        diffs = np.diff(seg)
        idx = np.where(np.abs(diffs) > 0)[0]
        if len(idx) == 1:
            i = int(idx[0])
            if i >= margin and (len(seg) - i - 1) >= margin:
                step_size = abs(float(diffs[i]))
                near_ceiling = max(seg[i], seg[i + 1]) >= 97
                candidates.append((start / fs, step_size, near_ceiling))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[2], c[1]), reverse=True)
    t0, step_size, _ = candidates[0]
    return t0, step_size


def find_desaturation_event(sp: np.ndarray, fs: float, drop_min: float = 3.0,
                             win_s: float = 120.0,
                             search_frac: tuple[float, float] = (0.15, 0.85)) -> float | None:
    """Busca una ventana de barrido de win_s segundos con una caída y
    recuperación de SpO2 (evento real de desaturación) de al menos drop_min
    puntos. Devuelve el instante (s) del PUNTO MÍNIMO del evento (no el
    inicio de la ventana de barrido), para poder centrar la figura en él;
    o None si no se encuentra ninguno."""
    n = len(sp)
    lo, hi = int(n * search_frac[0]), int(n * search_frac[1])
    win = max(1, int(win_s * fs))
    grid = max(1, int(10 * fs))
    edge = max(1, int(win * 0.15))

    best = None
    for start in range(lo, max(lo, hi - win), grid):
        seg = sp[start:start + win]
        if np.any(np.isin(seg, INVALID_SPO2)) or np.any(np.isnan(seg)):
            continue
        baseline = float(np.median(seg[:edge]))
        dip = baseline - float(np.min(seg))
        if dip >= drop_min:
            tail = float(np.median(seg[-edge:]))
            if abs(tail - baseline) <= 2.0:  # se recupera, no es solo una caída sostenida
                if best is None or dip > best[1]:
                    t_min = (start + int(np.argmin(seg))) / fs
                    best = (t_min, dip)
    if best is None:
        return None
    print(f"[INFO] Evento de desaturación encontrado: caída de {best[1]:.1f} pp.")
    return best[0]


# =============================================================================
# 3) Figuras
# =============================================================================
def fig_airflow_resample_poly(pair: dict, cohort: str, out_dir: Path) -> None:
    fs_n, fs_r = pair["fs_afn"], pair["fs_afr"]
    t0 = find_clean_airflow_window(pair["af_n"], fs_n, pair["af_r"], fs_r)
    direction = "upsampling" if fs_r > fs_n else "downsampling"

    # Contexto: duración fija en segundos. Zoom: duración adaptada a fs_n
    # para que el nº de muestras nativas mostradas sea manejable tanto a
    # 4/32 Hz (STNF) como a 500 Hz (STLK) -- con duración fija en segundos
    # el zoom a 500 Hz salía con miles de puntos superpuestos ("mala pinta").
    t_ctx = 15.0
    t_zoom = float(np.clip(120.0 / fs_n, 0.5, 4.0))

    def sl(fs, t_start, t_len):
        i0, i1 = int(t_start * fs), int((t_start + t_len) * fs)
        return np.arange(i0, i1), i0, i1

    tn_idx, i0n, i1n = sl(fs_n, t0, t_ctx)
    tr_idx, i0r, i1r = sl(fs_r, t0, t_ctx)
    jn_idx, j0n, j1n = sl(fs_n, t0, t_zoom)
    jr_idx, j0r, j1r = sl(fs_r, t0, t_zoom)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5))

    # Panel de contexto: solo líneas, sin marcadores -- a fs nativas altas
    # (STLK, 500 Hz) los marcadores generaban una mancha de puntos ilegible.
    ax = axes[0]
    ax.plot((tn_idx - tn_idx[0]) / fs_n, pair["af_n"][i0n:i1n], color=COL_NATIVE,
            lw=0.6, alpha=0.6, label=f"Original ({fs_n:.0f} Hz)")
    ax.plot((tr_idx - tr_idx[0]) / fs_r, pair["af_r"][i0r:i1r], color=COL_REAL,
            lw=1.3, alpha=0.9, label=f"Remuestreada, resample_poly ({fs_r:.0f} Hz)")
    ax.set_ylabel("Airflow (u.a.)")
    ax.set_title(f"(a) Contexto — {cohort} {pair['id']}, {direction} {fs_n:.0f} → {fs_r:.0f} Hz")
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
    ax.axvspan(0, t_zoom, color="gray", alpha=0.08)

    # Panel de detalle: aquí sí interesan los marcadores (nivel de muestra).
    ax = axes[1]
    ax.plot((jn_idx - jn_idx[0]) / fs_n, pair["af_n"][j0n:j1n], color=COL_NATIVE,
            lw=1.2, marker="o", ms=4.5, label=f"Original ({fs_n:.0f} Hz)")
    ax.plot((jr_idx - jr_idx[0]) / fs_r, pair["af_r"][j0r:j1r], color=COL_REAL,
            lw=1.0, marker="+", ms=6, alpha=0.8, label=f"Remuestreada ({fs_r:.0f} Hz)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Airflow (u.a.)")
    ax.set_title(f"(b) Detalle — ventana de {t_zoom:.2f} s")
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0)

    fig.suptitle(f"Fig. 1 — Conservación de morfología en resample_poly (airflow, {cohort}, sujeto {pair['id']})",
                 fontsize=10.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / f"Fig1_airflow_{cohort}_{pair['id']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura -> {out}")


def fig_spo2_zoh_vs_ringing(pair: dict, out_dir: Path) -> None:
    fs_n, fs_r = pair["fs_spn"], pair["fs_spr"]
    step_info = find_spo2_step(pair["sp_n"], fs_n)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # --- Panel (a): escalón real de la grabación elegida ---
    ax = axes[0]
    if step_info is not None:
        t0, step_size = step_info
        i0 = int(t0 * fs_n)
        win = int(20 * fs_n)
        seg = pair["sp_n"][i0:i0 + win].astype(float)
        t_native = np.arange(len(seg)) / fs_n

        j0 = int(t0 * fs_r)
        n_up = int(round(len(seg) * fs_r / fs_n))
        real_seg = pair["sp_r"][j0:j0 + n_up].astype(float)  # método real (ya en resampled_data/)
        ring_seg = fft_resample(seg, n_up)                    # alternativa NO usada, solo ilustrativa
        t_up = np.arange(len(real_seg)) / fs_r

        ax.plot(t_native, seg, color=COL_NATIVE, marker="o", ms=5, lw=1.3,
                drawstyle="steps-post", label=f"Original ({fs_n:.0f} Hz)")
        ax.plot(t_up, real_seg, color=COL_REAL, lw=1.6, alpha=0.85,
                label=f"Método real, ya en resampled_data/ ({fs_r:.0f} Hz)")
        ax.plot(t_up, ring_seg, color=COL_RING, lw=1.3, ls="--", alpha=0.9,
                label="Remuestreo filtrado/FFT (NO usado, solo ilustrativo)")
        overshoot = float(ring_seg.max() - seg.max())
        ax.set_title(f"(a) Escalón real ({pair['id']})\nescalón={step_size:.0f} pp, overshoot≈{overshoot:.2f} pp")
    else:
        ax.text(0.5, 0.5, "No se encontró ningún escalón aislado\nen este sujeto",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(a) Escalón real — no disponible en este sujeto")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("SpO$_2$ (%)")
    ax.legend(loc="lower right", fontsize=7.3, framealpha=1.0)

    # --- Panel (b): escalón sintético idealizado (100% -> 88%), para ver el overshoot > 100% ---
    # La meseta se define en SEGUNDOS (no en nº de muestras) para que el
    # escalón tenga sentido sea cual sea la fs nativa del sujeto (4 Hz en
    # STNF, 500 Hz en STLK): con muestras fijas, a 500 Hz el "escalón" duraba
    # milisegundos y el FFT no tenía datos suficientes (bug detectado en test).
    plateau_s = 5.0
    n_half = max(4, int(round(plateau_s * fs_n)))
    seg_s = np.concatenate([np.full(n_half, 100.0), np.full(n_half, 88.0)])
    t_native_s = np.arange(len(seg_s)) / fs_n
    n_up_s = int(round(len(seg_s) * fs_r / fs_n))
    idx = np.minimum((np.arange(n_up_s) * fs_n / fs_r).astype(int), len(seg_s) - 1)
    zoh_s = seg_s[idx]  # zero-order hold explícito (equivalente al método real, ver resample_spo2_exact)
    ring_s = fft_resample(seg_s, n_up_s)
    t_up_s = np.arange(n_up_s) / fs_r

    ax = axes[1]
    ax.axhline(100, color="gray", lw=0.8, ls=":", label="Límite fisiológico (100 %)")
    ax.plot(t_native_s, seg_s, color=COL_NATIVE, lw=1.3,
            drawstyle="steps-post", label=f"Escalón sintético ({fs_n:.0f} Hz)")
    ax.plot(t_up_s, zoh_s, color=COL_REAL, lw=1.6, alpha=0.85, label="Zero-order hold — método real")
    ax.plot(t_up_s, ring_s, color=COL_RING, lw=1.3, ls="--", alpha=0.9, label="Remuestreo filtrado/FFT")
    ax.set_xlim(plateau_s - 2, plateau_s + 2)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("SpO$_2$ (%)")
    overshoot_b = float(ring_s.max() - 100.0)
    ax.set_title(f"(b) Escalón sintético ilustrativo (no real)\novershoot = {overshoot_b:.2f} pp > 100 %")
    ax.legend(loc="center right", fontsize=7.3, framealpha=1.0)

    fig.suptitle("Fig. 2 — SpO$_2$: zero-order hold (método real) vs. remuestreo filtrado (Gibbs ringing)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = out_dir / f"Fig2_spo2_ringing_{pair['id']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura -> {out}")


def fig_stlk_downsampling_validation(pair: dict, out_dir: Path) -> None:
    fs_n, fs_r = pair["fs_spn"], pair["fs_spr"]
    if fs_r >= fs_n:
        print(f"[WARN] {pair['id']} no es un caso de downsampling (fs_n={fs_n}, fs_r={fs_r}); se omite Fig3.")
        return

    t_center = find_desaturation_event(pair["sp_n"], fs_n)
    win_s = 90.0
    if t_center is None:
        print("[WARN] No se encontró evento de desaturación claro; se usa un tramo central genérico.")
        t_center = len(pair["sp_n"]) / fs_n * 0.5

    t0 = max(0.0, t_center - win_s / 2)  # centra la ventana de la figura en el mínimo del evento

    i0 = int(t0 * fs_n)
    win_n = int(win_s * fs_n)
    seg_n = pair["sp_n"][i0:i0 + win_n].astype(float)
    t_n = np.arange(len(seg_n)) / fs_n

    j0 = int(t0 * fs_r)
    win_r = int(win_s * fs_r)
    seg_r = pair["sp_r"][j0:j0 + win_r].astype(float)
    t_r = np.arange(len(seg_r)) / fs_r

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t_n, seg_n, color=COL_NATIVE, lw=1.4, alpha=0.9, label=f"Original ({fs_n:.0f} Hz)")
    ax.plot(t_r, seg_r, color=COL_REAL, lw=1.6, ls="--", alpha=0.85, label=f"Remuestreada, método real ({fs_r:.0f} Hz)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("SpO$_2$ (%)")
    ax.set_title(f"Fig. 3 — STLK {pair['id']}: downsampling 500→50 Hz sin filtro\n"
                 f"sobre un evento de desaturación real (caída conservada)")
    ax.legend(loc="lower right", fontsize=8, framealpha=1.0)
    fig.tight_layout()
    out = out_dir / f"Fig3_stlk_downsampling_validacion_{pair['id']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura -> {out}")


def fig_spo2_spectrum(pair: dict, out_dir: Path, new_nyquist: float = 25.0) -> None:
    sp = pair["sp_n"].astype(float)
    fs = pair["fs_spn"]
    valid = ~np.isin(sp, INVALID_SPO2)
    sp_clean = sp[valid]
    if len(sp_clean) < 4096:
        print("[WARN] Señal SpO2 nativa demasiado corta/incompleta para el espectro; se omite Fig4.")
        return

    f, pxx = welch(sp_clean, fs=fs, nperseg=min(8192, len(sp_clean)))
    frac_below = float(pxx[f <= new_nyquist].sum() / pxx.sum())

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(f, pxx, color=COL_NATIVE, lw=1)
    ax.axvline(new_nyquist, color=COL_RING, ls="--",
               label=f"Nueva Nyquist tras bajar a 50 Hz ({new_nyquist:.0f} Hz)")
    ax.set_xlim(0, fs / 2)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("PSD (u.a.²/Hz)")
    ax.set_title(f"Fig. 4 — Espectro SpO$_2$ nativa STLK {pair['id']} (fs={fs:.0f} Hz)\n"
                 f"{frac_below*100:.2f}% de la potencia está por debajo de {new_nyquist:.0f} Hz")
    ax.legend(fontsize=8, framealpha=1.0)
    fig.tight_layout()
    out = out_dir / f"Fig4_spo2_spectrum_STLK_{pair['id']}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura -> {out}")


# =============================================================================
# 4) main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stnf", default=None, help="ID STNF forzado (ej. STNF00136). Si se omite, se elige al azar.")
    parser.add_argument("--stlk", default=None, help="ID STLK forzado (ej. STLK00012). Si se omite, se elige al azar.")
    parser.add_argument("--seed", type=int, default=None, help="Semilla para el sorteo (reproducibilidad).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    pairs = find_valid_pairs(DTA_RESULTS_DIR, RESAMPLED_DATA_DIR)

    stnf_id = pick_subject(pairs, "STNF", args.stnf, rng)
    stlk_id = pick_subject(pairs, "STLK", args.stlk, rng)

    stnf_pair = load_pair(DTA_RESULTS_DIR, RESAMPLED_DATA_DIR, stnf_id)
    stlk_pair = load_pair(DTA_RESULTS_DIR, RESAMPLED_DATA_DIR, stlk_id)

    fig_airflow_resample_poly(stnf_pair, "STNF", OUTPUT_DIR)
    fig_airflow_resample_poly(stlk_pair, "STLK", OUTPUT_DIR)

    # Fig2 (ringing): usa el sujeto con el escalón real más grande de los dos elegidos
    step_stnf = find_spo2_step(stnf_pair["sp_n"], stnf_pair["fs_spn"])
    step_stlk = find_spo2_step(stlk_pair["sp_n"], stlk_pair["fs_spn"])
    size_stnf = step_stnf[1] if step_stnf else -1
    size_stlk = step_stlk[1] if step_stlk else -1
    fig_spo2_zoh_vs_ringing(stnf_pair if size_stnf >= size_stlk else stlk_pair, OUTPUT_DIR)

    fig_stlk_downsampling_validation(stlk_pair, OUTPUT_DIR)
    fig_spo2_spectrum(stlk_pair, OUTPUT_DIR)

    print(f"\n[DONE] Figuras guardadas en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()