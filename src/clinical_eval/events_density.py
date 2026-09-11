#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
events_density.py
-----------------
Mide la densidad temporal de los eventos respiratorios anotados, para
comprobar si los pacientes mas graves presentan intervalos entre eventos
consecutivos por debajo de la resolucion de la ventana de analisis (15 s).

No usa el modelo ni las predicciones: trabaja solo sobre mask_events,
es decir, sobre las etiquetas verdaderas del tecnico.

Entrada : delay_<ID>.npz  (claves: mask_events, mask_sleep, fs)
          ahi_labels.csv  (columnas: s_code, ahi)

Salida  : un CSV por sujeto, un CSV resumen por severidad y tres figuras,
          para cada cohorte por separado.

Uso:
    python -m src.clinical_eval.events_density
    python -m src.clinical_eval.events_density --cohorts STNF
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # sin ventana grafica: el cluster no tiene display
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------

BASE_DIR = Path("/mnt/home/users/ac_aux/portega/ProyectoPython3.0")
DELAY_DIR = BASE_DIR / "data" / "module_results" / "delay_data"
AHI_CSV = BASE_DIR / "ahi_labels.csv"
OUT_DIR = BASE_DIR / "src" / "clinical_eval" / "events_density_results"

COHORTS = ["STNF", "STLK"]

# Umbrales de hueco (en segundos) para los que se reporta la fraccion
# acumulada. El principal es 15 s, la duracion de la ventana de analisis;
# los otros dos son el control de robustez del corte.
GAP_THRESHOLDS_S = [10, 15, 20]
MAIN_THRESHOLD_S = 15

# Sujetos excluidos por defectos de la base de datos de origen (§4.6.2).
# ---> RELLENAR con los dos IDs de STNF que ya se descartaron.
#      El script avisa al final de los candidatos que detecta por su cuenta.
EXCLUDE_SUBJECTS = [
    # "STNF000XX",   # scoring respiratorio ausente en el CSV
    # "STNF000YY",   # evento continuo de ~686 s (mascara corrupta)
]

# Cortes de severidad AASM sobre el AHI de referencia.
SEVERITY_BINS = [
    ("Normal", 0.0, 5.0),
    ("Leve", 5.0, 15.0),
    ("Moderado", 15.0, 30.0),
    ("Grave", 30.0, float("inf")),
]
SEVERITY_ORDER = [name for name, _, _ in SEVERITY_BINS]
SEVERITY_COLORS = {
    "Normal": "#4C72B0",
    "Leve": "#55A868",
    "Moderado": "#DD8452",
    "Grave": "#C44E52",
}

# Duracion (s) por encima de la cual un evento se considera implausible.
# Solo se usa para avisar, no para excluir automaticamente.
IMPLAUSIBLE_EVENT_S = 300.0


# ----------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------

def find_runs(mask: np.ndarray) -> np.ndarray:
    """Devuelve los tramos de unos consecutivos de un vector binario.

    Salida: array (n_tramos, 2) con [inicio, fin) en indices de muestra.
    """
    m = np.asarray(mask).astype(bool).astype(np.int8)
    if m.size == 0:
        return np.empty((0, 2), dtype=int)
    # El padding con ceros hace que un tramo pegado al borde tambien
    # produzca su transicion correspondiente.
    padded = np.concatenate(([0], m, [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return np.stack([starts, ends], axis=1)


def load_ahi_table(csv_path: Path) -> dict[str, float]:
    """Lee s_code -> ahi del CSV de referencia (separador ';', decimal ',')."""
    table: dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            code = (row.get("s_code") or "").strip()
            raw = (row.get("ahi") or "").strip().replace(",", ".")
            if not code or not raw:
                continue
            try:
                table[code] = float(raw)
            except ValueError:
                continue
    return table


def severity_of(ahi: float) -> str:
    for name, lo, hi in SEVERITY_BINS:
        if lo <= ahi < hi:
            return name
    return SEVERITY_ORDER[-1]


# ----------------------------------------------------------------------
# ANALISIS POR SUJETO
# ----------------------------------------------------------------------

def analyse_subject(npz_path: Path) -> dict | None:
    """Calcula los huecos entre eventos consecutivos de un sujeto.

    Solo se conservan los huecos contenidos integramente en sueno: si entre
    dos eventos hay una vigilia intermedia, ese hueco se descarta en lugar de
    contabilizarse como un intervalo enorme.
    """
    data = np.load(npz_path, allow_pickle=True)
    if "mask_events" not in data:
        return None

    mask_events = np.asarray(data["mask_events"]).astype(bool)
    fs = float(np.asarray(data["fs"]).item()) if "fs" in data else 50.0

    if "mask_sleep" in data:
        mask_sleep = np.asarray(data["mask_sleep"]).astype(bool)
        n = min(mask_events.size, mask_sleep.size)
        mask_events, mask_sleep = mask_events[:n], mask_sleep[:n]
    else:
        mask_sleep = np.ones_like(mask_events)

    # Los eventos se restringen al sueno antes de detectar los tramos.
    events = find_runs(mask_events & mask_sleep)
    n_events = int(events.shape[0])

    durations_s = (events[:, 1] - events[:, 0]) / fs if n_events else np.array([])

    gaps_s: list[float] = []
    n_gaps_discarded = 0
    for i in range(n_events - 1):
        gap_start = int(events[i, 1])
        gap_end = int(events[i + 1, 0])
        if gap_end <= gap_start:
            continue
        # El hueco solo cuenta si el sujeto permanece dormido durante todo el.
        if not mask_sleep[gap_start:gap_end].all():
            n_gaps_discarded += 1
            continue
        gaps_s.append((gap_end - gap_start) / fs)

    gaps = np.asarray(gaps_s, dtype=float)

    result = {
        "n_events": n_events,
        "n_gaps": int(gaps.size),
        "n_gaps_discarded_wake": n_gaps_discarded,
        "sleep_time_h": float(mask_sleep.sum() / fs / 3600.0),
        "median_gap_s": float(np.median(gaps)) if gaps.size else float("nan"),
        "p25_gap_s": float(np.percentile(gaps, 25)) if gaps.size else float("nan"),
        "max_event_dur_s": float(durations_s.max()) if n_events else 0.0,
        "gaps": gaps,
    }
    for thr in GAP_THRESHOLDS_S:
        key = f"frac_gap_lt_{thr}s"
        result[key] = float((gaps < thr).mean()) if gaps.size else float("nan")
    return result


# ----------------------------------------------------------------------
# ANALISIS POR COHORTE
# ----------------------------------------------------------------------

def analyse_cohort(cohort: str, ahi_table: dict[str, float]) -> tuple[list[dict], list[str]]:
    files = sorted(DELAY_DIR.glob(f"delay_{cohort}*.npz"))
    print(f"\n[{cohort}] {len(files)} ficheros encontrados en {DELAY_DIR}")

    rows: list[dict] = []
    warnings: list[str] = []

    for path in files:
        subject = path.stem.replace("delay_", "")

        if subject in EXCLUDE_SUBJECTS:
            print(f"   [SKIP] {subject}: excluido por configuracion")
            continue

        try:
            res = analyse_subject(path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{subject}: error al leer ({exc})")
            continue

        if res is None:
            warnings.append(f"{subject}: el npz no contiene mask_events")
            continue

        if res["n_events"] == 0:
            warnings.append(f"{subject}: 0 eventos anotados (candidato a exclusion)")
            continue
        if res["max_event_dur_s"] > IMPLAUSIBLE_EVENT_S:
            warnings.append(
                f"{subject}: evento de {res['max_event_dur_s']:.0f} s "
                f"(candidato a exclusion)"
            )
        if res["n_gaps"] < 2:
            warnings.append(f"{subject}: menos de 2 huecos utiles, no entra en las figuras")
            continue

        ahi = ahi_table.get(subject)
        if ahi is None:
            warnings.append(f"{subject}: sin AHI en {AHI_CSV.name}")
            continue

        res["subject"] = subject
        res["ahi"] = ahi
        res["severity"] = severity_of(ahi)
        rows.append(res)

    print(f"[{cohort}] {len(rows)} sujetos analizados")
    return rows, warnings


# ----------------------------------------------------------------------
# SALIDAS: CSV
# ----------------------------------------------------------------------

def write_per_subject_csv(cohort: str, rows: list[dict]) -> None:
    fields = [
        "subject", "ahi", "severity", "n_events", "sleep_time_h", "n_gaps",
        "n_gaps_discarded_wake", "median_gap_s", "p25_gap_s", "max_event_dur_s",
    ] + [f"frac_gap_lt_{t}s" for t in GAP_THRESHOLDS_S]

    out = OUT_DIR / f"{cohort}_por_sujeto.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda x: x["ahi"]):
            writer.writerow({k: r.get(k) for k in fields})
    print(f"   -> {out.name}")


def write_summary_csv(cohort: str, rows: list[dict]) -> None:
    """Tabla agregada por severidad: la que se pega en la memoria."""
    out = OUT_DIR / f"{cohort}_resumen_severidad.csv"
    fields = ["severidad", "n_sujetos", "n_huecos", "mediana_hueco_s"] + [
        f"pct_huecos_lt_{t}s" for t in GAP_THRESHOLDS_S
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sev in SEVERITY_ORDER:
            grp = [r for r in rows if r["severity"] == sev]
            if not grp:
                continue
            # Se agrupan todos los huecos del estrato, no se promedian
            # porcentajes por sujeto: asi cada hueco pesa lo mismo.
            allg = np.concatenate([r["gaps"] for r in grp])
            row = {
                "severidad": sev,
                "n_sujetos": len(grp),
                "n_huecos": int(allg.size),
                "mediana_hueco_s": round(float(np.median(allg)), 2),
            }
            for t in GAP_THRESHOLDS_S:
                row[f"pct_huecos_lt_{t}s"] = round(float((allg < t).mean()) * 100, 2)
            writer.writerow(row)
    print(f"   -> {out.name}")


# ----------------------------------------------------------------------
# SALIDAS: FIGURAS
# ----------------------------------------------------------------------

def fig_scatter(cohort: str, rows: list[dict], xlim, ylim) -> None:
    """Figura principal: AHI real frente a % de huecos por debajo de 15 s."""
    key = f"frac_gap_lt_{MAIN_THRESHOLD_S}s"
    x = np.array([r["ahi"] for r in rows])
    y = np.array([r[key] * 100 for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sev in SEVERITY_ORDER:
        idx = [i for i, r in enumerate(rows) if r["severity"] == sev]
        if not idx:
            continue
        ax.scatter(x[idx], y[idx], s=34, alpha=0.75,
                   color=SEVERITY_COLORS[sev], edgecolor="white",
                   linewidth=0.5, label=sev)

    # Recta de tendencia y coeficiente de Pearson.
    if x.size >= 3:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="0.35", linewidth=1.4)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.97, 0.05, f"r = {r:.3f}   n = {x.size}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7"))

    ax.set_xlabel("AHI de referencia (eventos/hora)")
    ax.set_ylabel(f"Huecos entre eventos por debajo de {MAIN_THRESHOLD_S} s (%)")
    ax.set_title(f"{cohort} — densidad de eventos frente a severidad")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(title="Severidad AASM", frameon=False, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / f"{cohort}_scatter_ahi_vs_densidad.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"   -> {out.name}")


def fig_ecdf(cohort: str, rows: list[dict], xmax: float) -> None:
    """Apoyo: ECDF de los huecos, una curva por estrato de severidad."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for sev in SEVERITY_ORDER:
        grp = [r for r in rows if r["severity"] == sev]
        if not grp:
            continue
        allg = np.sort(np.concatenate([r["gaps"] for r in grp]))
        ecdf = np.arange(1, allg.size + 1) / allg.size
        ax.step(allg, ecdf * 100, where="post", linewidth=1.9,
                color=SEVERITY_COLORS[sev],
                label=f"{sev} (n={len(grp)})")

    ax.axvline(MAIN_THRESHOLD_S, color="0.3", linestyle="--", linewidth=1.3)
    ax.annotate(f"ventana = {MAIN_THRESHOLD_S} s",
                xy=(MAIN_THRESHOLD_S, 4), xytext=(MAIN_THRESHOLD_S + 4, 4),
                fontsize=9, color="0.3")

    ax.set_xlabel("Hueco entre eventos consecutivos (s)")
    ax.set_ylabel("Fraccion acumulada de huecos (%)")
    ax.set_title(f"{cohort} — distribucion de los intervalos entre eventos")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(title="Severidad AASM", frameon=False, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / f"{cohort}_ecdf_huecos.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"   -> {out.name}")


def fig_bars(cohort: str, rows: list[dict], ymax: float) -> None:
    """Control de robustez: % de huecos por debajo de 10, 15 y 20 s."""
    sevs = [s for s in SEVERITY_ORDER if any(r["severity"] == s for r in rows)]
    width = 0.8 / len(GAP_THRESHOLDS_S)
    xpos = np.arange(len(sevs))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    greys = ["0.72", "0.45", "0.20"]
    for j, thr in enumerate(GAP_THRESHOLDS_S):
        vals = []
        for sev in sevs:
            allg = np.concatenate([r["gaps"] for r in rows if r["severity"] == sev])
            vals.append(float((allg < thr).mean()) * 100)
        bars = ax.bar(xpos + j * width - 0.4 + width / 2, vals, width,
                      color=greys[j % len(greys)], edgecolor="white",
                      label=f"< {thr} s")
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)

    ax.set_xticks(xpos)
    ax.set_xticklabels(sevs)
    ax.set_ylabel("Huecos por debajo del umbral (%)")
    ax.set_title(f"{cohort} — robustez frente al umbral de hueco")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT_DIR / f"{cohort}_barras_umbrales.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"   -> {out.name}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=COHORTS,
                        help="cohortes a procesar (por defecto STNF y STLK)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ahi_table = load_ahi_table(AHI_CSV)
    print(f"[INFO] AHI cargado para {len(ahi_table)} sujetos")

    per_cohort: dict[str, list[dict]] = {}
    all_warnings: list[str] = []

    for cohort in args.cohorts:
        rows, warns = analyse_cohort(cohort, ahi_table)
        all_warnings += [f"[{cohort}] {w}" for w in warns]
        if rows:
            per_cohort[cohort] = rows

    if not per_cohort:
        print("[ERROR] Ningun sujeto analizado. Revisa DELAY_DIR y el CSV de AHI.")
        return

    # Escalas comunes entre cohortes, para que las figuras sean comparables.
    all_rows = [r for rows in per_cohort.values() for r in rows]
    key = f"frac_gap_lt_{MAIN_THRESHOLD_S}s"
    xlim = (0, max(r["ahi"] for r in all_rows) * 1.05)
    ylim = (0, min(100, max(r[key] for r in all_rows) * 100 * 1.15 + 5))
    ecdf_xmax = 120.0
    bar_ymax = min(100, max(
        float((np.concatenate([r["gaps"] for r in rows]) < max(GAP_THRESHOLDS_S)).mean())
        for rows in per_cohort.values()) * 100 * 1.25 + 5)

    for cohort, rows in per_cohort.items():
        print(f"\n[{cohort}] generando salidas")
        write_per_subject_csv(cohort, rows)
        write_summary_csv(cohort, rows)
        fig_scatter(cohort, rows, xlim, ylim)
        fig_ecdf(cohort, rows, ecdf_xmax)
        fig_bars(cohort, rows, bar_ymax)

    if all_warnings:
        log = OUT_DIR / "avisos.txt"
        log.write_text("\n".join(all_warnings), encoding="utf-8")
        print(f"\n[AVISOS] {len(all_warnings)} incidencias -> {log}")
        for w in all_warnings[:15]:
            print(f"   {w}")

    print(f"\n[OK] Resultados en {OUT_DIR}")


if __name__ == "__main__":
    main()