#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lr_range_test.py
---------------------
Genera la gráfica del LR Range Test (Leslie Smith, 2017) a partir del
fichero lr_loss_log.json producido por lr_range_test.py.

Dos paneles (fiel a Smith 2017, Sec. 3.3):
  - Panel superior : Accuracy vs LR  ← interpretación principal (Smith Fig. 3)
  - Panel inferior : Loss vs LR      ← información complementaria

Líneas verticales:
  - base_lr    (verde)  : donde la accuracy empieza a subir
  - max_lr     (naranja): donde la accuracy empieza a caer
  - optimal_lr (rojo)   : max_lr / 2, LR inicial sugerido para Adam (Smith Sec. 3.3)

Salida:
  data/LR_Range_Test/lr_range_test_plot.png

USO (desde ProyectoPython3.0/):
  python -m src.design.plot_lr_range_test
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────
LOG_PATH = Path("data/LR_Range_Test/lr_loss_log.json")
OUT_PATH = Path("data/LR_Range_Test/lr_range_test_plot.png")

# ── Cargar datos ───────────────────────────────────────────────────────
print(f"[INFO] Cargando {LOG_PATH}...")
with open(LOG_PATH, "r") as f:
    log = json.load(f)

lrs           = np.array(log["lrs"])
losses_raw    = np.array(log["losses_raw"])
losses_smooth = np.array(log["losses_smooth"])
accs_raw      = np.array(log["accs_raw"])
accs_smooth   = np.array(log["accs_smooth"])
base_lr       = log["base_lr"]
max_lr        = log["max_lr"]
optimal_lr    = log["optimal_lr"]

print(f"[INFO] base_lr    : {base_lr:.2e}")
print(f"[INFO] max_lr     : {max_lr:.2e}")
print(f"[INFO] optimal_lr : {optimal_lr:.2e}  (= max_lr / 2, LR inicial Adam)")

# ── Figura: 2 paneles ─────────────────────────────────────────────────
fig, (ax_acc, ax_loss) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle(
    "LR Range Test — CNN apnea STNF\n"
    "(Smith 2017, Sec. 3.3 — batch=2000, 1 época)",
    fontsize=13
)

# ── Panel superior: Accuracy ──────────────────────────────────────────
ax_acc.plot(lrs, accs_raw,
            color="steelblue", alpha=0.2, linewidth=1.0,
            label="Accuracy (raw)")
ax_acc.plot(lrs, accs_smooth,
            color="steelblue", linewidth=2.0,
            label="Accuracy (suavizada, β=0.9)")

# Zona útil entre base_lr y max_lr
ax_acc.axvspan(base_lr, max_lr,
               alpha=0.10, color="green",
               label=f"Zona útil [{base_lr:.1e}, {max_lr:.1e}]")

# Líneas verticales
ax_acc.axvline(x=base_lr, color="green", linestyle="--", linewidth=1.5,
               label=f"base_lr = {base_lr:.1e}  (accuracy empieza a subir)")
ax_acc.axvline(x=max_lr, color="orange", linestyle="--", linewidth=1.5,
               label=f"max_lr = {max_lr:.1e}  (accuracy empieza a caer)")
ax_acc.axvline(x=optimal_lr, color="red", linestyle="-", linewidth=2.0,
               label=f"optimal_lr = {optimal_lr:.1e}  (= max_lr / 2 → LR inicial Adam)")

ax_acc.set_ylabel("Accuracy", fontsize=11)
ax_acc.set_xscale("log")
ax_acc.grid(True, which="both", linestyle="--", alpha=0.4)
ax_acc.legend(fontsize=9, loc="lower left")
ax_acc.set_title("Panel principal — interpretación según Smith (Fig. 3)", fontsize=10)

# ── Panel inferior: Loss ──────────────────────────────────────────────
ax_loss.plot(lrs, losses_raw,
             color="coral", alpha=0.2, linewidth=1.0,
             label="Loss (raw)")
ax_loss.plot(lrs, losses_smooth,
             color="coral", linewidth=2.0,
             label="Loss (suavizada, β=0.9)")

ax_loss.axvspan(base_lr, max_lr,
                alpha=0.10, color="green")
ax_loss.axvline(x=base_lr,    color="green",  linestyle="--", linewidth=1.5)
ax_loss.axvline(x=max_lr,     color="orange", linestyle="--", linewidth=1.5)
ax_loss.axvline(x=optimal_lr, color="red",    linestyle="-",  linewidth=2.0)

ax_loss.set_ylabel("Loss (BCE)", fontsize=11)
ax_loss.set_xlabel("Learning Rate (escala log)", fontsize=11)
ax_loss.set_xscale("log")
ax_loss.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax_loss.grid(True, which="both", linestyle="--", alpha=0.4)
ax_loss.legend(fontsize=9, loc="upper left")
ax_loss.set_title("Panel complementario", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"[OK] Gráfica guardada en: {OUT_PATH}")
plt.close()